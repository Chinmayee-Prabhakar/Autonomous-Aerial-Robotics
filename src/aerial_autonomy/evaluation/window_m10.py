"""M10 image-observed rotating-arm flight with M1 dynamics and M7 references."""

import argparse
import json
import math
from pathlib import Path
import tomllib

import numpy as np
from PIL import Image

from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.window_m9 import load_m9_config
from aerial_autonomy.perception.window_geometry import StaticWindow,WindowModel
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m9_guidance import ImageWindowGuidance,TargetModel
from aerial_autonomy.perception.window_m9_render import RenderSettings,WindowImageRenderer
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld,WindowWall
from aerial_autonomy.perception.window_m10_arm import (ArmShape,RotatingArm,
    arm_clearance_m,swept_arm_clearance_m)
from aerial_autonomy.perception.window_m10_timing import (AngularTracker,
    choose_crossing,wrap_angle)
from aerial_autonomy.perception.window_m10_vision import PixelArmDetector,RotatingArmRenderer
from aerial_autonomy.planning.geometry import Workspace
from aerial_autonomy.simulation.dynamics import VehicleState,advance_state,world_acceleration
from aerial_autonomy.simulation.rotation import quaternion_to_matrix


def load_m10_config(path):
    with Path(path).open("rb") as f: raw=tomllib.load(f)
    _,vehicle,limits,camera,_,_,_=load_m9_config(raw["inputs"]["m9_config"])
    w=raw["window"]
    model=WindowModel(w["window_id"],w["width_m"],w["height_m"])
    window=StaticWindow(model,np.array(w["center_world_m"]),np.array(w["right_world"]),
                        np.array(w["up_world"]),w["frame_thickness_m"])
    wall=WindowWall(window,w["outer_width_m"],w["outer_height_m"])
    space=raw["workspace"]
    if (abs(space["vehicle_radius_m"]-.18)>1e-12 or
            abs(space["actual_clearance_m"]-.07)>1e-12 or
            space["planning_clearance_m"]<space["actual_clearance_m"]):
        raise ValueError("accepted M9 0.25 m flight envelope must be preserved")
    world=WindowCourseWorld(Workspace(tuple(space["minimum"]),tuple(space["maximum"])),
                            (wall,),space["vehicle_radius_m"],space["actual_clearance_m"])
    route=raw["far_side_connection"]["waypoints_world_m"]
    if len(route)<3 or min(world.segment_clearance_m(a,b)
                           for a,b in zip(route[:-1],route[1:]))<=0:
        raise ValueError("far-side connecting route must clear the stationary frame")
    a=raw["arm"]
    shape=ArmShape(a["pivot_right_m"],a["pivot_up_m"],a["length_m"],a["arm_radius_m"])
    arm=RotatingArm(window,shape,a["initial_angle_rad"],a["angular_velocity_rad_s"])
    return raw,vehicle,limits,camera,world,arm


class RotatingWindowMission:
    """Truth is used only for RGB generation, plant collision and scoring.

    Operational timing receives pixels, M8/M9 image pose, known dimensions,
    ideal onboard vehicle feedback, and M7 trajectory bounds. It does not
    receive ``arm.initial_angle_rad`` or ``arm.angular_velocity_rad_s``.
    """

    def __init__(self,raw,vehicle,limits,camera,world,arm,seed=101,
                 initial_angle_rad=None,angular_velocity_rad_s=None,
                 noise_std_rgb=None,dropout_probability=0.,lateral_offset_m=0.,
                 capture_frames=False,max_wait_s=None):
        self.raw=raw;self.vehicle=vehicle;self.limits=limits;self.camera=camera
        self.world=world
        self.arm=RotatingArm(arm.window,arm.shape,
            arm.initial_angle_rad if initial_angle_rad is None else initial_angle_rad,
            arm.angular_velocity_rad_s if angular_velocity_rad_s is None else angular_velocity_rad_s)
        declared=raw["arm"]
        self.known_arm_length_m=declared["length_m"]
        self.known_arm_radius_m=declared["arm_radius_m"]
        self.rng=np.random.default_rng(seed)
        self.seed=seed;self.dropout_probability=dropout_probability
        self.noise_std_rgb=(raw["render"]["noise_std_rgb"] if noise_std_rgb is None
                            else noise_std_rgb)
        self.max_wait_s=(raw["guidance"]["max_wait_s"] if max_wait_s is None else max_wait_s)
        self.capture_frames=capture_frames
        target=world.walls[0].window.model
        self.target_id=target.window_id
        color={self.target_id:tuple(raw["window"]["color_rgb"])}
        self.renderer=RotatingArmRenderer(WindowImageRenderer(camera,world,color),self.arm)
        self.guidance=ImageWindowGuidance(camera,ColorWindowDetector(color),{
            self.target_id:TargetModel(target,world.walls[0].window.frame_thickness_m,
                world.walls[0].outer_width_m,world.walls[0].outer_height_m,"positive_normal")},
            world.workspace,vehicle.vehicle,limits,world.vehicle_radius_m,
            raw["workspace"]["planning_clearance_m"],
            raw["guidance"]["approach_distance_m"],raw["guidance"]["exit_distance_m"])
        self.detector=PixelArmDetector()
        self.tracker=AngularTracker(raw["guidance"]["max_observation_age_s"])
        self.controller=CascadedHoverController(vehicle.vehicle,vehicle.controller)
        self.state=VehicleState.at_rest()
        self.state.position_m=np.array(raw["guidance"]["start_world_m"],dtype=float)
        self.state.position_m[1]+=lateral_offset_m
        self.time_s=0.;self.next_image_s=0.;self.last_capture_s=-math.inf
        self.phase="ACQUIRE_WINDOW";self.failure=None
        self.last_fix=None;self.last_arm_detection=None
        self.initial_wait_s=None;self.commit_time_s=None
        self.hold_position_m=None
        self.selected_choice=None;self.crossing_events=[]
        self.envelope_entry_time_s=None;self.envelope_exit_time_s=None
        self.observations=[];self.truth_scoring=[];self.predictions=[];self.choices=[]
        self.states=[];self.frames=[];self.references=[];self.transitions=[{"time_s":0.,"state":self.phase}]
        self.minimum_arm_clearance_m=math.inf;self.minimum_frame_clearance_m=math.inf
        self.peak_tracking_error_m=0.;self.peak_speed_m_s=0.;self.peak_acceleration_m_s2=0.
        self.min_thrust_n=math.inf;self.max_thrust_n=0.;self.peak_tilt_deg=0.

    def transition(self,name):
        if name!=self.phase:
            self.phase=name;self.transitions.append({"time_s":float(self.time_s),"state":name})

    def observe(self,force=False):
        if not force and self.time_s+1e-8<self.next_image_s:
            return None
        settings=RenderSettings(self.noise_std_rgb,self.raw["render"]["illumination"],
                                self.raw["render"]["blur"])
        image=self.renderer.render(self.state,self.time_s,settings,self.rng)
        dropped=bool(self.rng.random()<self.dropout_probability)
        sensed=np.zeros_like(image) if dropped else image
        fix=self.guidance.perceive(sensed,self.state,self.target_id,self.time_s)
        detection=self.detector.detect(sensed,fix,self.time_s)
        self.tracker.update(detection)
        if detection.valid:self.last_arm_detection=detection
        estimate=self.tracker.estimate(self.time_s)
        if fix.valid:self.last_fix=fix
        record={"time_s":float(self.time_s),"guidance_state":self.phase,
                "dropout":dropped,"window_valid":fix.valid,"window_reason":fix.reason,
                "window_center_estimated_m":(fix.estimated_window.center_world_m.tolist()
                                             if fix.valid else None),
                "arm_valid":detection.valid,"arm_reason":detection.reason,
                "detected_angle_rad":detection.angle_rad,
                "detected_pivot_local_m":detection.pivot_local_m,
                "detected_tip_local_m":detection.tip_local_m,
                "detection_confidence":detection.confidence,"shaft_pixel_count":detection.shaft_pixel_count,
                "tracker_valid":estimate.valid,"tracker_reason":estimate.reason,
                "estimated_angle_rad":estimate.angle_rad,
                "estimated_rate_rad_s":estimate.angular_velocity_rad_s,
                "angle_bound_rad":estimate.angle_uncertainty_rad if estimate.valid else None,
                "rate_bound_rad_s":estimate.rate_uncertainty_rad_s if estimate.valid else None,
                "observation_age_s":estimate.observation_age_s if np.isfinite(estimate.observation_age_s) else None}
        self.observations.append(record)
        # Independent scoring uses simulator truth only after the operational update.
        true_angle=self.arm.angle_at(self.time_s)
        self.truth_scoring.append({"time_s":float(self.time_s),"true_angle_rad":true_angle,
            "true_rate_rad_s":self.arm.angular_velocity_rad_s,
            "detection_error_rad":(wrap_angle(detection.angle_rad-true_angle)
                                   if detection.valid else None),
            "tracking_angle_error_rad":(wrap_angle(estimate.angle_rad-true_angle)
                                        if estimate.valid else None),
            "tracking_rate_error_rad_s":(estimate.angular_velocity_rad_s-
                self.arm.angular_velocity_rad_s if estimate.valid else None)})
        if estimate.valid:
            for horizon in (.5,1.,2.,3.):
                predicted,bound=estimate.predict(self.time_s+horizon)
                self.predictions.append({"observation_time_s":float(self.time_s),
                    "horizon_s":horizon,"predicted_angle_rad":predicted,
                    "bound_rad":bound,"true_future_angle_rad":self.arm.angle_at(self.time_s+horizon),
                    "error_rad":wrap_angle(predicted-self.arm.angle_at(self.time_s+horizon))})
        if self.capture_frames and self.time_s-self.last_capture_s>=.6:
            self.frames.append({"time_s":float(self.time_s),"image":image.copy(),
                "detection":detection,"estimate":estimate,"phase":self.phase,
                "selected":self.selected_choice})
            self.last_capture_s=self.time_s
        self.next_image_s=self.time_s+self.raw["guidance"]["image_period_s"]
        return fix,estimate

    def step(self,reference):
        dt=self.raw["guidance"]["step_s"]
        control=self.controller.command(self.state,reference.position_m,0.,
                                         reference.velocity_m_s,reference.acceleration_m_s2)
        acceleration=world_acceleration(self.state,control.wrench,self.vehicle.vehicle)
        tracking=float(np.linalg.norm(self.state.position_m-reference.position_m))
        tilt=float(np.degrees(np.arccos(np.clip(
            quaternion_to_matrix(self.state.quaternion_bw)[2,2],-1.,1.))))
        future=advance_state(self.state,control.wrench,self.vehicle.vehicle,dt)
        frame_clearance=self.world.segment_clearance_m(self.state.position_m,future.position_m)
        arm_clearance=swept_arm_clearance_m(self.state.position_m,future.position_m,
            self.time_s,self.time_s+dt,self.arm,self.world.envelope_m)
        self.minimum_frame_clearance_m=min(self.minimum_frame_clearance_m,frame_clearance)
        self.minimum_arm_clearance_m=min(self.minimum_arm_clearance_m,arm_clearance)
        self.peak_tracking_error_m=max(self.peak_tracking_error_m,tracking)
        self.peak_speed_m_s=max(self.peak_speed_m_s,float(np.linalg.norm(self.state.velocity_m_s)))
        self.peak_acceleration_m_s2=max(self.peak_acceleration_m_s2,float(np.linalg.norm(acceleration)))
        self.min_thrust_n=min(self.min_thrust_n,control.wrench.thrust_n)
        self.max_thrust_n=max(self.max_thrust_n,control.wrench.thrust_n)
        self.peak_tilt_deg=max(self.peak_tilt_deg,tilt)
        self.states.append({"time_s":float(self.time_s),"position_m":self.state.position_m.copy(),
            "velocity_m_s":self.state.velocity_m_s.copy(),"quaternion_bw":self.state.quaternion_bw.copy(),
            "reference_position_m":reference.position_m.copy(),"acceleration_m_s2":acceleration.copy(),
            "tracking_error_m":tracking,"thrust_n":control.wrench.thrust_n,"tilt_deg":tilt,
            "frame_clearance_m":frame_clearance,"arm_clearance_m":arm_clearance,"phase":self.phase})
        if frame_clearance<=0 or arm_clearance<=0:
            self.failure="actual_frame_collision" if frame_clearance<=0 else "actual_arm_collision"
            self.transition("ABORT")
            return False
        if (self.peak_speed_m_s>self.limits.max_speed_m_s+1e-6 or
            self.peak_acceleration_m_s2>self.limits.max_acceleration_m_s2+1e-6 or
            self.max_thrust_n>self.limits.max_thrust_n+1e-6 or
            self.min_thrust_n<self.limits.min_thrust_n-1e-6 or
            self.peak_tilt_deg>self.limits.max_tilt_deg+1e-6):
            self.failure="actual_dynamic_limit";self.transition("ABORT");return False
        before=self.world.walls[0].plane_coordinate_m(self.state.position_m)
        after=self.world.walls[0].plane_coordinate_m(future.position_m)
        threshold=self.world.walls[0].window.frame_thickness_m/2+self.world.envelope_m
        if self.commit_time_s is not None and before>threshold>=after:
            self.envelope_entry_time_s=self.time_s+dt
        if self.commit_time_s is not None and before>-threshold>=after:
            self.envelope_exit_time_s=self.time_s+dt
        if before>0>=after:
            self.crossing_events.append({"plane_crossing_time_s":self.time_s+dt,
                "commit_time_s":self.commit_time_s,"direction":"outbound"})
        self.state=future;self.time_s+=dt
        return True

    def fly(self,trajectory,phase):
        self.transition(phase)
        start=self.time_s
        horizon=trajectory.duration_s+self.raw["guidance"]["settle_s"]
        count=math.ceil(horizon/self.raw["guidance"]["step_s"])
        for i in range(count):
            self.observe()
            reference=trajectory.evaluate(min(i*self.raw["guidance"]["step_s"],trajectory.duration_s))
            if not self.step(reference):return False
        return True

    def hold_step(self):
        from aerial_autonomy.trajectory.quintic import Reference
        self.observe()
        return self.step(Reference(self.hold_position_m.copy(),np.zeros(3),np.zeros(3)))

    def run(self):
        self.observe(force=True)
        if self.last_fix is None:
            self.failure="window_not_detected";self.transition("ABORT");return self.metrics()
        entry,_=self.guidance.entry_exit(self.last_fix)
        trajectory,report=self.guidance.plan_segment(self.state,self.last_fix,entry)
        self.references.append({"phase":"APPROACH","report":report})
        if trajectory is None:
            self.failure="approach_"+report["reason"];self.transition("ABORT");return self.metrics()
        if not self.fly(trajectory,"APPROACH"):
            return self.metrics()
        self.transition("ALIGN_AND_HOLD")
        self.initial_wait_s=self.time_s
        self.hold_position_m=self.state.position_m.copy()
        while self.time_s-self.initial_wait_s<self.max_wait_s:
            observed=self.observe()
            if observed is not None:
                _,estimate=observed
                if estimate.valid and self.last_fix is not None and self.last_arm_detection is not None:
                    _,exit_point=self.guidance.entry_exit(self.last_fix)
                    candidate,report=self.guidance.plan_segment(self.state,self.last_fix,exit_point)
                    if candidate is not None:
                        measured=self.last_arm_detection
                        pivot=np.asarray(measured.pivot_local_m)
                        extent=float(np.linalg.norm(np.asarray(measured.tip_local_m)-pivot))
                        planning_shape=ArmShape(float(pivot[0]),float(pivot[1]),
                            max(self.known_arm_length_m,extent+.05),self.known_arm_radius_m)
                        choice=choose_crossing(candidate,self.time_s,estimate,
                            self.last_fix.estimated_window,planning_shape,self.world.envelope_m,
                            self.raw["guidance"]["prediction_horizon_s"])
                        self.choices.append({"time_s":float(self.time_s),"valid":choice.valid,
                            "reason":choice.reason,"commit_time_s":choice.commit_time_s,
                            "plane_time_s":choice.predicted_plane_time_s,
                            "predicted_angle_rad":choice.predicted_angle_rad,
                            "predicted_min_clearance_m":choice.predicted_min_clearance_m})
                        if choice.valid and choice.commit_time_s<=self.time_s+.001:
                            self.selected_choice=choice
                            self.references.append({"phase":"COMMIT_TO_TRAVERSE","report":report,
                                "estimated_start_time_s":float(self.time_s),
                                "estimated_waypoints_m":[list(x) for x in candidate.waypoints_m]})
                            self.commit_time_s=self.time_s
                            if not self.fly(candidate,"COMMIT_TO_TRAVERSE"):
                                return self.metrics()
                            self.transition("CONFIRM_EXIT")
                            far=self.world.walls[0].plane_coordinate_m(self.state.position_m)
                            if len(self.crossing_events)!=1 or far>=-(.08+self.world.envelope_m):
                                self.failure="crossing_not_confirmed";self.transition("ABORT")
                            else:self.transition("COMPLETE")
                            return self.metrics()
            if not self.hold_step():return self.metrics()
        self.failure="no_safe_interval_within_wait_limit";self.transition("ABORT")
        return self.metrics()

    def metrics(self):
        valid=[x for x in self.truth_scoring if x["tracking_angle_error_rad"] is not None]
        def rmse(key):
            return float(np.sqrt(np.mean([x[key]**2 for x in valid]))) if valid else None
        return {"seed":self.seed,"success":self.phase=="COMPLETE" and self.failure is None,
            "failure_reason":self.failure,"crossing_count":len(self.crossing_events),
            "collision_count":int(self.failure in ("actual_frame_collision","actual_arm_collision")),
            "window_detection_count":sum(x["window_valid"] for x in self.observations),
            "arm_detection_count":sum(x["arm_valid"] for x in self.observations),
            "observation_count":len(self.observations),"missed_observations":self.tracker.misses,
            "angle_rmse_deg":math.degrees(rmse("tracking_angle_error_rad")) if valid else None,
            "rate_rmse_rad_s":rmse("tracking_rate_error_rad_s"),
            "waiting_s":(self.commit_time_s-self.initial_wait_s if self.commit_time_s is not None
                          else self.time_s-self.initial_wait_s if self.initial_wait_s is not None else 0.),
            "commit_time_s":self.commit_time_s,
            "predicted_plane_time_s":(self.selected_choice.predicted_plane_time_s
                                       if self.selected_choice else None),
            "actual_plane_time_s":(self.crossing_events[0]["plane_crossing_time_s"]
                                    if self.crossing_events else None),
            "envelope_entry_time_s":self.envelope_entry_time_s,
            "envelope_exit_time_s":self.envelope_exit_time_s,
            "predicted_arm_phase_rad":(self.selected_choice.predicted_angle_rad
                                       if self.selected_choice else None),
            "minimum_predicted_clearance_m":(self.selected_choice.predicted_min_clearance_m
                                              if self.selected_choice else None),
            "minimum_actual_arm_clearance_m":float(self.minimum_arm_clearance_m),
            "minimum_actual_frame_clearance_m":float(self.minimum_frame_clearance_m),
            "maximum_tracking_error_m":self.peak_tracking_error_m,
            "maximum_speed_m_s":self.peak_speed_m_s,
            "maximum_acceleration_m_s2":self.peak_acceleration_m_s2,
            "minimum_thrust_n":self.min_thrust_n,"maximum_thrust_n":self.max_thrust_n,
            "maximum_tilt_deg":self.peak_tilt_deg,"duration_s":self.time_s}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/window_m10.toml")
    parser.add_argument("--output-dir",default="outputs/window_m10")
    args=parser.parse_args(argv)
    raw,vehicle,limits,camera,world,arm=load_m10_config(args.config)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    result=run_experiments(raw,vehicle,limits,camera,world,arm,output)
    print(json.dumps(result,indent=2))


def _write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _save_mission(mission,folder):
    folder=Path(folder);(folder/"operational").mkdir(parents=True,exist_ok=True)
    (folder/"evaluation").mkdir(exist_ok=True)
    _write_json(folder/"operational"/"observations.json",mission.observations)
    _write_json(folder/"operational"/"prediction_records.json",mission.predictions)
    _write_json(folder/"operational"/"crossing_choices.json",mission.choices)
    _write_json(folder/"operational"/"reference_plans.json",mission.references)
    _write_json(folder/"operational"/"guidance_transitions.json",mission.transitions)
    _write_json(folder/"evaluation"/"perception_truth_scoring.json",mission.truth_scoring)
    _write_json(folder/"evaluation"/"crossings.json",mission.crossing_events)
    _write_json(folder/"evaluation"/"metrics.json",mission.metrics())
    rows=mission.states
    if rows:
        np.savez_compressed(folder/"evaluation"/"flight_trace.npz",
            time_s=np.array([x["time_s"] for x in rows]),
            position_m=np.array([x["position_m"] for x in rows]),
            velocity_m_s=np.array([x["velocity_m_s"] for x in rows]),
            quaternion_bw=np.array([x["quaternion_bw"] for x in rows]),
            reference_position_m=np.array([x["reference_position_m"] for x in rows]),
            acceleration_m_s2=np.array([x["acceleration_m_s2"] for x in rows]),
            arm_clearance_m=np.array([x["arm_clearance_m"] for x in rows]),
            frame_clearance_m=np.array([x["frame_clearance_m"] for x in rows]),
            tracking_error_m=np.array([x["tracking_error_m"] for x in rows]),
            thrust_n=np.array([x["thrust_n"] for x in rows]),
            tilt_deg=np.array([x["tilt_deg"] for x in rows]),
            phase=np.array([x["phase"] for x in rows]))


def _detector_study(hero,folder):
    cases=[];state=VehicleState.at_rest();rng=np.random.default_rng(83)
    for x,y,z in ((17.4,0,1.6),(17.8,.18,1.7),(18.2,-.15,1.5)):
        state.position_m=np.array([x,y,z])
        for index,angle in enumerate(np.linspace(-math.pi,math.pi,8,endpoint=False)):
            # Study truth varies only in the simulator renderer.
            test_arm=RotatingArm(hero.arm.window,hero.arm.shape,float(angle),0.)
            renderer=RotatingArmRenderer(hero.renderer.base,test_arm)
            image=renderer.render(state,0.,RenderSettings(3.),rng)
            fix=hero.guidance.perceive(image,state,hero.target_id,0.)
            det=hero.detector.detect(image,fix,0.)
            record={"view_position_m":state.position_m.tolist(),"true_angle_rad":float(angle),
                "window_valid":fix.valid,"arm_valid":det.valid,"failure_reason":det.reason,
                "angle_error_deg":math.degrees(wrap_angle(det.angle_rad-angle)) if det.valid else None,
                "pivot_error_m":float(np.linalg.norm(np.array(det.pivot_local_m)-
                    np.array([test_arm.shape.pivot_right_m,test_arm.shape.pivot_up_m]))) if det.valid else None}
            cases.append(record)
            if index in (0,2,4,6) and x==17.4:
                Image.fromarray(image).save(folder/f"detector_angle_{index}.png")
    _write_json(folder/"static_detector_study.json",cases)
    return cases


def run_experiments(raw,vehicle,limits,camera,world,arm,output):
    from aerial_autonomy.visualization.window_m10_plot import (save_camera_gif,
        save_flight_gif,save_overview,save_tracking,save_prediction,save_timing,
        save_control,save_abort)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    visuals=output/"figures";visuals.mkdir(exist_ok=True)
    hero=RotatingWindowMission(raw,vehicle,limits,camera,world,arm,
                               raw["experiments"]["hero_seed"],capture_frames=True)
    hero_metrics=hero.run();_save_mission(hero,output/"trials"/"hero")
    if hero_metrics["success"]:
        save_camera_gif(hero,visuals/"rotating_arm_camera.gif")
        timeline=save_flight_gif(hero,visuals/"quadrotor_flight_3d.gif")
        closeup_timeline=save_flight_gif(hero,visuals/"quadrotor_crossing_closeup.gif",closeup=True)
        save_overview(hero,visuals/"trajectory_overview.png")
        save_tracking(hero,visuals/"arm_tracking.png")
        save_prediction(hero,visuals/"motion_prediction.png")
        save_timing(hero,visuals/"crossing_timing.png")
        save_control(hero,visuals/"clearance_and_control.png")
        _write_json(output/"animation_timeline.json",{"flight_frame_times_s":timeline,
            "closeup_frame_times_s":closeup_timeline,
            "camera_frame_times_s":[x["time_s"] for x in hero.frames]})
        frame_folder=output/"selected_frames";frame_folder.mkdir(exist_ok=True)
        for i in np.linspace(0,len(hero.frames)-1,min(6,len(hero.frames)),dtype=int):
            Image.fromarray(hero.frames[int(i)]["image"]).save(frame_folder/f"camera_{int(i):03d}.png")
    study=_detector_study(hero,visuals)
    trials=[];exp=raw["experiments"]
    for index,seed in enumerate(exp["trial_seeds"]):
        mission=RotatingWindowMission(raw,vehicle,limits,camera,world,arm,seed,
            initial_angle_rad=exp["initial_phases_rad"][index],
            angular_velocity_rad_s=exp["angular_velocities_rad_s"][index],
            noise_std_rgb=exp["noise_std_rgb"][index],
            dropout_probability=exp["dropout_probability"][index],
            lateral_offset_m=exp["initial_lateral_offsets_m"][index])
        result=mission.run();result["condition"]={"initial_phase_rad":exp["initial_phases_rad"][index],
            "angular_velocity_rad_s":exp["angular_velocities_rad_s"][index],
            "noise_std_rgb":exp["noise_std_rgb"][index],
            "dropout_probability":exp["dropout_probability"][index],
            "initial_lateral_offset_m":exp["initial_lateral_offsets_m"][index]}
        trials.append(result);_save_mission(mission,output/"trials"/f"seed_{seed}")
    unsafe=RotatingWindowMission(raw,vehicle,limits,camera,world,arm,211,
        initial_angle_rad=math.pi,angular_velocity_rad_s=0.,max_wait_s=3.5)
    unsafe_metrics=unsafe.run();_save_mission(unsafe,output/"trials"/"unsafe_static_arm")
    save_abort(unsafe,visuals/"unsafe_abort.png")
    _write_json(output/"trial_summary.json",trials)
    _write_json(output/"config_snapshot.json",raw)
    successful=[x for x in trials if x["success"]]
    valid_study=[x for x in study if x["arm_valid"]]
    gates=raw["acceptance"]
    summary={"hero":hero_metrics,"trial_count":len(trials),
        "trial_successes":len(successful),"trial_failures":[x for x in trials if not x["success"]],
        "static_detector_views":len(study),"static_detector_valid":len(valid_study),
        "mean_static_angle_error_deg":float(np.mean([abs(x["angle_error_deg"]) for x in valid_study])),
        "unsafe_case":unsafe_metrics,
        "minimum_successful_arm_clearance_m":min((x["minimum_actual_arm_clearance_m"] for x in successful),default=None),
        "minimum_successful_frame_clearance_m":min((x["minimum_actual_frame_clearance_m"] for x in successful),default=None)}
    route=raw["far_side_connection"]["waypoints_world_m"]
    summary["far_side_connection_min_static_clearance_m"]=min(
        world.segment_clearance_m(a,b) for a,b in zip(route[:-1],route[1:]))
    summary["predeclared_gates_pass"]=(hero_metrics["success"] and
        len(successful)>=gates["minimum_successful_trials"] and
        all(x["crossing_count"]==1 and x["collision_count"]==0 for x in successful) and
        summary["minimum_successful_arm_clearance_m"]>gates["minimum_actual_arm_clearance_m"] and
        summary["minimum_successful_frame_clearance_m"]>gates["minimum_actual_frame_clearance_m"] and
        hero_metrics["angle_rmse_deg"]<gates["maximum_angle_error_deg"] and
        hero_metrics["rate_rmse_rad_s"]<gates["maximum_rate_error_rad_s"] and
        not unsafe_metrics["success"] and unsafe_metrics["crossing_count"]==0 and
        unsafe_metrics["collision_count"]==0)
    _write_json(output/"aggregate_metrics.json",summary)
    _write_json(output/"visual_manifest.json",[str(x) for x in sorted(visuals.iterdir())])
    return summary


if __name__=="__main__":main()
