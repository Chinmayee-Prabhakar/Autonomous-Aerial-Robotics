"""M9 image-detected static-window flight and independent truth scoring."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

import numpy as np

from aerial_autonomy.config import load_config
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m9_guidance import (ImageWindowGuidance,
    ImagePoseFix, NavigationPolicy, NavigationState, TargetModel)
from aerial_autonomy.perception.window_m9_render import RenderSettings, WindowImageRenderer
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld, WindowWall
from aerial_autonomy.perception.window_pose import reproject_estimate
from aerial_autonomy.planning.geometry import COLLISION_TOLERANCE_M, World, Workspace
from aerial_autonomy.simulation.dynamics import VehicleState, advance_state, world_acceleration
from aerial_autonomy.simulation.rotation import euler_to_quaternion, quaternion_to_matrix
from aerial_autonomy.trajectory.quintic import TrajectoryLimits, generate_trajectory


def load_m9_config(path: str | Path):
    with Path(path).open("rb") as stream:
        raw=tomllib.load(stream)
    vehicle=load_config(raw["inputs"]["vehicle_config"])
    with Path(raw["inputs"]["m7_trajectory_config"]).open("rb") as stream:
        m7=tomllib.load(stream)
    limits=TrajectoryLimits(**m7["trajectory"])
    camera_raw=raw["camera"]
    camera=CameraCalibration(camera_raw["fx_px"],camera_raw["fy_px"],camera_raw["cx_px"],
        camera_raw["cy_px"],camera_raw["width_px"],camera_raw["height_px"],
        np.asarray(camera_raw["rotation_camera_to_body"]),
        np.asarray(camera_raw["translation_camera_in_body_m"]))
    w=raw["workspace"]
    workspace=Workspace(tuple(w["minimum"]),tuple(w["maximum"]))
    if (abs(w["vehicle_radius_m"]-.18)>1e-12 or
            abs(w["actual_clearance_m"]-.07)>1e-12 or
            w["planning_clearance_m"]<w["actual_clearance_m"]):
        raise ValueError("M9 must preserve the accepted 0.25 m actual envelope")
    walls=[]; targets={}; colors={}
    for item in raw["windows"]:
        model=WindowModel(item["window_id"],item["width_m"],item["height_m"])
        window=StaticWindow(model,np.asarray(item["center_world_m"]),
            np.asarray(item["right_world"]),np.asarray(item["up_world"]),
            item["frame_thickness_m"])
        walls.append(WindowWall(window,item["outer_width_m"],item["outer_height_m"]))
        targets[model.window_id]=TargetModel(model,item["frame_thickness_m"],
            item["outer_width_m"],item["outer_height_m"],"positive_normal")
        colors[model.window_id]=tuple(item["color_rgb"])
    if len(walls)<3 or len(targets)!=len(walls):
        raise ValueError("M9 scene needs three uniquely identified windows")
    sequence=raw["mission"]["ordered_window_ids"]
    if set(sequence)!=set(targets) or len(sequence)!=len(targets):
        raise ValueError("mission order must visit every window exactly once")
    world=WindowCourseWorld(workspace,tuple(walls),w["vehicle_radius_m"],w["actual_clearance_m"])
    return raw,vehicle,limits,camera,world,targets,colors


@dataclass(frozen=True)
class FlightCondition:
    name: str
    seed: int
    noise_std_rgb: float
    illumination: float
    blur: bool
    dropout_probability: float
    initial_lateral_offset_m: float = 0.


class WindowMission:
    """Simulator orchestrator. Truth is confined to rendering, physics and scoring."""

    def __init__(self,raw,vehicle,limits,camera,true_world,targets,colors,
                 condition: FlightCondition,mission_ids=None,reverse=False,
                 capture_frames=False):
        self.raw=raw;self.vehicle=vehicle;self.limits=limits;self.camera=camera
        self.true_world=true_world
        self.colors=colors
        self.condition=condition
        self.mission_ids=list(mission_ids or raw["mission"]["ordered_window_ids"])
        self.reverse=reverse
        self.capture_frames=capture_frames
        if reverse:
            if len(self.mission_ids)!=1:
                raise ValueError("M9 reverse validation uses one static window")
            selected=self.mission_ids[0]
            targets=dict(targets)
            old=targets[selected]
            targets[selected]=TargetModel(old.model,old.frame_thickness_m,old.outer_width_m,
                                          old.outer_height_m,"negative_normal")
        self.renderer=WindowImageRenderer(camera,true_world,colors)
        self.guidance=ImageWindowGuidance(camera,ColorWindowDetector(colors),targets,
            true_world.workspace,vehicle.vehicle,limits,true_world.vehicle_radius_m,
            raw["workspace"]["planning_clearance_m"],
            raw["guidance"]["approach_distance_m"],raw["guidance"]["exit_distance_m"])
        g=raw["guidance"]
        self.policy=NavigationPolicy(g["image_period_s"],g["max_loss_s"],
            g["max_tracking_error_m"],g["final_segment_error_m"],g["max_pose_jump_m"])
        self.controller=CascadedHoverController(vehicle.vehicle,vehicle.controller)
        start=np.asarray(raw["mission"]["start_world_m"],dtype=float).copy()
        if reverse:
            start=np.array([8.,0.,0.])
        else:
            start[1]+=condition.initial_lateral_offset_m
        self.state=VehicleState.at_rest()
        self.state.position_m=start
        self.yaw=np.pi if reverse else 0.
        self.state.quaternion_bw=euler_to_quaternion(0.,0.,self.yaw)
        self.time_s=0.
        self.next_image_s=0.
        self.last_capture_s=-2.
        self.rng=np.random.default_rng(condition.seed)
        self.state_name=NavigationState.TAKEOFF
        self.transitions=[{"time_s":0.,"state":self.state_name,"target":None}]
        self.perception=[]
        self.scoring=[]
        self.crossings=[]
        self.references=[]
        self.actual=[]
        self.frames=[]
        self.failure=None
        self.recoveries=0
        self.minimum_clearance_m=float("inf")
        self.peak_tracking_error_m=0.
        self.peak_speed_m_s=0.
        self.peak_acceleration_m_s2=0.
        self.peak_thrust_n=0.
        self.minimum_thrust_n=float("inf")
        self.peak_tilt_deg=0.
        self.reference_reports=[]
        self.last_estimated_center_by_id={}

    def _transition(self,name,target=None):
        self.state_name=name
        self.transitions.append({"time_s":float(self.time_s),"state":name,"target":target})

    def _settings(self):
        return RenderSettings(self.condition.noise_std_rgb,self.condition.illumination,
                              self.condition.blur)

    def _observe(self,target_id: str,force=False):
        if not force and self.time_s+1e-9<self.next_image_s:
            return None
        image=self.renderer.render(self.state,self._settings(),self.rng)
        dropped=bool(self.rng.random()<self.condition.dropout_probability)
        observed=np.zeros_like(image) if dropped else image
        fix=self.guidance.perceive(observed,self.state,target_id,self.time_s)
        if fix.valid:
            center=fix.estimated_window.center_world_m
            prior=self.last_estimated_center_by_id.get(target_id)
            if prior is not None and np.linalg.norm(center-prior)>self.policy.max_pose_jump_m:
                fix=ImagePoseFix(target_id,False,"temporal_pose_jump",fix.detection,
                                 fix.estimate,None)
            else:
                self.last_estimated_center_by_id[target_id]=center.copy()
        self.next_image_s=self.time_s+self.policy.image_period_s
        det=fix.detection
        record={"time_s":float(self.time_s),"target_id":target_id,"state":self.state_name,
                "frame_dropout":dropped,"detected":det.valid,"detection_reason":det.reason,
                "detection_confidence":det.confidence,"color_pixel_count":det.color_pixel_count,
                "detected_corners_uv":det.corners_uv.tolist() if det.valid else None,
                "pose_valid":fix.valid,"pose_reason":fix.reason,
                "estimated_center_world_m":(fix.estimated_window.center_world_m.tolist()
                                            if fix.valid else None),
                "relative_translation_camera_m":(fix.estimate.translation_window_in_camera_m.tolist()
                    if fix.estimate is not None and fix.estimate.valid else None),
                "reprojection_rmse_px":(fix.estimate.reprojection_rmse_px
                    if fix.estimate is not None and fix.estimate.valid else None)}
        self.perception.append(record)
        # Evaluation truth is read only after perception has completed.
        truth_wall=self.true_world.wall_for_id(target_id)
        ideal,_,inside=self.camera.project(self.camera.world_to_camera(
            self.state,truth_wall.window.world_corners_m()))
        corner_error=(float(np.sqrt(np.mean(np.sum((det.corners_uv-ideal)**2,axis=1))))
                      if det.valid and np.all(inside) else None)
        translation_error=(float(np.linalg.norm(fix.estimated_window.center_world_m-
            truth_wall.window.center_world_m)) if fix.valid else None)
        rotation_error=(float(np.degrees(np.arccos(np.clip((np.trace(
            fix.estimated_window.rotation_window_to_world@
            truth_wall.window.rotation_window_to_world.T)-1)/2,-1,1))))
            if fix.valid else None)
        self.scoring.append({"time_s":float(self.time_s),"target_id":target_id,
            "corner_rmse_px":corner_error,"translation_error_m":translation_error,
            "orientation_error_deg":rotation_error,
            "valid":fix.valid,"failure_reason":fix.reason})
        if self.capture_frames and (self.time_s-self.last_capture_s>=.8 or force):
            self.frames.append({"time_s":float(self.time_s),"target_id":target_id,
                "state":self.state_name,"image":image.copy(),"detection":det,
                "pose_valid":fix.valid,"reprojection_rmse_px":record["reprojection_rmse_px"],
                "estimated_reproject_uv":(reproject_estimate(fix.estimate,
                    self.guidance.targets[target_id].model,self.camera)
                    if fix.valid else None)})
            self.last_capture_s=self.time_s
        return fix

    def _hold_reacquire(self,target_id: str):
        self._transition(NavigationState.HOLD,target_id)
        self.recoveries+=1
        dt=float(self.raw["guidance"].get("step_s",.01))
        hold_position=self.state.position_m.copy()
        count=int(np.ceil(self.raw["guidance"]["reacquire_s"]/dt))
        base_yaw=self.yaw
        for offset_deg in (0.,-15.,15.,-30.,30.,-45.,45.):
            self.yaw=base_yaw+float(np.deg2rad(offset_deg))
            for _ in range(count):
                fix=self._observe(target_id)
                if fix is not None and fix.valid:
                    return fix
                control=self.controller.command(self.state,hold_position,self.yaw)
                acceleration=world_acceleration(self.state,control.wrench,self.vehicle.vehicle)
                tilt=float(np.degrees(np.arccos(np.clip(
                    quaternion_to_matrix(self.state.quaternion_bw)[2,2],-1,1))))
                self.peak_speed_m_s=max(self.peak_speed_m_s,float(np.linalg.norm(self.state.velocity_m_s)))
                self.peak_acceleration_m_s2=max(self.peak_acceleration_m_s2,float(np.linalg.norm(acceleration)))
                self.peak_thrust_n=max(self.peak_thrust_n,control.wrench.thrust_n)
                self.minimum_thrust_n=min(self.minimum_thrust_n,control.wrench.thrust_n)
                self.peak_tilt_deg=max(self.peak_tilt_deg,tilt)
                self.peak_tracking_error_m=max(self.peak_tracking_error_m,
                    float(np.linalg.norm(self.state.position_m-hold_position)))
                self.actual.append({"time_s":float(self.time_s),"position_m":self.state.position_m.copy(),
                    "velocity_m_s":self.state.velocity_m_s.copy(),"quaternion_bw":self.state.quaternion_bw.copy(),
                    "reference_position_m":hold_position.copy(),
                    "reference_velocity_m_s":np.zeros(3),"reference_acceleration_m_s2":np.zeros(3),
                    "acceleration_m_s2":acceleration.copy(),"thrust_n":control.wrench.thrust_n,
                    "tilt_deg":tilt,"clearance_m":self.true_world.segment_clearance_m(
                        self.state.position_m,self.state.position_m),
                    "tracking_error_m":float(np.linalg.norm(self.state.position_m-hold_position)),
                    "phase":NavigationState.HOLD,"target_id":target_id})
                next_state=advance_state(self.state,control.wrench,self.vehicle.vehicle,dt)
                clearance=self.true_world.segment_clearance_m(self.state.position_m,next_state.position_m)
                self.minimum_clearance_m=min(self.minimum_clearance_m,clearance)
                if clearance<=COLLISION_TOLERANCE_M:
                    self.failure="hold_collision"
                    return None
                if (self.peak_speed_m_s>self.limits.max_speed_m_s+1e-7 or
                    self.peak_acceleration_m_s2>self.limits.max_acceleration_m_s2+1e-7 or
                    self.peak_thrust_n>self.limits.max_thrust_n+1e-7 or
                    self.minimum_thrust_n<self.limits.min_thrust_n-1e-7 or
                    self.peak_tilt_deg>self.limits.max_tilt_deg+1e-7):
                    self.failure="hold_dynamic_limit"
                    return None
                self.state=next_state
                self.time_s+=dt
        self.failure="persistent_visual_loss"
        return None

    def _acquire(self,target_id: str):
        self._transition(NavigationState.ACQUIRE,target_id)
        fix=self._observe(target_id,force=True)
        if fix.valid:
            return fix
        return self._hold_reacquire(target_id)

    def _align_yaw_from_fix(self,fix):
        """Aim camera/body along the image-estimated passage normal."""
        sign=1 if self.guidance.targets[fix.target_id].expected_side=="positive_normal" else -1
        direction=-sign*fix.estimated_window.normal_world
        self.yaw=float(np.arctan2(direction[1],direction[0]))

    def _fly(self,trajectory,target_id: str | None,phase: str):
        self._transition(phase,target_id)
        dt=float(self.raw["guidance"].get("step_s",.01))
        horizon=trajectory.duration_s+self.raw["guidance"]["settle_s"]
        steps=int(np.ceil(horizon/dt))
        start_time=self.time_s
        last_valid_time=self.time_s
        lost_before_crossing=False
        true_wall=self.true_world.wall_for_id(target_id) if phase==NavigationState.TRAVERSE else None
        side=1 if (target_id and self.guidance.targets[target_id].expected_side=="positive_normal") else -1
        started_side=(true_wall is not None and
            true_wall.plane_coordinate_m(self.state.position_m)*side>
            true_wall.window.frame_thickness_m/2+self.true_world.envelope_m)
        plane_crossed=False
        plane_crossing_time=None
        traverse_clearance=float("inf")
        for step in range(steps+1):
            local_t=min(step*dt,trajectory.duration_s)
            reference=trajectory.evaluate(local_t)
            control=self.controller.command(self.state,reference.position_m,self.yaw,
                reference.velocity_m_s,reference.acceleration_m_s2)
            acceleration=world_acceleration(self.state,control.wrench,self.vehicle.vehicle)
            rotation=quaternion_to_matrix(self.state.quaternion_bw)
            tilt=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))))
            tracking=float(np.linalg.norm(self.state.position_m-reference.position_m))
            self.peak_tracking_error_m=max(self.peak_tracking_error_m,tracking)
            self.peak_speed_m_s=max(self.peak_speed_m_s,float(np.linalg.norm(self.state.velocity_m_s)))
            self.peak_acceleration_m_s2=max(self.peak_acceleration_m_s2,float(np.linalg.norm(acceleration)))
            self.peak_thrust_n=max(self.peak_thrust_n,control.wrench.thrust_n)
            self.minimum_thrust_n=min(self.minimum_thrust_n,control.wrench.thrust_n)
            self.peak_tilt_deg=max(self.peak_tilt_deg,tilt)
            point_clearance=self.true_world.segment_clearance_m(self.state.position_m,self.state.position_m)
            self.minimum_clearance_m=min(self.minimum_clearance_m,point_clearance)
            if true_wall is not None:
                traverse_clearance=min(traverse_clearance,point_clearance)
            self.actual.append({"time_s":float(self.time_s),"position_m":self.state.position_m.copy(),
                "velocity_m_s":self.state.velocity_m_s.copy(),"quaternion_bw":self.state.quaternion_bw.copy(),
                "reference_position_m":reference.position_m.copy(),
                "reference_velocity_m_s":reference.velocity_m_s.copy(),
                "reference_acceleration_m_s2":reference.acceleration_m_s2.copy(),
                "acceleration_m_s2":acceleration.copy(),"thrust_n":control.wrench.thrust_n,
                "tilt_deg":tilt,"clearance_m":point_clearance,"tracking_error_m":tracking,
                "phase":phase,"target_id":target_id})
            if target_id is not None:
                fix=self._observe(target_id)
                if fix is not None:
                    if fix.valid:
                        last_valid_time=self.time_s
                    elif phase==NavigationState.APPROACH and self.time_s-last_valid_time>self.policy.max_loss_s:
                        lost_before_crossing=True
            if point_clearance<=COLLISION_TOLERANCE_M:
                self.failure="actual_collision"
            elif tracking>self.policy.max_tracking_error_m:
                self.failure="tracking_error_limit"
            elif (self.peak_speed_m_s>self.limits.max_speed_m_s+1e-7 or
                  self.peak_acceleration_m_s2>self.limits.max_acceleration_m_s2+1e-7 or
                  self.peak_thrust_n>self.limits.max_thrust_n+1e-7 or
                  self.minimum_thrust_n<self.limits.min_thrust_n-1e-7 or
                  self.peak_tilt_deg>self.limits.max_tilt_deg+1e-7):
                self.failure="actual_dynamic_limit"
            if self.failure or step==steps:
                break
            next_state=advance_state(self.state,control.wrench,self.vehicle.vehicle,dt)
            swept=self.true_world.segment_clearance_m(self.state.position_m,next_state.position_m)
            self.minimum_clearance_m=min(self.minimum_clearance_m,swept)
            if true_wall is not None:
                traverse_clearance=min(traverse_clearance,swept)
            if swept<=COLLISION_TOLERANCE_M:
                self.failure="actual_swept_collision"
                break
            if true_wall is not None:
                before=true_wall.plane_coordinate_m(self.state.position_m)*side
                after=true_wall.plane_coordinate_m(next_state.position_m)*side
                if before>0 and after<=0:
                    plane_crossed=True
                    plane_crossing_time=start_time+(step+1)*dt
            self.state=next_state
            self.time_s=start_time+(step+1)*dt
        if self.failure:
            self._transition(NavigationState.ABORT,target_id)
            return False
        final_error=float(np.linalg.norm(self.state.position_m-
                                         np.asarray(trajectory.waypoints_m[-1])))
        if final_error>self.policy.final_segment_error_m:
            self.failure="segment_did_not_settle"
        elif lost_before_crossing and phase==NavigationState.APPROACH:
            # The certified pre-window segment may finish, but crossing requires reacquisition.
            self._transition(NavigationState.HOLD,target_id)
        if true_wall is not None and not self.failure:
            exit_distance=true_wall.window.frame_thickness_m/2+self.true_world.envelope_m
            far_enough=true_wall.plane_coordinate_m(self.state.position_m)*side < -exit_distance
            if not started_side or not plane_crossed or not far_enough:
                self.failure="crossing_not_confirmed"
            else:
                self.crossings.append({"window_id":target_id,"plane_crossing_time_s":float(plane_crossing_time),
                    "entry_side":self.guidance.targets[target_id].expected_side,
                    "exit_envelope_cleared":True,"minimum_clearance_m":float(traverse_clearance)})
        if self.failure:
            self._transition(NavigationState.ABORT,target_id)
        return self.failure is None

    def run(self):
        start=self.state.position_m.copy()
        takeoff=np.asarray(self.raw["mission"]["takeoff_world_m"],dtype=float).copy()
        if self.reverse:
            takeoff[0]=start[0]
        takeoff[1]=start[1]
        simple_world=World(self.true_world.workspace,(),self.true_world.vehicle_radius_m,
                           self.raw["workspace"]["planning_clearance_m"])
        trajectory,report=generate_trajectory([start,takeoff],simple_world,
                                                self.vehicle.vehicle,self.limits)
        self.reference_reports.append({"phase":"takeoff","report":report})
        if trajectory is None or not self._fly(trajectory,None,NavigationState.TAKEOFF):
            self.failure=self.failure or "takeoff_infeasible"
            self._transition(NavigationState.ABORT,None)
            return self.metrics()
        for target_id in self.mission_ids:
            fix=self._acquire(target_id)
            if fix is None:
                self._transition(NavigationState.ABORT,target_id)
                return self.metrics()
            self._align_yaw_from_fix(fix)
            entry,_=self.guidance.entry_exit(fix)
            trajectory,report=self.guidance.plan_segment(self.state,fix,entry)
            self.reference_reports.append({"phase":"approach","target_id":target_id,"report":report})
            if trajectory is None:
                self.failure="approach_"+str(report["reason"])
                self._transition(NavigationState.ABORT,target_id)
                return self.metrics()
            if not self._fly(trajectory,target_id,NavigationState.APPROACH):
                return self.metrics()
            self._transition(NavigationState.ALIGN,target_id)
            refined=self._acquire(target_id)
            if refined is None:
                self._transition(NavigationState.ABORT,target_id)
                return self.metrics()
            self._align_yaw_from_fix(refined)
            _,exit_point=self.guidance.entry_exit(refined)
            trajectory,report=self.guidance.plan_segment(self.state,refined,exit_point)
            self.reference_reports.append({"phase":"traverse","target_id":target_id,"report":report})
            if trajectory is None:
                self.failure="traverse_"+str(report["reason"])
                self._transition(NavigationState.ABORT,target_id)
                return self.metrics()
            if not self._fly(trajectory,target_id,NavigationState.TRAVERSE):
                return self.metrics()
            self._transition(NavigationState.CONFIRM_EXIT,target_id)
        self._transition(NavigationState.COMPLETE,None)
        return self.metrics()

    def metrics(self):
        detected=[r for r in self.scoring if r["corner_rmse_px"] is not None]
        valid=[r for r in self.scoring if r["valid"]]
        pre_cross=[r for r,p in zip(self.scoring,self.perception)
                   if p["state"] in (NavigationState.ACQUIRE,NavigationState.APPROACH,
                                     NavigationState.ALIGN,NavigationState.HOLD)]
        def mean(key,rows):
            values=[r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        return {"condition":vars(self.condition),"mission_ids":self.mission_ids,
            "reverse":self.reverse,"completed":self.failure is None and
            self.state_name==NavigationState.COMPLETE,"failure_reason":self.failure,
            "crossing_order":[x["window_id"] for x in self.crossings],
            "crossing_count":len(self.crossings),"collision_count":int(self.failure in
                ("actual_collision","actual_swept_collision","hold_collision")),
            "abort_count":int(self.failure is not None),"recovery_count":self.recoveries,
            "image_count":len(self.scoring),"detection_count":len(detected),
            "valid_pose_count":len(valid),"detection_success_rate":len(detected)/len(self.scoring)
                if self.scoring else 0.,"valid_pose_rate":len(valid)/len(self.scoring)
                if self.scoring else 0.,"pre_crossing_detection_rate":
                sum(r["valid"] for r in pre_cross)/len(pre_cross) if pre_cross else 0.,
            "mean_corner_error_px":mean("corner_rmse_px",detected),
            "mean_translation_error_m":mean("translation_error_m",valid),
            "mean_orientation_error_deg":mean("orientation_error_deg",valid),
            "mean_reprojection_rmse_px":mean("reprojection_rmse_px",self.perception),
            "detection_failures":dict(Counter(r["failure_reason"] for r in self.scoring if not r["valid"])),
            "min_actual_clearance_m":float(self.minimum_clearance_m),
            "max_tracking_error_m":float(self.peak_tracking_error_m),
            "max_actual_speed_m_s":float(self.peak_speed_m_s),
            "max_actual_acceleration_m_s2":float(self.peak_acceleration_m_s2),
            "min_applied_thrust_n":float(self.minimum_thrust_n),
            "max_applied_thrust_n":float(self.peak_thrust_n),
            "max_actual_tilt_deg":float(self.peak_tilt_deg),
            "simulated_duration_s":float(self.time_s)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/window_m9.toml")
    parser.add_argument("--output-dir",default="outputs/window_m9")
    parser.add_argument("--single-replay",action="store_true",
                        help="one nominal flight without saved study artifacts")
    args=parser.parse_args()
    raw,vehicle,limits,camera,world,targets,colors=load_m9_config(args.config)
    if args.single_replay:
        condition=FlightCondition("nominal",raw["mission"]["nominal_seeds"][0],
                                  raw["render"]["noise_std_rgb"],raw["render"]["illumination"],
                                  raw["render"]["blur"],0.)
        flight=WindowMission(raw,vehicle,limits,camera,world,targets,colors,condition)
        print(json.dumps(flight.run(),indent=2))
    else:
        print(json.dumps(run_experiments(raw,vehicle,limits,camera,world,targets,colors,
                                          args.output_dir),indent=2))


def _save_flight(flight: WindowMission,folder: Path):
    folder.mkdir(parents=True,exist_ok=True)
    operational=folder/"operational"
    evaluation=folder/"evaluation"
    operational.mkdir(exist_ok=True);evaluation.mkdir(exist_ok=True)
    def write(path,data):
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write(operational/"perception.json",flight.perception)
    write(operational/"navigation_transitions.json",flight.transitions)
    write(operational/"reference_feasibility.json",flight.reference_reports)
    write(evaluation/"image_pose_scoring.json",flight.scoring)
    write(evaluation/"crossing_events.json",flight.crossings)
    write(evaluation/"metrics.json",flight.metrics())
    if flight.actual:
        keys=("time_s","position_m","velocity_m_s","quaternion_bw",
              "reference_position_m","reference_velocity_m_s","reference_acceleration_m_s2",
              "acceleration_m_s2","thrust_n","tilt_deg","clearance_m","tracking_error_m")
        np.savez_compressed(evaluation/"flight_trace.npz",
            **{key:np.asarray([row[key] for row in flight.actual]) for key in keys},
            phase=np.asarray([row["phase"] for row in flight.actual]),
            target_id=np.asarray([row["target_id"] or "" for row in flight.actual]))


def run_experiments(raw,vehicle,limits,camera,world,targets,colors,output_dir):
    from aerial_autonomy.evaluation.window_m9_study import run_detector_study
    from aerial_autonomy.visualization.window_m9_plot import (camera_overlay,
        save_camera_gif,save_3d_gif,save_trajectory_3d,save_perception_plot,save_safety_plot)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"config_snapshot.json").write_text(json.dumps(raw,indent=2)+"\n",encoding="utf-8")
    study,_=run_detector_study(raw,vehicle,limits,camera,world,targets,colors,output_dir)
    conditions=[]
    for seed in raw["mission"]["nominal_seeds"]:
        conditions.append(FlightCondition("nominal",seed,raw["render"]["noise_std_rgb"],
            raw["render"]["illumination"],raw["render"]["blur"],0.))
    for seed,offset in zip(raw["mission"]["stressed_seeds"],
                           raw["mission"]["stressed_initial_lateral_offsets_m"]):
        conditions.append(FlightCondition("stressed",seed,
            raw["mission"]["stressed_noise_std_rgb"],.85,seed==43,
            raw["mission"]["stressed_dropout_probability"],offset))
    results=[];hero=None
    for condition in conditions:
        flight=WindowMission(raw,vehicle,limits,camera,world,targets,colors,condition,
                             capture_frames=(condition.name=="nominal" and condition.seed==11))
        result=flight.run()
        results.append(result)
        _save_flight(flight,output_dir/"trials"/f"{condition.name}_seed_{condition.seed}")
        if flight.capture_frames:
            hero=flight
        print(f"{condition.name} seed {condition.seed}: completed={result['completed']} "
              f"crossed={result['crossing_order']} failure={result['failure_reason']}",flush=True)
    single=WindowMission(raw,vehicle,limits,camera,world,targets,colors,
        FlightCondition("single",11,3.,1.,False,0.),mission_ids=["alpha"])
    single_result=single.run();_save_flight(single,output_dir/"trials"/"single_alpha")
    reverse=WindowMission(raw,vehicle,limits,camera,world,targets,colors,
        FlightCondition("reverse",11,3.,1.,False,0.),mission_ids=["alpha"],reverse=True)
    reverse_result=reverse.run();_save_flight(reverse,output_dir/"trials"/"reverse_alpha")
    forced_loss=WindowMission(raw,vehicle,limits,camera,world,targets,colors,
        FlightCondition("forced_loss",101,3.,1.,False,1.),mission_ids=["alpha"])
    loss_result=forced_loss.run();_save_flight(forced_loss,output_dir/"trials"/"forced_loss")
    figures=output_dir/"figures";figures.mkdir(exist_ok=True)
    visuals=[str(path) for path in sorted(figures.glob("camera_*.png"))]
    timeline={}
    if hero is not None and hero.metrics()["completed"]:
        camera_path=figures/"camera_mission.gif"
        save_camera_gif(hero.frames,camera_path);visuals.append(str(camera_path))
        frame_folder=output_dir/"selected_frames";frame_folder.mkdir(exist_ok=True)
        for index in np.linspace(0,len(hero.frames)-1,14,dtype=int):
            frame=hero.frames[int(index)]
            title=f"t={frame['time_s']:.1f} s {frame['state']} {frame['target_id']}"
            status=("image detection valid" if frame["detection"].valid else
                    "rejected: "+str(frame["detection"].reason))
            path=frame_folder/f"camera_{int(index):03d}.png"
            camera_overlay(frame["image"],frame["detection"],
                frame["estimated_reproject_uv"],title,status).save(path)
            visuals.append(str(path))
        static_path=figures/"flight_3d.png"
        save_trajectory_3d(hero.actual,world,colors,hero.crossings,static_path)
        visuals.append(str(static_path))
        animation_path=figures/"flight_3d.gif"
        frame_times=save_3d_gif(hero.actual,world,colors,hero.crossings,animation_path)
        visuals.append(str(animation_path))
        perception_path=figures/"perception_errors.png"
        save_perception_plot(hero.scoring,perception_path);visuals.append(str(perception_path))
        safety_path=figures/"tracking_safety.png"
        save_safety_plot(hero.actual,hero.crossings,safety_path);visuals.append(str(safety_path))
        camera_times=np.asarray([frame["time_s"] for frame in hero.frames])
        for event in hero.crossings:
            time=event["plane_crossing_time_s"]
            timeline[event["window_id"]]={"crossing_time_s":time,
                "camera_frame_index":int(np.argmin(abs(camera_times-time))),
                "3d_frame_index":int(np.argmin(abs(np.asarray(frame_times)-time)))}
        (output_dir/"animation_timeline.json").write_text(json.dumps(timeline,indent=2)+"\n",
                                                          encoding="utf-8")
    detector_valid=[r for r in study if r["detected"]]
    scored=[r for r in detector_valid if r["corner_rmse_px"] is not None]
    pose_scored=[r for r in study if r["pose_valid"]]
    oracle_scored=[r for r in study if r["oracle_corner_pose_valid"]]
    flight_successes=[r for r in results if r["completed"]]
    nominal=[r for r in results if r["condition"]["name"]=="nominal"]
    stressed=[r for r in results if r["condition"]["name"]=="stressed"]
    aggregate={"baseline_m8_head":"b57153e16f63daa20a8b35ae68110a0529329877",
        "detector_study_cases":len(study),"detector_study_valid":len(detector_valid),
        "detector_study_false_positives":sum(r["false_positive"] for r in study),
        "detector_study_mean_corner_error_px":float(np.mean([r["corner_rmse_px"] for r in scored])),
        "detector_study_mean_translation_error_m":float(np.mean(
            [r["translation_error_m"] for r in pose_scored])),
        "detector_study_mean_orientation_error_deg":float(np.mean(
            [r["orientation_error_deg"] for r in pose_scored])),
        "m8_oracle_diagnostic_valid":len(oracle_scored),
        "m8_oracle_diagnostic_mean_translation_error_m":float(np.mean(
            [r["oracle_translation_error_m"] for r in oracle_scored])),
        "m8_oracle_diagnostic_mean_orientation_error_deg":float(np.mean(
            [r["oracle_orientation_error_deg"] for r in oracle_scored])),
        "m8_oracle_diagnostic_mean_reprojection_rmse_px":float(np.mean(
            [r["oracle_reprojection_rmse_px"] for r in oracle_scored])),
        "nominal_completed":sum(r["completed"] for r in nominal),
        "nominal_trials":len(nominal),"stressed_completed":sum(r["completed"] for r in stressed),
        "stressed_trials":len(stressed),"continuous_flight_completion_rate":
            len(flight_successes)/len(results),"single_window":single_result,
        "reverse_window":reverse_result,"forced_loss":loss_result,
        "minimum_successful_clearance_m":min((r["min_actual_clearance_m"] for r in flight_successes),default=None),
        "all_trial_failures":[{"condition":r["condition"],"reason":r["failure_reason"],
             "crossings":r["crossing_order"]} for r in results if not r["completed"]],
        "visual_paths":visuals,"animation_timeline":timeline}
    gates=raw["acceptance"]
    aggregate["predeclared_gates_pass"]=bool(
        aggregate["nominal_completed"]>=gates["min_completed_nominal"] and
        aggregate["stressed_completed"]>=gates["min_completed_stressed"] and
        aggregate["minimum_successful_clearance_m"]>gates["minimum_actual_clearance_m"] and
        aggregate["detector_study_mean_corner_error_px"]<gates["max_mean_corner_error_px"] and
        aggregate["detector_study_mean_translation_error_m"]<gates["max_mean_translation_error_m"] and
        aggregate["detector_study_mean_orientation_error_deg"]<gates["max_mean_orientation_error_deg"] and
        single_result["completed"] and reverse_result["completed"] and
        not loss_result["completed"] and loss_result["crossing_count"]==0)
    (output_dir/"trial_summary.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    (output_dir/"aggregate_metrics.json").write_text(json.dumps(aggregate,indent=2)+"\n",encoding="utf-8")
    (output_dir/"visual_manifest.json").write_text(json.dumps(visuals,indent=2)+"\n",encoding="utf-8")
    return {"predeclared_gates_pass":aggregate["predeclared_gates_pass"],
            "nominal_completed":aggregate["nominal_completed"],
            "stressed_completed":aggregate["stressed_completed"],
            "minimum_successful_clearance_m":aggregate["minimum_successful_clearance_m"],
            "hero_crossings":timeline,"visual_count":len(visuals)}


if __name__=="__main__":
    main()
