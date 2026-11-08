"""One-clock, one-plant M11 mission composed from the accepted M7--M10 APIs.

Truth is owned by the renderer, plant, and independent safety monitor. The
guidance object receives only RGB-derived window poses and onboard state;
the timing choice receives only RGB-derived arm motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.window_m9 import load_m9_config
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.mission_m11_detect import TargetColorDetector
from aerial_autonomy.perception.window_m9_guidance import ImageWindowGuidance, TargetModel
from aerial_autonomy.perception.window_m9_render import RenderSettings, WindowImageRenderer
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld, WindowWall
from aerial_autonomy.perception.window_m10_arm import (ArmShape, RotatingArm,
    arm_clearance_m, swept_arm_clearance_m)
from aerial_autonomy.perception.window_m10_timing import (AngularTracker, MotionEstimate,
    choose_crossing, reference_dynamic_clearance, wrap_angle)
from aerial_autonomy.perception.window_m10_vision import PixelArmDetector, RotatingArmRenderer
from aerial_autonomy.planning.geometry import Workspace
from aerial_autonomy.simulation.dynamics import VehicleState, advance_state, world_acceleration
from aerial_autonomy.simulation.rotation import euler_to_quaternion, quaternion_to_matrix
from aerial_autonomy.trajectory.quintic import Reference, generate_trajectory


EXPECTED_CROSSINGS = (("alpha", "outbound"), ("beta", "outbound"),
    ("gamma", "outbound"), ("delta", "outbound"), ("gamma", "return"),
    ("beta", "return"), ("alpha", "return"))


def load_m11_config(path: str | Path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    _, vehicle, limits, camera, m9_world, _, colors = load_m9_config(raw["inputs"]["m9_config"])
    space = raw["workspace"]
    if (abs(space["vehicle_radius_m"] - .18) > 1e-12 or
            abs(space["actual_clearance_m"] - .07) > 1e-12 or
            space["planning_clearance_m"] < space["actual_clearance_m"]):
        raise ValueError("M11 must retain the 0.25 m actual flight envelope")
    d = raw["delta"]
    model = WindowModel(d["window_id"], d["width_m"], d["height_m"])
    window = StaticWindow(model, np.asarray(d["center_world_m"]),
        np.asarray(d["right_world"]), np.asarray(d["up_world"]), d["frame_thickness_m"])
    wall = WindowWall(window, d["outer_width_m"], d["outer_height_m"])
    world = WindowCourseWorld(Workspace(tuple(space["minimum"]), tuple(space["maximum"])),
        (*m9_world.walls, wall), space["vehicle_radius_m"], space["actual_clearance_m"])
    a = raw["arm"]
    shape = ArmShape(a["pivot_right_m"], a["pivot_up_m"], a["length_m"], a["arm_radius_m"])
    arm = RotatingArm(window, shape, a["initial_angle_rad"], a["angular_velocity_rad_s"])
    colors = {**colors, model.window_id: tuple(d["color_rgb"])}
    study = geometry_study(raw, world, arm, vehicle, limits)
    if not study["feasible"]:
        raise ValueError("M11 declared geometry or bypass is infeasible: " + str(study))
    return raw, vehicle, limits, camera, world, arm, colors, study


def geometry_study(raw, world: WindowCourseWorld, arm: RotatingArm,
                   vehicle=None, limits=None) -> dict:
    """Seed-independent frame, phase, and complete-crossing geometry gates."""
    w, s = arm.window, arm.shape
    half_width, half_height = w.model.width_m / 2, w.model.height_m / 2
    tip_to_frame = min(half_width - abs(s.pivot_right_m) - s.length_m - s.arm_radius_m,
                       half_height - abs(s.pivot_up_m) - s.length_m - s.arm_radius_m)
    lateral = raw["guidance"]["delta_corridor_right_m"]
    vehicle_to_frame = min(half_width - abs(lateral) - world.envelope_m,
                           half_height - world.envelope_m)
    route = raw["route"]["bypass_world_m"]
    route_clearance = min(world.segment_clearance_m(a, b) for a, b in zip(route[:-1], route[1:]))
    center = w.center_world_m + w.right_world * lateral
    phases = np.linspace(0, 2 * math.pi, 721)
    clearances = [arm_clearance_m(center, w, s, float(angle), world.envelope_m)
                  for angle in phases]
    # The operational predictor carries further uncertainty and full-crossing time.
    safe_fraction = float(np.mean(np.asarray(clearances) > .15))
    tracking_reserve = (raw["workspace"]["planning_clearance_m"] -
                        raw["workspace"]["actual_clearance_m"])
    study = {"feasible": tip_to_frame >= .02-1e-12 and
            vehicle_to_frame > tracking_reserve and
            route_clearance > .05 and safe_fraction > .05 and min(clearances) < 0,
            "arm_length_to_opening_width": s.length_m / w.model.width_m,
            "arm_length_to_opening_height": s.length_m / w.model.height_m,
            "arm_length_to_half_width": s.length_m / half_width,
            "arm_radius_m": s.arm_radius_m, "arm_tip_to_frame_m": tip_to_frame,
            "vehicle_corridor_to_frame_m": vehicle_to_frame,
            "tracking_reserve_m": tracking_reserve,
            "frame_clearance_after_tracking_reserve_m": vehicle_to_frame-tracking_reserve,
            "minimum_bypass_clearance_m": route_clearance,
            "corridor_safe_phase_fraction_at_plane": safe_fraction,
            "corridor_min_arm_clearance_at_plane_m": float(min(clearances)),
            "corridor_max_arm_clearance_at_plane_m": float(max(clearances)),
            "envelope_m": world.envelope_m}
    if vehicle is not None and limits is not None:
        start = center + w.normal_world * raw["guidance"]["delta_approach_distance_m"]
        end = center - w.normal_world * raw["guidance"]["delta_exit_distance_m"]
        trajectory, report = generate_trajectory([start, end], world, vehicle.vehicle,
                                                  limits)
        study["full_crossing_reference_feasible"] = bool(report["success"])
        study["full_crossing_reference_report"] = report
        if trajectory is not None:
            # M10's minimum tracker uncertainty floors are carried across the
            # entire envelope passage. Actual scheduling uses measured values.
            phases = np.linspace(0, 2*math.pi, 72, endpoint=False)
            predicted = [reference_dynamic_clearance(trajectory, 0.,
                MotionEstimate(True, 0., float(phase), arm.angular_velocity_rad_s,
                               .045, .025, 0., None, 7), w, s, world.envelope_m)
                for phase in phases]
            study["full_crossing_duration_s"] = trajectory.duration_s
            study["full_crossing_safe_phase_fraction"] = float(np.mean(
                np.asarray(predicted) > .035))
            study["full_crossing_min_predicted_arm_clearance_m"] = float(min(predicted))
            study["full_crossing_max_predicted_arm_clearance_m"] = float(max(predicted))
            study["feasible"] = (study["feasible"] and
                study["full_crossing_safe_phase_fraction"] > .05)
        else:
            study["feasible"] = False
    return study


def physical_crossings(rows: list[dict], world: WindowCourseWorld) -> list[dict]:
    """Derive opening events solely from saved physical positions and geometry."""
    events = []
    positions = np.asarray([r["position_m"] for r in rows])
    for wall in world.walls:
        coordinates = wall.world_to_local(positions)
        plane = coordinates[:, 2]
        threshold = wall.window.frame_thickness_m / 2 + world.envelope_m
        for i in range(1, len(rows)):
            a, b = plane[i-1], plane[i]
            if a * b > 0 or a == b:
                continue
            fraction = float(np.clip(a / (a-b), 0, 1))
            local = coordinates[i-1] + fraction * (coordinates[i] - coordinates[i-1])
            t = rows[i-1]["time_s"] + fraction * (rows[i]["time_s"]-rows[i-1]["time_s"])
            inside = (abs(local[0]) + world.envelope_m < wall.window.model.width_m / 2 and
                      abs(local[1]) + world.envelope_m < wall.window.model.height_m / 2)
            if not inside:
                if wall.window.model.window_id == "delta":
                    events.append({"window_id": "delta", "classification": "external_plane_bypass",
                        "plane_crossing_time_s": float(t), "local_right_m": float(local[0])})
                continue
            sign = 1 if a > b else -1
            before = next((j for j in range(i-1, -1, -1) if plane[j] * sign >= threshold), None)
            after = next((j for j in range(i, len(rows)) if plane[j] * sign <= -threshold), None)
            events.append({"window_id": wall.window.model.window_id,
                "direction": "outbound" if sign == 1 else "return",
                "classification": "opening", "entry_time_s": (float(rows[before]["time_s"])
                    if before is not None else None), "plane_crossing_time_s": float(t),
                "exit_time_s": float(rows[after]["time_s"]) if after is not None else None,
                "local_right_m": float(local[0]), "local_up_m": float(local[1]),
                "mission_phase": rows[i]["phase"],
                "minimum_static_clearance_m": (float(min(r["static_clearance_m"]
                    for r in rows[before:after+1])) if before is not None and
                    after is not None else None),
                "minimum_dynamic_clearance_m": (float(min(r["dynamic_clearance_m"]
                    for r in rows[before:after+1])) if before is not None and
                    after is not None else None)})
    return sorted(events, key=lambda item: item["plane_crossing_time_s"])


@dataclass(frozen=True)
class MissionCondition:
    seed: int
    initial_phase_rad: float
    noise_std_rgb: float = 3.0
    dropout_probability: float = 0.0
    forced_visual_loss: bool = False
    stationary_arm: bool = False


class CompleteMission:
    """Owns one monotonically advancing state, controller, RNG and clock."""

    def __init__(self, raw, vehicle, limits, camera, world, arm, colors,
                 condition: MissionCondition, capture_frames=False,
                 known_arm_shape: ArmShape | None = None):
        self.raw, self.vehicle, self.limits, self.camera = raw, vehicle, limits, camera
        self.world, self.colors, self.condition = world, colors, condition
        self.arm = RotatingArm(arm.window, arm.shape, condition.initial_phase_rad,
            0.0 if condition.stationary_arm else arm.angular_velocity_rad_s)
        declared = raw["arm"]
        self.known_arm_shape = known_arm_shape or ArmShape(declared["pivot_right_m"],
            declared["pivot_up_m"], declared["length_m"], declared["arm_radius_m"])
        self.renderer = RotatingArmRenderer(WindowImageRenderer(camera, world, colors), self.arm)
        self.arm_detector, self.tracker = PixelArmDetector(), AngularTracker(
            raw["guidance"]["max_observation_age_s"])
        self.controller = CascadedHoverController(vehicle.vehicle, vehicle.controller)
        self.rng = np.random.default_rng(condition.seed)
        self.state = VehicleState.at_rest()
        self.state.position_m = np.zeros(3)
        self.home_m = self.state.position_m.copy()
        self.time_s, self.yaw_rad = 0.0, 0.0
        self.phase, self.target_id = "INITIALIZE", None
        self.transitions = [{"time_s": 0.0, "state": "INITIALIZE", "target_id": None}]
        self.records, self.observations, self.truth_scoring = [], [], []
        self.reference_reports, self.choices, self.frames = [], [], []
        self.failure, self.landed, self.waiting_s = None, False, 0.0
        self.next_image_s = 0.0
        self.last_fix = None
        self.min_static_clearance_m = math.inf
        self.min_dynamic_clearance_m = math.inf
        self.peak_tracking_error_m = 0.0
        self.peak_speed_m_s = 0.0
        self.peak_acceleration_m_s2 = 0.0
        self.peak_thrust_n = 0.0
        self.min_thrust_n = math.inf
        self.peak_tilt_deg = 0.0
        self.peak_torque_nm = np.zeros(3)
        self.capture_frames = capture_frames
        self.last_frame_s = -math.inf
        self._record_initial()

    def _record_initial(self):
        self.records.append({"time_s": 0.0, "position_m": self.state.position_m.copy(),
            "velocity_m_s": self.state.velocity_m_s.copy(),
            "quaternion_bw": self.state.quaternion_bw.copy(),
            "body_rate_rad_s": self.state.body_rate_rad_s.copy(),
            "reference_position_m": self.state.position_m.copy(),
            "reference_velocity_m_s": np.zeros(3),
            "reference_acceleration_m_s2": np.zeros(3),
            "desired_quaternion_bw": self.state.quaternion_bw.copy(),
            "thrust_n": 0.0, "torque_nm": np.zeros(3), "phase": self.phase,
            "target_id": None, "static_clearance_m": self.world.segment_clearance_m(
                self.state.position_m, self.state.position_m),
            "dynamic_clearance_m": arm_clearance_m(self.state.position_m, self.arm.window,
                self.arm.shape, self.arm.angle_at(0), self.world.envelope_m)})

    def transition(self, name, target_id=None):
        if self.phase != name or self.target_id != target_id:
            self.phase, self.target_id = name, target_id
            self.transitions.append({"time_s": float(self.time_s),
                "state": name, "target_id": target_id})

    def abort(self, reason):
        if self.failure is None:
            self.failure = reason
        self.transition("ABORT", self.target_id)
        return False

    def _step(self, reference: Reference) -> bool:
        dt = self.raw["guidance"]["step_s"]
        control = self.controller.command(self.state, reference.position_m, self.yaw_rad,
            reference.velocity_m_s, reference.acceleration_m_s2)
        acceleration = world_acceleration(self.state, control.wrench, self.vehicle.vehicle)
        future = advance_state(self.state, control.wrench, self.vehicle.vehicle, dt)
        static = self.world.segment_clearance_m(self.state.position_m, future.position_m)
        dynamic = swept_arm_clearance_m(self.state.position_m, future.position_m,
            self.time_s, self.time_s+dt, self.arm, self.world.envelope_m)
        tracking = float(np.linalg.norm(self.state.position_m-reference.position_m))
        speed = float(np.linalg.norm(future.velocity_m_s))
        accel = float(np.linalg.norm(acceleration))
        tilt = float(np.degrees(np.arccos(np.clip(
            quaternion_to_matrix(future.quaternion_bw)[2, 2], -1, 1))))
        self.min_static_clearance_m = min(self.min_static_clearance_m, static)
        self.min_dynamic_clearance_m = min(self.min_dynamic_clearance_m, dynamic)
        self.peak_tracking_error_m = max(self.peak_tracking_error_m, tracking)
        self.peak_speed_m_s = max(self.peak_speed_m_s, speed)
        self.peak_acceleration_m_s2 = max(self.peak_acceleration_m_s2, accel)
        self.peak_thrust_n = max(self.peak_thrust_n, control.wrench.thrust_n)
        self.min_thrust_n = min(self.min_thrust_n, control.wrench.thrust_n)
        self.peak_tilt_deg = max(self.peak_tilt_deg, tilt)
        self.peak_torque_nm = np.maximum(self.peak_torque_nm, abs(control.wrench.torque_nm))
        if static <= 0 or dynamic <= 0:
            return self.abort("static_collision" if static <= 0 else "dynamic_arm_collision")
        if future.position_m[2] < -1e-4:
            return self.abort("ground_penetration")
        limits = self.raw["acceptance"]
        if (speed > limits["maximum_speed_m_s"]+1e-6 or
                accel > limits["maximum_acceleration_m_s2"]+1e-6 or
                control.wrench.thrust_n > limits["maximum_thrust_n"]+1e-6 or
                control.wrench.thrust_n < limits["minimum_thrust_n"]-1e-6 or
                tilt > limits["maximum_tilt_deg"]+1e-6):
            return self.abort("actual_dynamic_limit")
        self.state = future
        self.time_s += dt
        self.records.append({"time_s": float(self.time_s),
            "position_m": future.position_m.copy(), "velocity_m_s": future.velocity_m_s.copy(),
            "quaternion_bw": future.quaternion_bw.copy(),
            "body_rate_rad_s": future.body_rate_rad_s.copy(),
            "reference_position_m": reference.position_m.copy(),
            "reference_velocity_m_s": reference.velocity_m_s.copy(),
            "reference_acceleration_m_s2": reference.acceleration_m_s2.copy(),
            "desired_quaternion_bw": euler_to_quaternion(*control.desired_euler_rad),
            "thrust_n": float(control.wrench.thrust_n),
            "torque_nm": control.wrench.torque_nm.copy(), "phase": self.phase,
            "target_id": self.target_id, "static_clearance_m": static,
            "dynamic_clearance_m": dynamic, "tracking_error_m": tracking,
            "actual_acceleration_m_s2": acceleration.copy(), "tilt_deg": tilt})
        return True

    def _hold(self, duration_s, phase=None, target_id=None):
        if phase is not None:
            self.transition(phase, target_id)
        position = self.state.position_m.copy()
        reference = Reference(position, np.zeros(3), np.zeros(3))
        for _ in range(math.ceil(duration_s / self.raw["guidance"]["step_s"])):
            if not self._step(reference):
                return False
        return True

    def _yaw_to(self, target_yaw, target_id=None):
        difference = (target_yaw-self.yaw_rad+math.pi) % (2*math.pi)-math.pi
        if abs(difference) < .015:
            return True
        self.transition("TURN_FOR_CAMERA", target_id)
        initial = self.yaw_rad
        duration = abs(difference)/self.raw["guidance"]["max_yaw_rate_rad_s"]*1.9
        count = math.ceil(duration/self.raw["guidance"]["step_s"])
        position = self.state.position_m.copy()
        for k in range(count):
            u = (k+1)/count
            self.yaw_rad = initial+difference*(10*u**3-15*u**4+6*u**5)
            if not self._step(Reference(position, np.zeros(3), np.zeros(3))):
                return False
        return self._hold(.7, "SETTLE_TURN", target_id)

    def _guidance(self, target_id, reverse=False, delta=False):
        wall = self.world.wall_for_id(target_id)
        item = wall.window.model
        catalog = {target_id: TargetModel(item, wall.window.frame_thickness_m,
            wall.outer_width_m, wall.outer_height_m,
            "negative_normal" if reverse else "positive_normal")}
        g = self.raw["guidance"]
        return ImageWindowGuidance(self.camera, TargetColorDetector({
            target_id: self.colors[target_id]}), catalog,
            self.world.workspace, self.vehicle.vehicle, self.limits,
            self.world.vehicle_radius_m, self.raw["workspace"]["planning_clearance_m"],
            g["delta_approach_distance_m"] if delta else g["static_approach_distance_m"],
            g["delta_exit_distance_m"] if delta else g["static_exit_distance_m"])

    def _observe(self, guidance, target_id, force=False):
        if not force and self.time_s+1e-9 < self.next_image_s:
            return None
        render = self.raw["render"]
        image = self.renderer.render(self.state, self.time_s,
            RenderSettings(self.condition.noise_std_rgb, render["illumination"],
                render["blur"]), self.rng)
        dropped = self.condition.forced_visual_loss or bool(
            self.rng.random() < self.condition.dropout_probability)
        sensed = np.zeros_like(image) if dropped else image
        fix = guidance.perceive(sensed, self.state, target_id, self.time_s)
        if fix.valid:
            self.last_fix = fix
        arm_detection = None
        estimate = None
        if target_id == "delta":
            arm_detection = self.arm_detector.detect(sensed, fix, self.time_s)
            self.tracker.update(arm_detection)
            estimate = self.tracker.estimate(self.time_s)
        record = {"time_s": float(self.time_s), "target_id": target_id,
            "phase": self.phase, "dropout": dropped, "window_valid": fix.valid,
            "window_reason": fix.reason, "detected_corners_uv":
                fix.detection.corners_uv.tolist() if fix.detection.valid else None,
            "estimated_center_world_m": (fix.estimated_window.center_world_m.tolist()
                if fix.valid else None), "reprojection_rmse_px":
                fix.estimate.reprojection_rmse_px if fix.valid else None,
            "arm_valid": arm_detection.valid if arm_detection else None,
            "detected_angle_rad": arm_detection.angle_rad if arm_detection else None,
            "tracker_valid": estimate.valid if estimate else None,
            "estimated_angle_rad": estimate.angle_rad if estimate and estimate.valid else None,
            "estimated_rate_rad_s": estimate.angular_velocity_rad_s
                if estimate and estimate.valid else None}
        self.observations.append(record)
        truth = self.world.wall_for_id(target_id).window
        self.truth_scoring.append({"time_s": float(self.time_s), "target_id": target_id,
            "true_center_world_m": truth.center_world_m.tolist(),
            "translation_error_m": (float(np.linalg.norm(
                fix.estimated_window.center_world_m-truth.center_world_m))
                if fix.valid else None),
            "true_arm_angle_rad": self.arm.angle_at(self.time_s) if target_id == "delta" else None,
            "true_arm_rate_rad_s": self.arm.angular_velocity_rad_s if target_id == "delta" else None,
            "arm_angle_error_rad": (wrap_angle(estimate.angle_rad-self.arm.angle_at(self.time_s))
                if estimate and estimate.valid else None),
            "arm_rate_error_rad_s": (estimate.angular_velocity_rad_s-
                self.arm.angular_velocity_rad_s if estimate and estimate.valid else None)})
        if self.capture_frames and self.time_s-self.last_frame_s >= .8:
            self.frames.append({"time_s": float(self.time_s), "image": image.copy(),
                "target_id": target_id, "phase": self.phase,
                "corners_uv": fix.detection.corners_uv.copy() if fix.detection.valid else None,
                "arm_angle_rad": arm_detection.angle_rad if arm_detection and arm_detection.valid else None,
                "pose_valid": fix.valid})
            self.last_frame_s = self.time_s
        self.next_image_s = self.time_s + self.raw["guidance"]["image_period_s"]
        return fix, estimate

    def _acquire(self, guidance, target_id):
        self.transition("ACQUIRE_WINDOW", target_id)
        fix, _ = self._observe(guidance, target_id, force=True)
        if fix.valid:
            return fix
        base_yaw = self.yaw_rad
        for offset_deg in (-15, 15, -30, 30, -45, 45, 0):
            if not self._yaw_to(base_yaw+math.radians(offset_deg), target_id):
                return None
            self.transition("HOLD_REACQUIRE", target_id)
            fix, _ = self._observe(guidance, target_id, force=True)
            if fix.valid:
                return fix
        self.abort("persistent_visual_loss")
        return None

    def _fly(self, trajectory, phase, target_id=None, guidance=None):
        self.transition(phase, target_id)
        dt = self.raw["guidance"]["step_s"]
        horizon = trajectory.duration_s+self.raw["guidance"]["settle_s"]
        start = self.time_s
        for step in range(math.ceil(horizon/dt)):
            t = min(step*dt, trajectory.duration_s)
            reference = trajectory.evaluate(t)
            if guidance is not None:
                self._observe(guidance, target_id)
            if not self._step(reference):
                return False
        if np.linalg.norm(self.state.position_m-np.asarray(trajectory.waypoints_m[-1])) > .17:
            return self.abort("segment_did_not_settle")
        if self.time_s <= start:
            return self.abort("clock_did_not_advance")
        return True

    def _plan(self, points, phase, guidance=None, fix=None):
        start_speed = float(np.linalg.norm(self.state.velocity_m_s))
        if start_speed > .04:
            if not self._hold(.8, "SETTLE_BEFORE_REPLAN", self.target_id):
                return None
            start_speed = float(np.linalg.norm(self.state.velocity_m_s))
        if start_speed > .04:
            self.abort("trajectory_transition_not_settled")
            return None
        points = [self.state.position_m.copy(), *points[1:]]
        if guidance is None:
            trajectory, report = generate_trajectory(points, self.world,
                self.vehicle.vehicle, self.limits)
        else:
            trajectory, report = guidance.plan_segment(self.state, fix, points[-1])
        self.reference_reports.append({"time_s": float(self.time_s),
            "phase": phase, "target_id": self.target_id, "report": report,
            "initial_speed_m_s": start_speed,
            "waypoints_world_m": [np.asarray(p).tolist() for p in points]})
        if trajectory is None:
            self.abort(phase.lower()+"_"+str(report["reason"]))
        return trajectory

    def _verified_crossing(self, target_id, direction):
        events = [x for x in physical_crossings(self.records, self.world)
                  if x["classification"] == "opening"]
        expected = EXPECTED_CROSSINGS[:len(events)]
        actual = tuple((x["window_id"], x["direction"]) for x in events)
        if actual != expected or not events or actual[-1] != (target_id, direction):
            return self.abort("crossing_order_or_geometry")
        if events[-1]["entry_time_s"] is None or events[-1]["exit_time_s"] is None:
            return self.abort("incomplete_envelope_crossing")
        return True

    def _static_window(self, target_id, reverse=False):
        direction = "return" if reverse else "outbound"
        guidance = self._guidance(target_id, reverse=reverse)
        if reverse and not self._yaw_to(math.pi, target_id):
            return False
        fix = self._acquire(guidance, target_id)
        if fix is None:
            return False
        sign = -1 if reverse else 1
        normal = fix.estimated_window.normal_world
        if not self._yaw_to(math.atan2((-sign*normal)[1], (-sign*normal)[0]), target_id):
            return False
        entry, _ = guidance.entry_exit(fix)
        trajectory = self._plan([self.state.position_m, entry], "APPROACH_STATIC",
                                guidance, fix)
        if trajectory is None or not self._fly(trajectory, "APPROACH_STATIC", target_id, guidance):
            return False
        fix = self._acquire(guidance, target_id)
        if fix is None:
            return False
        _, exit_point = guidance.entry_exit(fix)
        trajectory = self._plan([self.state.position_m, exit_point], "TRAVERSE_STATIC",
                                guidance, fix)
        if trajectory is None or not self._fly(trajectory, "TRAVERSE_STATIC", target_id, guidance):
            return False
        self.transition("CONFIRM_STATIC_EXIT", target_id)
        return self._verified_crossing(target_id, direction)

    def _delta(self):
        target_id = "delta"
        guidance = self._guidance(target_id, delta=True)
        fix = self._acquire(guidance, target_id)
        if fix is None:
            return False
        lateral = self.raw["guidance"]["delta_corridor_right_m"]
        entry, _ = guidance.entry_exit(fix)
        entry = entry + fix.estimated_window.right_world*lateral
        trajectory = self._plan([self.state.position_m, entry], "APPROACH_DELTA",
                                guidance, fix)
        if trajectory is None or not self._fly(trajectory, "APPROACH_DELTA", target_id, guidance):
            return False
        self.transition("WAIT_FOR_SAFE_PHASE", target_id)
        wait_start = self.time_s
        max_wait = self.raw["guidance"]["max_wait_s"]
        while self.time_s-wait_start < max_wait:
            fix, estimate = self._observe(guidance, target_id, force=True)
            if fix.valid and estimate.valid:
                _, exit_point = guidance.entry_exit(fix)
                exit_point = exit_point + fix.estimated_window.right_world*lateral
                candidate, report = guidance.plan_segment(self.state, fix, exit_point)
                if candidate is not None:
                    choice = choose_crossing(candidate, self.time_s, estimate,
                        fix.estimated_window, self.known_arm_shape, self.world.envelope_m,
                        self.raw["guidance"]["prediction_horizon_s"])
                    self.choices.append({"time_s": float(self.time_s),
                        "valid": choice.valid, "reason": choice.reason,
                        "commit_time_s": choice.commit_time_s,
                        "predicted_clearance_m": choice.predicted_min_clearance_m,
                        "predicted_plane_time_s": choice.predicted_plane_time_s,
                        "estimated_angle_rad": estimate.angle_rad,
                        "estimated_rate_rad_s": estimate.angular_velocity_rad_s,
                        "angle_uncertainty_rad": estimate.angle_uncertainty_rad,
                        "rate_uncertainty_rad_s": estimate.rate_uncertainty_rad_s})
                    if choice.valid and choice.commit_time_s <= self.time_s+.001:
                        self.reference_reports.append({"time_s": float(self.time_s),
                            "phase": "TRAVERSE_DELTA", "target_id": target_id,
                            "report": report,
                            "waypoints_world_m": [list(x) for x in candidate.waypoints_m]})
                        self.waiting_s = self.time_s-wait_start
                        if not self._fly(candidate, "TRAVERSE_DELTA", target_id, guidance):
                            return False
                        self.transition("CONFIRM_DYNAMIC_EXIT", target_id)
                        return self._verified_crossing(target_id, "outbound")
            if not self._hold(self.raw["guidance"]["image_period_s"],
                              "WAIT_FOR_SAFE_PHASE", target_id):
                return False
        self.waiting_s = self.time_s-wait_start
        return self.abort("no_safe_dynamic_interval")

    def _bypass(self):
        route = [self.state.position_m.copy(),
            *[np.asarray(x, dtype=float) for x in self.raw["route"]["bypass_world_m"]]]
        trajectory = self._plan(route, "BYPASS_DELTA")
        if trajectory is None or not self._fly(trajectory, "BYPASS_DELTA"):
            return False
        events = physical_crossings(self.records, self.world)
        if (sum(x["window_id"] == "delta" and x["classification"] == "opening"
                for x in events) != 1 or
                not any(x["window_id"] == "delta" and
                    x["classification"] == "external_plane_bypass" for x in events)):
            return self.abort("bypass_not_verified")
        return True

    def _landing(self):
        home_hover = self.home_m+np.array([0., 0., 1.0])
        trajectory = self._plan([self.state.position_m, home_hover], "RETURN_HOME")
        if trajectory is None or not self._fly(trajectory, "RETURN_HOME"):
            return False
        trajectory = self._plan([self.state.position_m, self.home_m], "LAND")
        if trajectory is None:
            return False
        self.transition("LAND")
        dt = self.raw["guidance"]["step_s"]
        for step in range(math.ceil((trajectory.duration_s+2.0)/dt)):
            reference = trajectory.evaluate(min(step*dt, trajectory.duration_s))
            if not self._step(reference):
                return False
            p, v = self.state.position_m, self.state.velocity_m_s
            landing = self.raw["landing"]
            tilt = self.records[-1]["tilt_deg"]
            if (p[2] <= landing["touchdown_height_m"] and v[2] <= 0 and
                np.linalg.norm(p[:2]-self.home_m[:2]) <= landing["horizontal_error_m"] and
                np.linalg.norm(v[:2]) <= landing["horizontal_speed_m_s"] and
                abs(v[2]) <= landing["vertical_speed_m_s"] and
                np.linalg.norm(self.state.body_rate_rad_s) <= landing["body_rate_rad_s"] and
                tilt <= landing["tilt_deg"]):
                self.landed = True
                self.transition("COMPLETE")
                return True
        return self.abort("landing_tolerance_unmet")

    def run(self):
        self.transition("TAKEOFF")
        takeoff = np.array([0., 0., 1.6])
        trajectory = self._plan([self.state.position_m, takeoff], "TAKEOFF")
        if trajectory is None or not self._fly(trajectory, "TAKEOFF"):
            return self.metrics()
        for target in ("alpha", "beta", "gamma"):
            if not self._static_window(target):
                return self.metrics()
        if not self._delta() or not self._bypass():
            return self.metrics()
        for target in ("gamma", "beta", "alpha"):
            if not self._static_window(target, reverse=True):
                return self.metrics()
        self._landing()
        return self.metrics()

    def metrics(self):
        events = physical_crossings(self.records, self.world)
        for event in events:
            if event["classification"] != "opening":
                continue
            valid = [x for x in self.observations if x["target_id"] == event["window_id"]
                and x["window_valid"] and x["time_s"] <= event["entry_time_s"]]
            latest = valid[-1] if valid else None
            event["last_valid_image_pose_time_s"] = latest["time_s"] if latest else None
            event["image_pose_available_before_entry"] = latest is not None
            if event["window_id"] == "delta":
                tracked = [x for x in self.observations if x["target_id"] == "delta"
                    and x["tracker_valid"] and x["time_s"] <= event["entry_time_s"]]
                event["arm_tracker_available_before_entry"] = bool(tracked)
        openings = [x for x in events if x["classification"] == "opening"]
        order = [(x["window_id"], x["direction"]) for x in openings]
        p, v = self.state.position_m, self.state.velocity_m_s
        complete = (self.failure is None and self.landed and
            tuple(order) == EXPECTED_CROSSINGS and
            sum(x["classification"] == "external_plane_bypass" for x in events) >= 1 and
            self.min_static_clearance_m > self.raw["acceptance"]["minimum_static_clearance_m"] and
            self.min_dynamic_clearance_m > self.raw["acceptance"]["minimum_dynamic_clearance_m"])
        tracking = [x.get("tracking_error_m", 0.) for x in self.records]
        pose_errors = [x["translation_error_m"] for x in self.truth_scoring
                       if x["translation_error_m"] is not None]
        angle_errors = [x["arm_angle_error_rad"] for x in self.truth_scoring
                        if x["arm_angle_error_rad"] is not None]
        rate_errors = [x["arm_rate_error_rad_s"] for x in self.truth_scoring
                       if x["arm_rate_error_rad_s"] is not None]
        thrust_limit = self.vehicle.controller.max_total_thrust_n
        torque_limit = self.vehicle.controller.max_torque_nm
        saturation_count = sum(x["thrust_n"] >= thrust_limit-1e-8 or
            np.any(abs(x["torque_nm"])-torque_limit >= -1e-8)
            for x in self.records[1:])
        return {"condition": vars(self.condition), "success": bool(complete),
            "failure_reason": self.failure, "phase": self.phase,
            "duration_s": float(self.time_s), "landing_success": self.landed,
            "home_world_m": self.home_m.tolist(), "landing_position_world_m": p.tolist(),
            "landing_velocity_world_m_s": v.tolist(),
            "landing_horizontal_error_m": float(np.linalg.norm(p[:2]-self.home_m[:2])),
            "crossing_order": order, "crossing_events": events,
            "delta_opening_count": sum(x["window_id"] == "delta" for x in openings),
            "delta_external_bypass_count": sum(x["classification"] == "external_plane_bypass"
                for x in events), "minimum_static_clearance_m": float(self.min_static_clearance_m),
            "minimum_dynamic_clearance_m": float(self.min_dynamic_clearance_m),
            "peak_tracking_error_m": float(self.peak_tracking_error_m),
            "rms_tracking_error_m": float(np.sqrt(np.mean(np.square(tracking)))),
            "peak_speed_m_s": float(self.peak_speed_m_s),
            "peak_acceleration_m_s2": float(self.peak_acceleration_m_s2),
            "minimum_thrust_n": float(self.min_thrust_n),
            "peak_thrust_n": float(self.peak_thrust_n),
            "peak_torque_nm": self.peak_torque_nm.tolist(),
            "peak_tilt_deg": float(self.peak_tilt_deg),
            "control_saturation_step_count": int(saturation_count),
            "distance_m": float(sum(np.linalg.norm(
                b["position_m"]-a["position_m"]) for a, b in zip(
                    self.records[:-1], self.records[1:]))),
            "window_observation_count": len(self.observations),
            "valid_window_pose_count": sum(x["window_valid"] for x in self.observations),
            "mean_window_pose_translation_error_m": float(np.mean(pose_errors))
                if pose_errors else None,
            "valid_arm_observation_count": sum(x["arm_valid"] is True
                for x in self.observations),
            "arm_angle_rmse_deg": float(np.degrees(np.sqrt(np.mean(np.square(angle_errors)))))
                if angle_errors else None,
            "arm_rate_rmse_rad_s": float(np.sqrt(np.mean(np.square(rate_errors))))
                if rate_errors else None,
            "arm_waiting_s": float(self.waiting_s)}


def save_trial(mission: CompleteMission, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    metrics = mission.metrics()
    scene = {"world_frame": "ENU", "body_frame": "FLU", "camera_frame": "RDF",
        "camera_fx_fy_cx_cy_px": [mission.camera.fx_px, mission.camera.fy_px,
            mission.camera.cx_px, mission.camera.cy_px],
        "camera_image_width_height_px": [mission.camera.width_px, mission.camera.height_px],
        "rotation_camera_to_body": mission.camera.rotation_camera_to_body.tolist(),
        "translation_camera_in_body_m": mission.camera.translation_camera_in_body_m.tolist(),
        "vehicle_visual_model": "M10 FLU four-arm/rotor display geometry",
        "vehicle_collision_envelope_m": mission.world.envelope_m,
        "windows": [{"window_id": wall.window.model.window_id,
            "center_world_m": wall.window.center_world_m.tolist(),
            "right_world": wall.window.right_world.tolist(),
            "up_world": wall.window.up_world.tolist(),
            "opening_width_m": wall.window.model.width_m,
            "opening_height_m": wall.window.model.height_m,
            "frame_thickness_m": wall.window.frame_thickness_m,
            "outer_width_m": wall.outer_width_m,
            "outer_height_m": wall.outer_height_m}
            for wall in mission.world.walls],
        "arm": {"window_id": "delta", "pivot_right_m": mission.arm.shape.pivot_right_m,
            "pivot_up_m": mission.arm.shape.pivot_up_m,
            "length_m": mission.arm.shape.length_m,
            "radius_m": mission.arm.shape.arm_radius_m,
            "initial_angle_rad": mission.arm.initial_angle_rad,
            "angular_velocity_rad_s": mission.arm.angular_velocity_rad_s}}
    (directory / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (directory / "transitions.json").write_text(json.dumps(mission.transitions, indent=2), encoding="utf-8")
    (directory / "operational_observations.json").write_text(json.dumps(
        mission.observations, indent=2), encoding="utf-8")
    (directory / "evaluation_truth.json").write_text(json.dumps(
        mission.truth_scoring, indent=2), encoding="utf-8")
    (directory / "reference_reports.json").write_text(json.dumps(
        mission.reference_reports, indent=2), encoding="utf-8")
    (directory / "timing_choices.json").write_text(json.dumps(
        mission.choices, indent=2), encoding="utf-8")
    fields = ("time_s", "position_m", "velocity_m_s", "quaternion_bw",
        "body_rate_rad_s", "reference_position_m", "reference_velocity_m_s",
        "reference_acceleration_m_s2", "desired_quaternion_bw", "thrust_n",
        "torque_nm", "static_clearance_m", "dynamic_clearance_m")
    np.savez_compressed(directory / "physical_trace.npz", **{
        key: np.asarray([row[key] for row in mission.records]) for key in fields},
        phase=np.asarray([row["phase"] for row in mission.records]),
        target_id=np.asarray([row["target_id"] or "" for row in mission.records]))
    if mission.frames:
        np.savez_compressed(directory / "camera_frames.npz",
            image_rgb=np.asarray([row["image"] for row in mission.frames]),
            time_s=np.asarray([row["time_s"] for row in mission.frames]),
            target_id=np.asarray([row["target_id"] for row in mission.frames]),
            phase=np.asarray([row["phase"] for row in mission.frames]),
            corners_uv=np.asarray([row["corners_uv"] if row["corners_uv"] is not None
                else np.full((4, 2), np.nan) for row in mission.frames]),
            arm_angle_rad=np.asarray([row["arm_angle_rad"] if row["arm_angle_rad"] is not None
                else np.nan for row in mission.frames]),
            pose_valid=np.asarray([row["pose_valid"] for row in mission.frames]))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mission_m11.toml")
    parser.add_argument("--output-dir", default="outputs/mission_m11")
    parser.add_argument("--hero-only", action="store_true")
    parser.add_argument("--failure-cases-only", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args(argv)
    raw, vehicle, limits, camera, world, arm, colors, study = load_m11_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    cases = zip(raw["experiments"]["trial_seeds"], raw["experiments"]["initial_phases_rad"],
        raw["experiments"]["noise_std_rgb"], raw["experiments"]["dropout_probability"])
    if not args.failure_cases_only and not args.sensitivity_only:
        results = []
        for index, (seed, phase, noise, dropout) in enumerate(cases):
            if args.hero_only and index:
                break
            mission = CompleteMission(raw, vehicle, limits, camera, world, arm, colors,
                MissionCondition(seed, phase, noise, dropout), capture_frames=index == 0)
            mission.run()
            results.append(save_trial(mission, output / f"trial_{seed}"))
            print(seed, results[-1]["success"], results[-1]["failure_reason"],
                  results[-1]["crossing_order"], flush=True)
        aggregate = {"attempted": len(results), "completed": sum(x["success"] for x in results),
            "trials": results}
        (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    else:
        aggregate = None
    if not args.hero_only and not args.sensitivity_only:
        f = raw["failure_cases"]
        failure_cases = (
            ("visual_loss", MissionCondition(f["visual_loss_seed"], .9,
                forced_visual_loss=True)),
            ("blocked_arm", MissionCondition(f["blocked_arm_seed"],
                f["blocked_arm_phase_rad"], stationary_arm=True)))
        failures = []
        for name, condition in failure_cases:
            mission = CompleteMission(raw, vehicle, limits, camera, world, arm, colors,
                condition, capture_frames=name == "blocked_arm")
            mission.run()
            result = save_trial(mission, output / f"failure_{name}")
            failures.append({"case": name, **result})
            print(name, result["success"], result["failure_reason"], flush=True)
        (output / "failure_cases.json").write_text(json.dumps(failures, indent=2),
                                                     encoding="utf-8")
    if not args.hero_only and not args.failure_cases_only:
        s = raw["sensitivity"]
        shape = ArmShape(arm.shape.pivot_right_m, arm.shape.pivot_up_m,
            s["arm_length_m"], s["arm_radius_m"])
        larger_arm = RotatingArm(arm.window, shape, s["initial_phase_rad"],
                                arm.angular_velocity_rad_s)
        sensitivity_geometry = geometry_study(raw, world, larger_arm, vehicle, limits)
        if not sensitivity_geometry["feasible"]:
            raise ValueError("sensitivity arm is geometrically infeasible")
        mission = CompleteMission(raw, vehicle, limits, camera, world, larger_arm, colors,
            MissionCondition(s["seed"], s["initial_phase_rad"]),
            known_arm_shape=shape)
        mission.run()
        sensitivity = {"geometry": sensitivity_geometry,
            "trial": save_trial(mission, output / "sensitivity_larger_arm")}
        (output / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2),
                                                   encoding="utf-8")
        print("sensitivity", sensitivity["trial"]["success"],
              sensitivity["trial"]["failure_reason"], flush=True)
    return aggregate


if __name__ == "__main__":
    main()
