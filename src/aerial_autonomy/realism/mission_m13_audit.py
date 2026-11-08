"""Independent physical and RGB-provenance audit of a saved M13 attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import (
    CompleteMission, MissionCondition, load_m11_config, physical_crossings,
)
from aerial_autonomy.perception.window_m10_arm import RotatingArm, swept_arm_clearance_m
from aerial_autonomy.perception.window_m10_timing import AngularTracker
from aerial_autonomy.planning.geometry import AABB, segment_aabb_distance
from aerial_autonomy.realism.detector import (
    AdaptiveColorWindowDetector, AdaptivePixelArmDetector,
)
from aerial_autonomy.realism.mission_m13 import ComplementaryWindowDetector
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state
from PIL import Image


def audit(directory: Path, detector_policy: str = "m9_then_m12") -> dict:
    root = Path(directory)
    trace = np.load(root / "trial" / "physical_trace.npz")
    observations = json.loads((root / "trial" / "operational_observations.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "static_scene_plan.json").read_text())
    captures = [json.loads(line) for line in (root / "captures.jsonl").read_text().splitlines()]
    raw, vehicle, limits, camera, world, nominal_arm, colors, _ = load_m11_config(
        "configs/mission_m11.toml")
    condition = summary["metrics"]["condition"]
    arm = RotatingArm(nominal_arm.window, nominal_arm.shape,
        condition["initial_phase_rad"],
        0.0 if condition["stationary_arm"] else nominal_arm.angular_velocity_rad_s)
    boxes = [AABB(tuple(item["minimum_m"]), tuple(item["maximum_m"]))
             for item in plan["decor"]]
    t = np.asarray(trace["time_s"])
    if len(t) < 2 or abs(t[0]) > 1e-12 or not np.all(np.diff(t) > 0):
        raise AssertionError("physical clock reset or nonmonotonic")
    if np.linalg.norm(trace["position_m"][0]) > 1e-12:
        raise AssertionError("M13 did not start at home")
    residual = 0.0
    minimum_decor = float("inf")
    for i in range(1, len(t)):
        start = VehicleState(trace["position_m"][i-1], trace["velocity_m_s"][i-1],
                             trace["quaternion_bw"][i-1], trace["body_rate_rad_s"][i-1])
        wrench = Wrench(float(trace["thrust_n"][i]), trace["torque_nm"][i])
        predicted = advance_state(start, wrench, vehicle.vehicle, float(t[i]-t[i-1]))
        residual = max(residual, *(float(np.max(abs(getattr(predicted, key)-trace[key][i])))
            for key in ("position_m", "velocity_m_s", "quaternion_bw", "body_rate_rad_s")))
        p, q = trace["position_m"][i-1], trace["position_m"][i]
        static = world.segment_clearance_m(p, q)
        dynamic = swept_arm_clearance_m(p, q, float(t[i-1]), float(t[i]),
                                        arm, world.envelope_m)
        decor = min((segment_aabb_distance(p, q, box)-world.envelope_m
                     for box in boxes), default=float("inf"))
        minimum_decor = min(minimum_decor, decor)
        if abs(static-float(trace["static_clearance_m"][i])) > 1e-7:
            raise AssertionError(f"static clearance mismatch at step {i}")
        if abs(dynamic-float(trace["dynamic_clearance_m"][i])) > 1e-7:
            raise AssertionError(f"moving-arm clearance mismatch at step {i}")
    if residual > 1e-7:
        raise AssertionError(f"plant replay residual {residual}")
    if len(captures) != len(observations):
        raise AssertionError("capture/observation count mismatch")
    by_id = {item["image_id"]: item for item in captures}
    if len(by_id) != len(captures):
        raise AssertionError("duplicate image IDs")
    if captures and "scene_sha256" in captures[0]:
        scene_digest = hashlib.sha256(Path(captures[0]["scene_path"].replace("\\", "/")).read_bytes()).hexdigest()
        plan_digest = hashlib.sha256((root / "static_scene_plan.json").read_bytes()).hexdigest()
        if any(item.get("scene_sha256") != scene_digest or
               item.get("plan_sha256") != plan_digest for item in captures):
            raise AssertionError("capture scene/configuration identity mismatch")
    for previous, current in zip(captures, captures[1:]):
        if current["capture_time_s"] < previous["capture_time_s"]:
            raise AssertionError("out-of-order RGB capture")
    replay_mission = CompleteMission(raw, vehicle, limits, camera, world,
        nominal_arm, colors, MissionCondition(condition["seed"],
            condition["initial_phase_rad"]))
    if detector_policy == "m9_then_m12":
        detector = ComplementaryWindowDetector(colors)
    elif detector_policy == "m12_only":
        detector = AdaptiveColorWindowDetector(colors)
    else:
        raise ValueError("unknown detector policy")
    arm_detector = AdaptivePixelArmDetector()
    tracker = AngularTracker(raw["guidance"]["max_observation_age_s"])
    transitions = json.loads((root / "trial" / "transitions.json").read_text())
    bypass_times = [item["time_s"] for item in transitions
                    if item["state"] == "BYPASS_DELTA"]
    bypass_time = bypass_times[0] if bypass_times else float("inf")
    guidance_by_key = {}
    replayed_valid = 0
    rgb_removal_checked = False
    for observation in observations:
        capture = by_id[observation["image_id"]]
        image_path = Path(capture["image_path"].replace("\\", "/"))
        if not image_path.is_file() or hashlib.sha256(image_path.read_bytes()).hexdigest() != capture["sha256"]:
            raise AssertionError("RGB bytes differ from capture hash")
        if (abs(observation["time_s"]-capture["capture_time_s"]) > 1e-8 or
                observation["guidance_update_time_s"]+1e-8 < capture["capture_time_s"] or
                observation["image_sha256"] != capture["sha256"]):
            raise AssertionError("image/guidance time or hash mismatch")
        i = int(np.argmin(abs(t-capture["capture_time_s"])))
        if (abs(float(t[i])-capture["capture_time_s"]) > 1e-7 or
                np.linalg.norm(trace["position_m"][i]-capture["vehicle_position_m"]) > 1e-7 or
                np.linalg.norm(trace["quaternion_bw"][i]-capture["vehicle_quaternion_bw"]) > 1e-7):
            raise AssertionError("capture did not use current physical pose")
        target = observation["target_id"]
        reverse = observation["time_s"] > bypass_time
        key = (target, reverse)
        if key not in guidance_by_key:
            guidance = replay_mission._guidance(target, reverse=reverse,
                                                delta=target == "delta")
            guidance.detector = detector
            guidance_by_key[key] = guidance
        state = VehicleState(trace["position_m"][i], trace["velocity_m_s"][i],
                             trace["quaternion_bw"][i], trace["body_rate_rad_s"][i])
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        sensed = np.zeros_like(rgb) if observation["dropout"] else rgb
        fix = guidance_by_key[key].perceive(sensed, state, target,
                                             observation["time_s"])
        if fix.valid != observation["window_valid"]:
            raise AssertionError("saved RGB does not reproduce window validity")
        if fix.valid:
            replayed_valid += 1
            if np.linalg.norm(fix.estimated_window.center_world_m-
                    observation["estimated_center_world_m"]) > 1e-7:
                raise AssertionError("saved RGB does not reproduce window pose")
            if not rgb_removal_checked:
                blank = guidance_by_key[key].perceive(np.zeros_like(rgb), state,
                                                        target, observation["time_s"])
                if blank.valid:
                    raise AssertionError("blank RGB unexpectedly retained guidance")
                rgb_removal_checked = True
        if target == "delta":
            arm_fix = arm_detector.detect(sensed, fix, observation["time_s"])
            tracker.update(arm_fix)
            estimate = tracker.estimate(observation["time_s"])
            if (arm_fix.valid != observation["arm_valid"] or
                    estimate.valid != observation["tracker_valid"]):
                raise AssertionError("saved RGB does not reproduce arm tracking")
    rows = [{"time_s": float(t[i]), "position_m": trace["position_m"][i],
             "phase": str(trace["phase"][i]),
             "static_clearance_m": float(trace["static_clearance_m"][i]),
             "dynamic_clearance_m": float(trace["dynamic_clearance_m"][i])}
            for i in range(len(t))]
    events = physical_crossings(rows, world)
    result = {"schema": "m13_saved_attempt_audit_v1", "passed": True,
              "detector_policy": detector_policy,
              "mission_success": summary["metrics"]["success"],
              "physical_steps": len(t)-1, "rgb_captures": len(captures),
              "replayed_valid_window_poses": replayed_valid,
              "rgb_removal_changes_guidance": rgb_removal_checked,
              "maximum_plant_replay_residual": residual,
              "minimum_static_clearance_m": float(min(trace["static_clearance_m"][1:])),
              "minimum_dynamic_clearance_m": float(min(trace["dynamic_clearance_m"][1:])),
              "minimum_decor_clearance_m": minimum_decor,
              "crossings": events}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--detector-policy", choices=("m9_then_m12", "m12_only"),
                        default="m9_then_m12")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.directory), args.detector_policy), indent=2))


if __name__ == "__main__":
    main()
