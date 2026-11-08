"""Independent replay of saved M11 plant inputs, collisions and openings."""

import argparse
import json
from pathlib import Path

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import EXPECTED_CROSSINGS, physical_crossings
from aerial_autonomy.perception.window_m10_timing import wrap_angle
from aerial_autonomy.perception.window_m10_arm import RotatingArm, swept_arm_clearance_m
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, advance_state


def audit_saved_trial(folder, vehicle, world, base_arm, write=True):
    folder = Path(folder)
    with np.load(folder / "physical_trace.npz") as archive:
        trace = {key: archive[key] for key in archive.files}
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    condition = metrics["condition"]
    arm = RotatingArm(base_arm.window, base_arm.shape,
        condition["initial_phase_rad"],
        0.0 if condition["stationary_arm"] else base_arm.angular_velocity_rad_s)
    time = trace["time_s"]
    position = trace["position_m"]
    velocity = trace["velocity_m_s"]
    quat = trace["quaternion_bw"]
    rates = trace["body_rate_rad_s"]
    thrust = trace["thrust_n"]
    torque = trace["torque_nm"]
    residual = 0.0
    static_min = float("inf")
    wall_min = float("inf")
    workspace_min = float("inf")
    arm_min = float("inf")
    rows = []
    for i in range(len(time)):
        if i:
            dt = float(time[i]-time[i-1])
            prior = VehicleState(position[i-1], velocity[i-1], quat[i-1], rates[i-1])
            replay = advance_state(prior, Wrench(float(thrust[i]), torque[i]),
                                   vehicle.vehicle, dt)
            residual = max(residual, float(np.max(abs(replay.position_m-position[i]))),
                           float(np.max(abs(replay.velocity_m_s-velocity[i]))),
                           float(np.max(abs(replay.quaternion_bw-quat[i]))),
                           float(np.max(abs(replay.body_rate_rad_s-rates[i]))))
            static = world.segment_clearance_m(position[i-1], position[i])
            wall_min = min(wall_min, *(wall.segment_clearance_m(
                position[i-1], position[i], world.envelope_m) for wall in world.walls))
            for endpoint in (position[i-1], position[i]):
                workspace_min = min(workspace_min,
                    float(np.min(endpoint-np.asarray(world.workspace.minimum)-world.envelope_m)),
                    float(np.min(np.asarray(world.workspace.maximum)-endpoint-world.envelope_m)))
            dynamic = swept_arm_clearance_m(position[i-1], position[i],
                float(time[i-1]), float(time[i]), arm, world.envelope_m)
            static_min = min(static_min, static)
            arm_min = min(arm_min, dynamic)
            residual = max(residual,
                abs(static-float(trace["static_clearance_m"][i])),
                abs(dynamic-float(trace["dynamic_clearance_m"][i])))
        rows.append({"time_s": float(time[i]), "position_m": position[i],
            "phase": str(trace["phase"][i]),
            "static_clearance_m": float(trace["static_clearance_m"][i]),
            "dynamic_clearance_m": float(trace["dynamic_clearance_m"][i])})
    crossings = physical_crossings(rows, world)
    opening = [(x["window_id"], x["direction"]) for x in crossings
               if x["classification"] == "opening"]
    external = sum(x["classification"] == "external_plane_bypass" for x in crossings)
    timing = None
    choice_path = folder / "timing_choices.json"
    if choice_path.exists():
        choices = json.loads(choice_path.read_text(encoding="utf-8"))
        selected = next((x for x in choices if x["valid"] and
            x["commit_time_s"] <= x["time_s"]+.001), None)
        actual_delta = next((x for x in crossings if x["window_id"] == "delta" and
            x["classification"] == "opening"), None)
        if selected and actual_delta:
            predicted_phase = (selected["estimated_angle_rad"] +
                selected["estimated_rate_rad_s"]*(selected["predicted_plane_time_s"]-
                selected["time_s"]))
            actual_phase = arm.angle_at(actual_delta["plane_crossing_time_s"])
            timing = {"commit_time_s": selected["commit_time_s"],
                "predicted_plane_time_s": selected["predicted_plane_time_s"],
                "actual_plane_time_s": actual_delta["plane_crossing_time_s"],
                "predicted_arm_phase_rad": float(wrap_angle(predicted_phase)),
                "actual_arm_phase_rad": float(wrap_angle(actual_phase)),
                "phase_error_rad": float(wrap_angle(predicted_phase-actual_phase)),
                "predicted_minimum_dynamic_clearance_m": selected["predicted_clearance_m"],
                "angle_uncertainty_rad": selected.get("angle_uncertainty_rad"),
                "rate_uncertainty_rad_s": selected.get("rate_uncertainty_rad_s")}
    result = {"sample_count": len(time), "time_monotonic": bool(np.all(np.diff(time) > 0)),
        "maximum_plant_and_clearance_replay_residual": residual,
        "minimum_replayed_static_clearance_m": static_min,
        "minimum_replayed_wall_clearance_m": wall_min,
        "minimum_replayed_workspace_clearance_m": workspace_min,
        "minimum_replayed_dynamic_clearance_m": arm_min,
        "minimum_vehicle_center_height_m": float(np.min(position[:, 2])),
        "opening_order": opening, "external_delta_plane_crossings": external,
        "control_input_applied_each_step": bool(residual < 1e-7),
        "full_mission_geometry_valid": (tuple(opening) == EXPECTED_CROSSINGS and
            external >= 1 and static_min > 0 and arm_min > 0),
        "recorded_success": metrics["success"]}
    result["timing_evidence"] = timing
    if write:
        (folder / "independent_audit.json").write_text(json.dumps(result, indent=2),
                                                        encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mission_m11.toml")
    parser.add_argument("--output-dir", default="outputs/mission_m11")
    args = parser.parse_args(argv)
    from aerial_autonomy.evaluation.mission_m11 import load_m11_config
    raw, vehicle, _, _, world, arm, _, _ = load_m11_config(args.config)
    root = Path(args.output_dir)
    results = {}
    for seed in raw["experiments"]["trial_seeds"]:
        results[f"trial_{seed}"] = audit_saved_trial(root / f"trial_{seed}",
                                                      vehicle, world, arm)
    for name in ("failure_visual_loss", "failure_blocked_arm"):
        results[name] = audit_saved_trial(root / name, vehicle, world, arm)
    s = raw["sensitivity"]
    from aerial_autonomy.perception.window_m10_arm import ArmShape
    shape = ArmShape(arm.shape.pivot_right_m, arm.shape.pivot_up_m,
                     s["arm_length_m"], s["arm_radius_m"])
    larger_arm = RotatingArm(arm.window, shape, s["initial_phase_rad"],
                             arm.angular_velocity_rad_s)
    results["sensitivity_larger_arm"] = audit_saved_trial(
        root / "sensitivity_larger_arm", vehicle, world, larger_arm)
    (root / "independent_audit_summary.json").write_text(json.dumps(results, indent=2),
                                                          encoding="utf-8")
    print({name: {"control_replay": value["control_input_applied_each_step"],
                  "geometry_valid": value["full_mission_geometry_valid"]}
           for name, value in results.items()})


if __name__ == "__main__":
    main()
