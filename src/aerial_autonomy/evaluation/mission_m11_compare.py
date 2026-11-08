"""Reproducible comparison of the preserved and centered-arm M11 trials."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from aerial_autonomy.evaluation.mission_m11 import load_m11_config


METRICS = ("success", "failure_reason", "duration_s", "landing_success",
    "minimum_static_clearance_m", "minimum_dynamic_clearance_m",
    "arm_angle_rmse_deg", "arm_rate_rmse_rad_s", "arm_waiting_s",
    "peak_speed_m_s", "peak_acceleration_m_s2", "peak_thrust_n",
    "peak_tilt_deg", "peak_tracking_error_m", "delta_opening_count",
    "delta_external_bypass_count")


def _read_result(config_path, output_dir):
    raw, _, _, _, _, arm, _, study = load_m11_config(config_path)
    root = Path(output_dir)
    if root.suffix.lower() == ".zip":
        with ZipFile(root) as archive:
            saved = json.loads(archive.read("outputs/mission_m11/aggregate.json"))
    else:
        saved = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    trials = {}
    for result in saved["trials"]:
        seed = str(result["condition"]["seed"])
        delta = next((event for event in result["crossing_events"]
                      if event["window_id"] == "delta" and
                      event["classification"] == "opening"), None)
        trials[seed] = {"seed": int(seed), **{name: result[name] for name in METRICS},
                        "delta_plane_crossing_time_s":
                            delta["plane_crossing_time_s"] if delta else None,
                        "delta_envelope_entry_time_s":
                            delta["entry_time_s"] if delta else None,
                        "delta_envelope_exit_time_s":
                            delta["exit_time_s"] if delta else None}
    shape = arm.shape
    width, height = arm.window.model.width_m, arm.window.model.height_m
    geometry = {"window_width_m": width, "window_height_m": height,
        "pivot_right_m": shape.pivot_right_m, "pivot_up_m": shape.pivot_up_m,
        "arm_length_m": shape.length_m, "arm_radius_m": shape.arm_radius_m,
        "arm_length_to_window_width": shape.length_m/width,
        "arm_length_to_window_height": shape.length_m/height,
        "arm_reach_to_window_width": (shape.length_m+shape.arm_radius_m)/width,
        "arm_reach_to_window_height": (shape.length_m+shape.arm_radius_m)/height,
        "minimum_tip_to_frame_m": study["arm_tip_to_frame_m"],
        "corridor_right_m": raw["guidance"]["delta_corridor_right_m"],
        "corridor_frame_clearance_after_envelope_m":
            study["vehicle_corridor_to_frame_m"],
        "safe_phase_fraction_at_plane":
            study["corridor_safe_phase_fraction_at_plane"],
        "safe_phase_fraction_full_predicted_crossing":
            study["full_crossing_safe_phase_fraction"],
        "full_predicted_crossing_duration_s": study["full_crossing_duration_s"],
        "full_predicted_max_arm_clearance_m":
            study["full_crossing_max_predicted_arm_clearance_m"]}
    completed = [value for value in trials.values() if value["success"]]
    summary = {"attempted": len(trials), "completed": len(completed),
        "landed": sum(value["landing_success"] for value in trials.values()),
        "minimum_static_clearance_m": min(value["minimum_static_clearance_m"]
                                          for value in completed) if completed else None,
        "minimum_dynamic_clearance_m": min(value["minimum_dynamic_clearance_m"]
                                           for value in completed) if completed else None,
        "maximum_duration_s": max(value["duration_s"] for value in completed)
                              if completed else None,
        "maximum_speed_m_s": max(value["peak_speed_m_s"] for value in completed)
                             if completed else None,
        "maximum_acceleration_m_s2": max(value["peak_acceleration_m_s2"]
                                         for value in completed) if completed else None,
        "maximum_thrust_n": max(value["peak_thrust_n"] for value in completed)
                            if completed else None,
        "maximum_tilt_deg": max(value["peak_tilt_deg"] for value in completed)
                            if completed else None}
    return {"config": str(config_path), "output_dir": str(output_dir),
            "geometry": geometry, "summary": summary, "trials": trials}


def compare(original_config, original_output, revised_config, revised_output,
            original_package=None):
    baseline = _read_result(original_config, original_output)
    revised = _read_result(revised_config, revised_output)
    if set(baseline["trials"]) != set(revised["trials"]):
        raise ValueError("original and revised seed sets differ")
    if original_package is not None:
        baseline["original_package_sha256"] = hashlib.sha256(
            Path(original_package).read_bytes()).hexdigest()
    return {"baseline_label": "original submitted M11 off-center arm",
        "revised_label": "centered enlarged M11 arm",
        "definition": "Clearances are extra separation beyond the 0.25 m modeled vehicle envelope. Safe phase fractions scan one full arm revolution; full-crossing predictions include M10's minimum 0.045 rad angle and 0.025 rad/s rate uncertainty floors and the complete M7 reference duration. Actual trial scheduling uses its measured uncertainty.",
        "original": baseline, "revised": revised,
        "matched_seed_count": len(baseline["trials"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", default="configs/mission_m11_original.toml")
    parser.add_argument("--original-output", default="M11_independent_review_original.zip")
    parser.add_argument("--revised-config", default="configs/mission_m11.toml")
    parser.add_argument("--revised-output", default="outputs/mission_m11")
    parser.add_argument("--original-package", default="M11_independent_review_original.zip")
    parser.add_argument("--output", default="outputs/mission_m11/comparison_original_centered.json")
    args = parser.parse_args(argv)
    result = compare(args.original_config, args.original_output, args.revised_config,
                     args.revised_output, args.original_package)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({"matched_seeds": result["matched_seed_count"],
           "original_completed": result["original"]["summary"]["completed"],
           "revised_completed": result["revised"]["summary"]["completed"]})


if __name__ == "__main__":
    main()
