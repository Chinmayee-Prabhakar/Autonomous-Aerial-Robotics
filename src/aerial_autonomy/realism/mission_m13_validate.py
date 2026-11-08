"""Aggregate every predefined M13 final and negative attempt without hiding failures."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from aerial_autonomy.evaluation.mission_m11 import EXPECTED_CROSSINGS
from aerial_autonomy.realism.mission_m13 import file_sha256


def validate(root: Path = Path("outputs/mission_m13"),
             config_path: Path = Path("configs/mission_m13.toml")) -> dict:
    root = Path(root)
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    spec = config["final_trials"]
    conditions = list(zip(spec["seeds"], spec["arm_initial_phases_rad"],
                          spec["dropout_probabilities"]))
    if len(conditions) != 3:
        raise ValueError("M13 final condition count changed")
    digest = file_sha256(config_path)
    trials = []
    for seed, phase, dropout in conditions:
        folder = root / f"final_trial_{seed}"
        summary = json.loads((folder / "summary.json").read_text())
        audit = json.loads((folder / "independent_audit.json").read_text())
        evaluation = json.loads((folder / "offline_evaluation.json").read_text())
        metrics = summary["metrics"]
        actual = metrics["condition"]
        condition_match = (actual["seed"] == seed and
            abs(actual["initial_phase_rad"]-phase) < 1e-12 and
            abs(actual["dropout_probability"]-dropout) < 1e-12 and
            summary["configuration"]["sha256"] == digest)
        order = tuple(tuple(item) for item in metrics["crossing_order"])
        passed = bool(condition_match and summary["trial_kind"] == "final" and
            summary["exception"] is None and metrics["success"] and
            metrics["landing_success"] and order == EXPECTED_CROSSINGS and
            metrics["delta_opening_count"] == 1 and
            metrics["delta_external_bypass_count"] == 1 and
            min(metrics["minimum_static_clearance_m"],
                metrics["minimum_dynamic_clearance_m"],
                summary["minimum_decor_clearance_m"]) > 0 and
            audit["passed"] and audit["mission_success"] and
            audit["rgb_removal_changes_guidance"] and
            audit["physical_steps"] > 0 and
            evaluation["total_observations"] == summary["images"])
        trials.append({"seed": seed, "arm_initial_phase_rad": phase,
            "dropout_probability": dropout, "passed": passed,
            "condition_match": condition_match,
            "success": metrics["success"], "failure_reason": metrics["failure_reason"],
            "duration_s": metrics["duration_s"], "images": summary["images"],
            "arm_waiting_s": metrics["arm_waiting_s"],
            "minimum_static_clearance_m": metrics["minimum_static_clearance_m"],
            "minimum_dynamic_clearance_m": metrics["minimum_dynamic_clearance_m"],
            "minimum_decor_clearance_m": summary["minimum_decor_clearance_m"],
            "landing_position_world_m": metrics["landing_position_world_m"],
            "false_operational_poses": evaluation["false_operational_poses"],
            "missed_in_frame_poses": evaluation["missed_in_frame_poses"]})
    negatives = []
    for name in ("visual_loss", "blocked_arm"):
        folder = root / f"negative_{name}"
        summary = json.loads((folder / "summary.json").read_text())
        audit = json.loads((folder / "independent_audit.json").read_text())
        metrics = summary["metrics"]
        passed = (summary["trial_kind"] == "negative" and
                  not metrics["success"] and audit["passed"])
        if name == "visual_loss":
            passed = (passed and not metrics["crossing_order"] and
                      metrics["failure_reason"] == "persistent_visual_loss")
        else:
            passed = (passed and metrics["delta_opening_count"] == 0 and
                      metrics["failure_reason"] == "no_safe_dynamic_interval")
        negatives.append({"case": name, "passed": passed,
            "failure_reason": metrics["failure_reason"],
            "crossing_order": metrics["crossing_order"],
            "arm_waiting_s": metrics["arm_waiting_s"]})
    result = {"schema": "m13_final_aggregate_v1", "attempted": len(trials),
              "completed": sum(item["success"] for item in trials),
              "all_predefined_cases_passed": all(item["passed"] for item in trials+negatives),
              "trials": trials, "negative_cases": negatives}
    (root / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
