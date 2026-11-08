"""Offline geometric scoring of saved M13 RGB observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.realism.bridge import project_blender_camera, window_face_corners


def evaluate(directory: Path) -> dict:
    root = Path(directory)
    observations = json.loads((root / "trial" / "operational_observations.json").read_text())
    truth = json.loads((root / "trial" / "evaluation_truth.json").read_text())
    captures = [json.loads(line) for line in (root / "captures.jsonl").read_text().splitlines()]
    if len(observations) != len(captures) or len(truth) != len(captures):
        raise AssertionError("scoring record count mismatch")
    _, _, _, camera, world, _, _, _ = load_m11_config("configs/mission_m11.toml")
    windows = {wall.window.model.window_id: {
        "center_world_m": wall.window.center_world_m.tolist(),
        "right_world": wall.window.right_world.tolist(),
        "up_world": wall.window.up_world.tolist(),
        "opening_width_m": wall.window.model.width_m,
        "opening_height_m": wall.window.model.height_m,
        "frame_thickness_m": wall.window.frame_thickness_m}
        for wall in world.walls}
    rows = []
    for capture, obs, score in zip(captures, observations, truth):
        xyz, side = window_face_corners(windows[obs["target_id"]],
                                        capture["camera_center_world_m"])
        uv, depth = project_blender_camera(xyz,
            capture["camera_center_world_m"],
            capture["camera_rotation_blender_to_world"], camera)
        in_frame = bool(np.all(depth > .05) and
            np.all((uv[:, 0] >= 0) & (uv[:, 0] < camera.width_px)) and
            np.all((uv[:, 1] >= 0) & (uv[:, 1] < camera.height_px)))
        detected = obs["detected_corners_uv"] is not None
        corner_error = (float(np.mean(np.linalg.norm(np.asarray(obs["detected_corners_uv"])-uv,
                                                   axis=1))) if detected else None)
        correct = bool(obs["window_valid"] and in_frame and
                       corner_error is not None and corner_error <= 12.0)
        rows.append({"image_id": capture["image_id"], "time_s": obs["time_s"],
            "target_id": obs["target_id"], "geometric_in_frame": in_frame,
            "geometric_side": side, "detector_output": detected,
            "operational_pose_valid": obs["window_valid"],
            "correct_pose_association": correct,
            "false_operational_pose": bool(obs["window_valid"] and not correct),
            "missed_in_frame_pose": bool(in_frame and not obs["window_valid"]),
            "corner_error_px": corner_error,
            "pose_translation_error_m": score["translation_error_m"],
            "capture_to_guidance_age_s": obs["guidance_update_time_s"]-
                                          capture["capture_time_s"]})
    valid_errors = [r["pose_translation_error_m"] for r in rows
                    if r["correct_pose_association"] and
                       r["pose_translation_error_m"] is not None]
    corner_errors = [r["corner_error_px"] for r in rows if r["correct_pose_association"]]
    times = [item["capture_time_s"] for item in captures]
    result = {"schema": "m13_offline_rgb_evaluation_v1",
        "total_observations": len(rows),
        "geometrically_in_frame": sum(r["geometric_in_frame"] for r in rows),
        "correct_pose_associations": len(corner_errors),
        "false_operational_poses": sum(r["false_operational_pose"] for r in rows),
        "missed_in_frame_poses": sum(r["missed_in_frame_pose"] for r in rows),
        "valid_operational_poses": sum(r["operational_pose_valid"] for r in rows),
        "mean_corner_error_px_on_correct": float(np.mean(corner_errors)) if corner_errors else None,
        "mean_translation_error_m_on_correct": float(np.mean(valid_errors)) if valid_errors else None,
        "pose_error_denominator": len(valid_errors),
        "maximum_image_age_s": max(r["capture_to_guidance_age_s"] for r in rows),
        "mean_render_wall_seconds": float(np.mean([c["render_wall_seconds"] for c in captures])),
        "simulated_observation_rate_hz": (len(times)-1)/(times[-1]-times[0])
            if len(times) > 1 else None,
        "geometric_in_frame_is_not_visibility_or_occlusion_truth": True}
    (root / "offline_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "offline_evaluation_per_image.json").write_text(json.dumps(rows, indent=2),
                                                           encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(evaluate(Path(args.directory)), indent=2))


if __name__ == "__main__":
    main()
