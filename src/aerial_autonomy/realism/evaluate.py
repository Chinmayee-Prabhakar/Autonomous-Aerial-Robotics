"""Paired M11-native versus M12-Blender RGB perception study.

Detector inputs are RGB bytes, target identity, known dimensions, and camera
calibration. Saved M11/Blender truth is read only for independent scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.perception.mission_m11_detect import TargetColorDetector
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m9_guidance import ImagePoseFix
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.window_observation import WindowObservation
from aerial_autonomy.perception.window_pose import estimate_window_pose
from aerial_autonomy.perception.window_m10_timing import AngularTracker, wrap_angle
from aerial_autonomy.perception.window_m10_vision import ArmDetection, PixelArmDetector
from aerial_autonomy.realism.detector import (AdaptiveColorWindowDetector,
                                               AdaptivePixelArmDetector)
from aerial_autonomy.simulation.rotation import quaternion_to_matrix
from aerial_autonomy.realism.bridge import (camera_blender_pose, nearest_index,
    project_blender_camera, project_path, state_from_trace, window_face_corners)


DETECTOR_LABELS = ('m11_operational', 'm9_comparator', 'm12_adapted')


def _truth_geometry(job, scene, camera):
    target = next(w for w in scene["windows"] if w["window_id"] == job["target_id"])
    corners = np.asarray(job["true_corners_uv"], dtype=float) if job["true_corners_uv"] else None
    usable = (corners is not None and np.all(np.isfinite(corners))
              and np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < camera.width_px)
              and np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < camera.height_px))
    center = np.asarray(target["center_world_m"], dtype=float)
    right = np.asarray(target["right_world"], dtype=float)
    up = np.asarray(target["up_world"], dtype=float)
    normal = np.cross(right, up)
    sign = 1 if job["expected_side"] == "positive_normal" else -1
    visible_center = center + sign * target["frame_thickness_m"] / 2 * normal
    q = np.asarray(job["quaternion_bw"])
    body_rotation = quaternion_to_matrix(q)
    camera_rotation = body_rotation @ camera.rotation_camera_to_body
    camera_center = np.asarray(job["camera_center_world_m"])
    true_translation = camera_rotation.T @ (visible_center - camera_center)
    true_rotation = camera_rotation.T @ np.column_stack((right, up, normal))
    return target, corners, usable, true_translation, true_rotation


def _window_detectors(target, colors, adapted):
    """Keep M11's operational color isolation distinct from the M9 fitter."""
    return {'m11_operational': TargetColorDetector({target: colors[target]}),
            'm9_comparator': ColorWindowDetector({target: colors[target]}),
            'm12_adapted': adapted}


def _native_truth_geometry(job, target, camera, trace, frames):
    """Project the saved physical window from the native image's own time/pose.

    camera_frames.corners_uv contains operational detections and is never read.
    Four-corner in-frame eligibility is geometric, not an occlusion judgment.
    """
    frame_id = job['source_camera_frame']
    frame_time = float(frames['time_s'][frame_id])
    if (abs(frame_time - job['time_s']) > 1e-8 or
            str(frames['target_id'][frame_id]) != job['target_id']):
        raise AssertionError('paired native image time/target mismatch: ' + job['job_id'])
    index = nearest_index(np.asarray(trace['time_s']), frame_time)
    if (abs(float(trace['time_s'][index]) - frame_time) > 1e-8 or
            index != job['trace_index']):
        raise AssertionError('native image/plant timestamp mismatch: ' + job['job_id'])
    state = state_from_trace(trace, index)
    center, rotation = camera_blender_pose(state, camera)
    corners_world, side = window_face_corners(target, center)
    if side != job['expected_side']:
        raise AssertionError('native physical face mismatch: ' + job['job_id'])
    corners, depth = project_blender_camera(corners_world, center, rotation, camera)
    if not np.all(depth > .05):
        corners = None
    usable = (corners is not None and np.all(np.isfinite(corners))
        and np.all((corners[:, 0] >= 0) & (corners[:, 0] < camera.width_px))
        and np.all((corners[:, 1] >= 0) & (corners[:, 1] < camera.height_px)))
    right = np.asarray(target['right_world'], dtype=float)
    up = np.asarray(target['up_world'], dtype=float)
    optical_to_world = rotation @ np.diag([1., -1., -1.])
    visible_center = np.mean(corners_world, axis=0)
    true_translation = optical_to_world.T @ (visible_center - center)
    true_rotation = optical_to_world.T @ np.column_stack((right, up, np.cross(right, up)))
    return (target, corners, bool(usable), true_translation, true_rotation), corners_world


def _pose_score(detection, job, camera, model, truth):
    target, corners, usable, true_translation, true_rotation = truth
    result = {"detection_valid": bool(detection.valid),
              "detection_reason": detection.reason,
              "color_pixel_count": int(detection.color_pixel_count),
              "confidence": float(detection.confidence),
              "line_rmse_px": detection.line_rmse_px,
              "corners_uv": detection.corners_uv.tolist() if detection.valid else None,
              "truth_usable": bool(usable), "false_detection": False,
              "truth_matched": False,
              "corner_mean_error_px": None, "corner_max_error_px": None,
              "pose_valid": False, "pose_reason": None,
              "translation_error_m": None, "orientation_error_deg": None,
              "reprojection_rmse_px": None}
    estimate = None
    if not detection.valid:
        return result, estimate
    if corners is None or not np.all(np.isfinite(corners)):
        result["false_detection"] = True
    else:
        error = np.linalg.norm(detection.corners_uv - corners, axis=1)
        result["corner_mean_error_px"] = float(np.mean(error))
        result["corner_max_error_px"] = float(np.max(error))
        result["false_detection"] = bool(not usable or np.mean(error) > 12)
    result["truth_matched"] = not result["false_detection"]
    observation = WindowObservation(job["target_id"], job["time_s"],
                                    detection.corners_uv, np.ones(4, dtype=bool), None)
    estimate = estimate_window_pose(observation, model, camera)
    result["pose_valid"] = bool(estimate.valid)
    result["pose_reason"] = estimate.reason
    if estimate.valid and result["truth_matched"]:
        result["translation_error_m"] = float(np.linalg.norm(
            estimate.translation_window_in_camera_m - true_translation))
        delta = estimate.rotation_window_to_camera @ true_rotation.T
        result["orientation_error_deg"] = float(np.degrees(np.arccos(np.clip(
            (np.trace(delta) - 1) / 2, -1, 1))))
        result["reprojection_rmse_px"] = float(estimate.reprojection_rmse_px)
    return result, estimate


def _arm_score(image, detection, estimate, job, model, adapter, arm_truth):
    output = {"valid": False, "reason": "window_pose_unavailable",
              "angle_rad": None, "angle_error_deg": None,
              "pivot_local_m": None, "tip_local_m": None,
              "shaft_pixel_count": 0, "confidence": 0.0}
    arm_detection = None
    if not detection.valid or estimate is None or not estimate.valid:
        return output, arm_detection
    # PixelArmDetector reads only this declared model's metric dimensions;
    # the placeholder pose intentionally contains no true window position.
    placeholder = StaticWindow(model, np.zeros(3), np.array([1., 0., 0.]),
                               np.array([0., 1., 0.]))
    fix = ImagePoseFix(job["target_id"], True, None, detection, estimate,
                       placeholder)
    arm_detection = adapter.detect(image, fix, job["time_s"])
    output.update({"valid": bool(arm_detection.valid),
                   "reason": arm_detection.reason,
                   "angle_rad": arm_detection.angle_rad,
                   "pivot_local_m": arm_detection.pivot_local_m,
                   "tip_local_m": arm_detection.tip_local_m,
                   "shaft_pixel_count": arm_detection.shaft_pixel_count,
                   "confidence": float(arm_detection.confidence)})
    if arm_detection.valid:
        output["angle_error_deg"] = float(np.degrees(abs(wrap_angle(
            arm_detection.angle_rad - arm_truth))))
    return output, arm_detection


def _overlay(path, image, entries, truth_corners):
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    for rank, (label, color) in enumerate((
            ('m11_operational', (255, 145, 40)),
            ('m9_comparator', (75, 185, 255)),
            ('m12_adapted', (25, 255, 95)))):
        corners = entries[label]["window"]["corners_uv"]
        if corners is not None:
            line = [tuple(x) for x in corners] + [tuple(corners[0])]
            draw.line(line, fill=color, width=2)
        arm = entries[label].get("arm")
        if arm and arm["valid"]:
            draw.text((5, 5 + rank * 13),
                      f"{label} arm {arm['angle_error_deg']:.1f} deg", fill=color)
    if truth_corners is not None:
        for x, y in truth_corners:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                             outline=(255, 255, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for algorithm, scores in row["scores"].items():
            for key in (("all",), (row["split"],),
                        (row["split"], row["target_id"]),
                        (row["split"], row["profile"]),
                        (row["split"], row["expected_side"]),
                        (row["split"], row["material"]),
                        (row["split"], "occluded" if row["occlusion"] else "clear")):
                groups[(algorithm, *key)].append(scores)
    def mean(values):
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None
    result = {}
    for key, cases in groups.items():
        name = "/".join(key)
        result[name] = {"count": len(cases),
            "ground_truth_usable": sum(x["window"]["truth_usable"] for x in cases),
            "detected": sum(x["window"]["detection_valid"] for x in cases),
            "pose_valid": sum(x["window"]["pose_valid"] for x in cases),
            "false_detections": sum(x["window"]["false_detection"] for x in cases),
            "correct_detections": sum(x["window"]["truth_matched"] for x in cases),
            "matched_pose_valid": sum(x["window"]["pose_valid"] and
                x["window"]["truth_matched"] for x in cases),
            "corner_mean_error_px": mean(x["window"]["corner_mean_error_px"]
                if x["window"]["truth_matched"] else None for x in cases),
            "translation_mean_error_m": mean(x["window"]["translation_error_m"]
                                             for x in cases),
            "orientation_mean_error_deg": mean(x["window"]["orientation_error_deg"]
                                               for x in cases),
            "reprojection_mean_rmse_px": mean(x["window"]["reprojection_rmse_px"]
                                             for x in cases),
            "arm_detected": sum(bool(x.get("arm", {}).get("valid")) for x in cases),
            "arm_mean_angle_error_deg": mean(x.get("arm", {}).get("angle_error_deg")
                                             for x in cases),
            "tracker_valid": sum(bool(x.get("tracker", {}).get("valid")) for x in cases),
            "tracker_rate_mean_error_rad_s": mean(
                x.get("tracker", {}).get("rate_error_rad_s") for x in cases),
            "tracker_prediction_mean_error_deg": mean(
                x.get("tracker", {}).get("prediction_error_deg") for x in cases)}
    return result


def _track_rows(rows, arm):
    """Causal arm tracking per domain, detector, and viewing side.

    Only prior pixel detections enter the tracker. Truth enters afterward for
    rate and 0.3-second prediction error scoring.
    """
    for domain in ('Blender RGB', 'M11 native RGB'):
        for label in DETECTOR_LABELS:
            for side in ('positive_normal', 'negative_normal'):
                tracker = AngularTracker()
                sequence = sorted((r for r in rows if r['domain'] == domain
                    and r['target_id'] == 'delta' and r['expected_side'] == side),
                    key=lambda r: r['time_s'])
                for row in sequence:
                    item = row['scores'][label]['arm']
                    detection = ArmDetection(bool(item['valid']), row['time_s'],
                        item['angle_rad'], item['pivot_local_m'], item['tip_local_m'],
                        item['confidence'], item['reason'], item['shaft_pixel_count'])
                    tracker.update(detection)
                    motion = tracker.estimate(row['time_s'])
                    record = {'valid': bool(motion.valid), 'reason': motion.reason,
                        'observation_count': motion.observation_count,
                        'misses_cumulative': tracker.misses,
                        'angular_velocity_rad_s': motion.angular_velocity_rad_s,
                        'rate_error_rad_s': None, 'prediction_error_deg': None,
                        'prediction_horizon_s': .3}
                    if motion.valid:
                        record['rate_error_rad_s'] = abs(
                            motion.angular_velocity_rad_s - arm.angular_velocity_rad_s)
                        predicted, _ = motion.predict(row['time_s'] + .3)
                        truth = (arm.initial_angle_rad + arm.angular_velocity_rad_s
                                 * (row['time_s'] + .3))
                        record['prediction_error_deg'] = math.degrees(
                            abs(wrap_angle(predicted - truth)))
                    row['scores'][label]['tracker'] = record


def evaluate(plan_path, output_dir, split="all"):
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = plan["settings"]
    _, _, _, camera, _, arm, colors, _ = load_m11_config(
        project_path(plan_path, settings["source"]["mission_config"]))
    trial = project_path(plan_path, settings["source"]["trial_dir"])
    original_frames = np.load(trial /
                              "camera_frames.npz")
    trace = np.load(trial /
                    "physical_trace.npz")
    adapted_window = AdaptiveColorWindowDetector(colors)
    algorithms = {"m11_operational": PixelArmDetector(),
                  "m9_comparator": PixelArmDetector(),
                  "m12_adapted": AdaptivePixelArmDetector()}
    rows = []
    truth_dir = output_dir / "annotations"
    detection_dir = output_dir / "detections"
    overlay_dir = output_dir / "overlays"
    for job in plan["jobs"]:
        if job["group"] != "evaluation" or (split != "all" and job["split"] != split):
            continue
        path = output_dir / "rendered" / "evaluation" / (job["job_id"] + ".png")
        if not path.is_file():
            raise FileNotFoundError(f"missing declared Blender frame: {path}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        target = job["target_id"]
        source = next(w for w in plan["scene"]["windows"] if w["window_id"] == target)
        model = WindowModel(target, source["opening_width_m"],
                            source["opening_height_m"])
        truth = _truth_geometry(job, plan["scene"], camera)
        arm_truth = (arm.initial_angle_rad + arm.angular_velocity_rad_s * job["time_s"])
        scores = {}
        for label, detector in _window_detectors(target, colors, adapted_window).items():
            detection = detector.detect(image, camera, target, job["expected_side"])
            window_score, estimate = _pose_score(detection, job, camera, model, truth)
            entry = {"window": window_score}
            if target == "delta":
                arm_score, _ = _arm_score(image, detection, estimate, job, model,
                                          algorithms[label], arm_truth)
                entry["arm"] = arm_score
            scores[label] = entry
        row = {"job_id": job["job_id"], "domain": "Blender RGB",
            "split": job["split"], "target_id": target,
            "expected_side": job["expected_side"], "profile": job["profile"],
            "material": job["material"], "occlusion": job["occlusion"],
            "time_s": job["time_s"], "source_camera_frame": job["source_camera_frame"],
            "scores": scores}
        rows.append(row)
        annotation = {"job_id": job["job_id"],
                      "true_corners_uv": job["true_corners_uv"],
                      "true_corners_world_m": job["true_corners_world_m"],
                      "true_arm_angle_rad": arm_truth if target == "delta" else None,
                      "truth_usable": bool(truth[2]),
                      "time_s": job["time_s"], "target_id": target}
        truth_dir.mkdir(parents=True, exist_ok=True)
        detection_dir.mkdir(parents=True, exist_ok=True)
        (truth_dir / (job["job_id"] + ".json")).write_text(
            json.dumps(annotation, indent=2), encoding="utf-8")
        (detection_dir / (job["job_id"] + ".json")).write_text(
            json.dumps(scores, indent=2), encoding="utf-8")
        _overlay(overlay_dir / (job["job_id"] + ".png"), image,
                 scores, truth[1])
        # Native truth comes from independent physical geometry and the saved
        # image-time plant state, never from saved operational detections.
        if job["source_camera_frame"] is not None:
            native = np.asarray(original_frames["image_rgb"][job["source_camera_frame"]])
            native_truth_tuple, native_world = _native_truth_geometry(
                job, source, camera, trace, original_frames)
            native_scores = {}
            for label, detector in _window_detectors(target, colors, adapted_window).items():
                detection = detector.detect(native, camera, target,
                                            job["expected_side"])
                window_score, estimate = _pose_score(detection, job, camera,
                                                      model, native_truth_tuple)
                entry = {"window": window_score}
                if target == "delta":
                    arm_score, _ = _arm_score(native, detection, estimate, job,
                        model, algorithms[label], arm_truth)
                    entry["arm"] = arm_score
                native_scores[label] = entry
            (truth_dir / (job["job_id"] + '_native.json')).write_text(
                json.dumps({'job_id': job['job_id'], 'source_camera_frame':
                    job['source_camera_frame'], 'time_s': job['time_s'],
                    'true_corners_uv': native_truth_tuple[1].tolist()
                        if native_truth_tuple[1] is not None else None,
                    'true_corners_world_m': native_world.tolist(),
                    'truth_usable': native_truth_tuple[2],
                    'geometry_source': 'physical window and image-time plant pose'},
                    indent=2), encoding='utf-8')
            rows.append({**row, "domain": "M11 native RGB", "scores": native_scores})
    _track_rows(rows, arm)
    for row in rows:
        group = 'blender' if row['domain'] == 'Blender RGB' else 'native'
        (output_dir / 'per_case').mkdir(parents=True, exist_ok=True)
        (output_dir / 'per_case' / f"{row['job_id']}_{group}.json").write_text(
            json.dumps(row, indent=2), encoding='utf-8')
    aggregate = {"schema": "m12_perception_v1", "total_cases": len(rows),
        "blender_cases": sum(r['domain'] == 'Blender RGB' for r in rows),
        "native_paired_cases": sum(r['domain'] == 'M11 native RGB' for r in rows),
        "split": split,
        "by_domain": {domain: _summary([r for r in rows if r['domain'] == domain])
                      for domain in ('Blender RGB', 'M11 native RGB')},
        "cases": rows}
    (output_dir / f'perception_aggregate_{split}.json').write_text(
        json.dumps(aggregate, indent=2), encoding='utf-8')
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='outputs/realism_m12/scene_plan.json')
    parser.add_argument('--output-dir', default='outputs/realism_m12')
    parser.add_argument('--split', choices=('all', 'development', 'held_out'),
                        default='all')
    args = parser.parse_args()
    result = evaluate(args.plan, args.output_dir, args.split)
    print(json.dumps({'blender_cases': result['blender_cases'],
        'native_cases': result['native_paired_cases'],
        'blender_all': {k: v for k, v in result['by_domain']['Blender RGB'].items()
                        if k.endswith('/all')}}, indent=2))


if __name__ == '__main__':
    main()
