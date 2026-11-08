"""Numerical ENU/FLU/RDF to Blender camera and scene bridge for M12."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import quaternion_to_matrix


# Blender cameras look down local -Z with local +Y up. M11 optical RDF has
# +Z forward and +Y down. The matrix is proper (determinant +1).
OPTICAL_RDF_FROM_BLENDER_CAMERA = np.diag([1., -1., -1.])


def camera_blender_pose(state: VehicleState, camera):
    """Return world position and local-Blender-camera-to-ENU rotation."""
    rotation_optical_to_world, center = camera.camera_to_world(state)
    return center, rotation_optical_to_world @ OPTICAL_RDF_FROM_BLENDER_CAMERA


def project_blender_camera(points_world_m, center_world_m, rotation_blender_to_world,
                           camera):
    """Blender camera geometry evaluated independently of camera.project()."""
    points = np.asarray(points_world_m, dtype=float)
    local = (points - np.asarray(center_world_m)) @ np.asarray(rotation_blender_to_world)
    depth = -local[:, 2]
    uv = np.column_stack((camera.fx_px * local[:, 0] / depth + camera.cx_px,
                          camera.fy_px * (-local[:, 1]) / depth + camera.cy_px))
    return uv, depth


def window_face_corners(window: dict, camera_center_world_m):
    """Physical inner trim corners on the visible face, ordered for M8/M9."""
    center = np.asarray(window["center_world_m"], dtype=float)
    right = np.asarray(window["right_world"], dtype=float)
    up = np.asarray(window["up_world"], dtype=float)
    normal = np.cross(right, up)
    side = 1 if np.dot(camera_center_world_m - center, normal) >= 0 else -1
    w, h = window["opening_width_m"] / 2, window["opening_height_m"] / 2
    local = np.asarray([[-w, -h], [w, -h], [w, h], [-w, h]])
    corners = (center + side * window["frame_thickness_m"] / 2 * normal
               + local[:, :1] * right + local[:, 1:] * up)
    return corners, ("positive_normal" if side > 0 else "negative_normal")


def state_from_trace(trace, index: int):
    return VehicleState(np.asarray(trace["position_m"][index]),
                        np.asarray(trace["velocity_m_s"][index]),
                        np.asarray(trace["quaternion_bw"][index]),
                        np.asarray(trace["body_rate_rad_s"][index]))


def nearest_index(times: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(times, value))
    return min((max(index - 1, 0), min(index, len(times) - 1)),
               key=lambda i: abs(float(times[i]) - value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(plan_path: str | Path, stored_path: str) -> Path:
    """Resolve a serialized project-relative path after ZIP extraction.

    The plan is stored at <project>/outputs/realism_m12/scene_plan.json.
    Older Windows plan keys are accepted while newly written keys use POSIX
    separators on every host.
    """
    relative = Path(stored_path.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('plan source path must be project-relative')
    return Path(plan_path).resolve().parents[2] / relative


def _point_box_clearance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    delta = np.maximum(np.maximum(lo - points, points - hi), 0.)
    return float(np.min(np.linalg.norm(delta, axis=1)))


def decor_spec(trace):
    """Furniture AABBs outside every saved M11 0.25 m collision envelope."""
    objects = []
    for index, x in enumerate((2.35, 7.45, 12.25, 17.85, 23.50)):
        objects.append({"name": f"equipment_bench_{index + 1}",
                        "kind": "bench", "minimum_m": [x - .62, -3.05, -.24],
                        "maximum_m": [x + .62, -2.25, 1.10]})
        objects.append({"name": f"storage_cabinet_{index + 1}",
                        "kind": "cabinet", "minimum_m": [x + .90, -5.15, -.24],
                        "maximum_m": [x + 1.50, -4.55, 1.72]})
    points = np.asarray(trace["position_m"], dtype=float)
    # A continuous-segment allowance bounds the missed point between 0.01 s
    # samples by half the largest adjacent displacement.
    intersample = float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) / 2)
    for item in objects:
        d = _point_box_clearance(points, np.asarray(item["minimum_m"]),
                                 np.asarray(item["maximum_m"]))
        item["minimum_extra_clearance_m"] = d - .25 - intersample
        if item["minimum_extra_clearance_m"] <= .10:
            raise ValueError(f"decorative object intersects M11 envelope: {item['name']}")
    return objects


def _job(job_id, group, time_s, index, trace, camera, windows, target_id=None,
         profile="bright", material="metal", split="presentation",
         source_camera_frame=None, yaw_offset_rad=0., occlusion=False):
    state = state_from_trace(trace, index)
    center, rotation = camera_blender_pose(state, camera)
    target = next((w for w in windows if w["window_id"] == target_id), None)
    corners = None
    corner_world = None
    side = None
    if target is not None:
        xyz, side = window_face_corners(target, center)
        corner_world = xyz.tolist()
        pixels, depth = project_blender_camera(xyz, center, rotation, camera)
        corners = pixels.tolist() if np.all(depth > .05) else None
    return {"job_id": job_id, "group": group, "time_s": float(trace["time_s"][index]),
            "trace_index": int(index), "position_world_m": state.position_m.tolist(),
            "quaternion_bw": state.quaternion_bw.tolist(),
            "camera_center_world_m": center.tolist(),
            "camera_rotation_blender_to_world": rotation.tolist(),
            "target_id": target_id, "expected_side": side,
            "true_corners_uv": corners, "true_corners_world_m": corner_world,
            "profile": profile, "material": material,
            "split": split, "source_camera_frame": source_camera_frame,
            "yaw_offset_rad": yaw_offset_rad, "occlusion": occlusion,
            "phase": str(trace["phase"][index]),
            "thrust_n": float(trace["thrust_n"][index]),
            "torque_nm": trace["torque_nm"][index].tolist(),
            "static_clearance_m": float(trace["static_clearance_m"][index]),
            "dynamic_clearance_m": float(trace["dynamic_clearance_m"][index])}


def build_plan(config_path: str | Path, output_dir: str | Path):
    config_path, output_dir = Path(config_path), Path(output_dir)
    with config_path.open("rb") as stream:
        settings = tomllib.load(stream)
    trial = Path(settings["source"]["trial_dir"])
    scene_path = trial / "scene.json"
    trace_path = trial / "physical_trace.npz"
    camera_frames_path = trial / "camera_frames.npz"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    trace = np.load(trace_path)
    frames = np.load(camera_frames_path)
    _, _, _, camera, _, _, colors, _ = load_m11_config(
        settings["source"]["mission_config"])
    if (camera.width_px != settings["render"]["camera_width_px"] or
            camera.height_px != settings["render"]["camera_height_px"]):
        raise ValueError("M12 must preserve M11 camera resolution")
    windows = scene["windows"]
    times = np.asarray(trace["time_s"])
    jobs = []
    for target in ("alpha", "beta", "gamma", "delta"):
        observed = [i for i, name in enumerate(frames["target_id"])
                    if str(name) == target]
        groups = {}
        for i in observed:
            index = nearest_index(times, float(frames["time_s"][i]))
            state = state_from_trace(trace, index)
            center, _ = camera_blender_pose(state, camera)
            window = next(w for w in windows if w["window_id"] == target)
            _, side = window_face_corners(window, center)
            groups.setdefault(side, []).append((i, index))
        for side, entries in groups.items():
            count = (settings["dataset"]["arm_observations"] if target == "delta"
                     else settings["dataset"]["static_distances_per_side"])
            picks = np.unique(np.linspace(0, len(entries) - 1,
                                           min(count, len(entries))).round().astype(int))
            for rank, pick in enumerate(picks):
                frame_id, index = entries[int(pick)]
                profile = ("bright", "dim", "shadow")[rank % 3]
                material = "paint" if rank % 3 == 1 else "metal"
                split = "development" if rank % 3 == 0 else "held_out"
                jobs.append(_job(f"{target}_{side}_{rank:02d}", "evaluation",
                    float(times[index]), index, trace, camera, windows, target,
                    profile, material, split, int(frame_id)))
            if target != "delta" and settings["dataset"]["include_partial_views"]:
                frame_id, index = entries[-1]
                jobs.append(_job(f"{target}_{side}_partial", "evaluation",
                    float(times[index]), index, trace, camera, windows, target,
                    "shadow", "metal", "held_out", int(frame_id), occlusion=True))
    # Full mission frames come from the accepted visualization timeline.
    timeline_path = trial / "figures" / "animation_timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    mission_times = timeline["mission_frame_times_s"]
    for rank, time_s in enumerate(mission_times):
        index = nearest_index(times, time_s)
        jobs.append(_job(f"mission_{rank:03d}", "mission", time_s, index,
                         trace, camera, windows))
        if rank % 2 == 0:
            target = str(trace['target_id'][index]) or None
            jobs.append(_job(f"sync_{rank:03d}", "sync", time_s, index,
                             trace, camera, windows, target))
    camera_indices = np.unique(np.linspace(0, len(frames["time_s"]) - 1,
            settings["presentation"]["onboard_frames"]).round().astype(int))
    for rank, frame_id in enumerate(camera_indices):
        index = nearest_index(times, float(frames["time_s"][frame_id]))
        jobs.append(_job(f"onboard_{rank:03d}", "onboard", float(times[index]),
            index, trace, camera, windows, str(frames["target_id"][frame_id]),
            source_camera_frame=int(frame_id)))
    arm_times = np.linspace(64., 72., settings["presentation"]["arm_frames"])
    for rank, time_s in enumerate(arm_times):
        index = nearest_index(times, time_s)
        jobs.append(_job(f"arm_{rank:03d}", "arm", time_s, index,
                         trace, camera, windows, "delta"))
    plan = {"schema": "realism_m12_v1", "source_m11_head":
                settings["source"]["expected_m11_head"],
            "source_hashes_sha256": {path.as_posix(): _sha256(path) for path in
                (scene_path, trace_path, camera_frames_path,
                 Path(settings["source"]["mission_config"]))},
            "settings": settings, "scene": scene,
            "color_rgb_by_window": {k: list(v) for k, v in colors.items()},
            "decor": decor_spec(trace), "jobs": jobs}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scene_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path, plan


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/realism_m12.toml")
    parser.add_argument("--output-dir", default="outputs/realism_m12")
    args = parser.parse_args()
    path, plan = build_plan(args.config, args.output_dir)
    print(json.dumps({"path": str(path), "jobs": len(plan["jobs"]),
                      "groups": {group: sum(j["group"] == group for j in plan["jobs"])
                                 for group in sorted({j["group"] for j in plan["jobs"]})}},
                     indent=2))


if __name__ == "__main__":
    main()
