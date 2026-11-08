"""Development runner for a new, single-plant Blender-image-guided mission.

The scene plan supplies appearance and static geometry only. Operational
guidance receives RGB, time, calibration, known target dimensions, and ideal
vehicle state; it never reads the source M11 flight or scene-plan jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import tomllib

import numpy as np
from PIL import Image

from aerial_autonomy.evaluation.mission_m11 import (
    CompleteMission, MissionCondition, load_m11_config, save_trial,
)
from aerial_autonomy.planning.geometry import AABB, segment_aabb_distance
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.realism.bridge import camera_blender_pose
from aerial_autonomy.realism.detector import (
    AdaptiveColorWindowDetector, AdaptivePixelArmDetector,
)
from aerial_autonomy.simulation.dynamics import advance_state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_only_plan(source_path: Path, output_path: Path) -> dict:
    """Copy the accepted facility description without M11 paths or image jobs."""
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    settings = {key: source["settings"][key] for key in ("render", "appearance")}
    decor = [{key: item[key] for key in ("name", "kind", "minimum_m", "maximum_m")}
             for item in source["decor"]]
    plan = {"schema": "m13_static_scene_v1", "settings": settings,
            "scene": source["scene"], "decor": decor}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


class BlenderCameraProvider:
    """Synchronous capture of the current state, with a durable RGB provenance log."""

    def __init__(self, blender: Path, scene: Path, script: Path, plan_path: Path,
                 output_dir: Path, camera, arm, max_images: int):
        self.blender, self.scene, self.script = map(Path, (blender, scene, script))
        self.plan_path, self.output_dir = Path(plan_path), Path(output_dir)
        self.camera, self.arm, self.max_images = camera, arm, max_images
        self.scene_sha256 = file_sha256(self.scene)
        self.plan_sha256 = file_sha256(self.plan_path)
        self.count = 0
        self.last_time_s = -math.inf
        self.last_capture = None
        (self.output_dir / "jobs").mkdir(parents=True, exist_ok=True)
        if max_images < 1:
            raise ValueError("max_images must be positive")

    def render(self, state, time_s, settings, rng):
        if not np.isfinite(time_s) or time_s < self.last_time_s:
            raise ValueError("camera capture time must be monotonic")
        if self.count >= self.max_images:
            raise RuntimeError("M13 camera image budget exhausted")
        image_id = f"rgb_{self.count:05d}"
        center, rotation = camera_blender_pose(state, self.camera)
        job = {"job_id": image_id, "group": "causal", "time_s": float(time_s),
               "position_world_m": state.position_m.tolist(),
               "quaternion_bw": state.quaternion_bw.tolist(),
               "camera_center_world_m": center.tolist(),
               "camera_rotation_blender_to_world": rotation.tolist(),
               "target_id": None, "profile": "bright", "material": "metal",
               "occlusion": False, "split": "m13_new_flight",
               "arm_initial_angle_rad": self.arm.initial_angle_rad,
               "arm_angular_velocity_rad_s": self.arm.angular_velocity_rad_s}
        job_path = self.output_dir / "jobs" / f"{image_id}.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        image_path = self.output_dir / "rendered" / "causal" / f"{image_id}.png"
        command = [str(self.blender), "-b", str(self.scene), "--python",
                   str(self.script), "--", "--plan", str(self.plan_path),
                   "--output-dir", str(self.output_dir), "--group", "causal",
                   "--job-file", str(job_path), "--device", "CPU"]
        start = time.perf_counter()
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=120, check=False)
        duration = time.perf_counter() - start
        ok = process.returncode == 0 and image_path.is_file()
        item = {"image_id": image_id, "capture_time_s": float(time_s),
                "vehicle_position_m": job["position_world_m"],
                "vehicle_quaternion_bw": job["quaternion_bw"],
                "camera_center_world_m": job["camera_center_world_m"],
                "camera_rotation_blender_to_world": job["camera_rotation_blender_to_world"],
                "arm_initial_angle_rad": self.arm.initial_angle_rad,
                "arm_angular_velocity_rad_s": self.arm.angular_velocity_rad_s,
                "scene_path": self.scene.as_posix(), "scene_sha256": self.scene_sha256,
                "plan_path": self.plan_path.as_posix(), "plan_sha256": self.plan_sha256,
                "render_ok": ok, "render_wall_seconds": duration,
                "image_path": image_path.as_posix(),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest() if ok else None}
        with (self.output_dir / "captures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        self.count += 1
        self.last_time_s = float(time_s)
        self.last_capture = item
        if not ok:
            (self.output_dir / "last_render_error.log").write_text(
                process.stdout[-12000:], encoding="utf-8")
            raise RuntimeError(f"Blender render failed for {image_id}")
        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


class ComplementaryWindowDetector:
    """Declared pixel-only rule: M9 RGB fit first, M12 adaptation on failure."""

    def __init__(self, colors):
        self.colors_by_id = dict(colors)
        self._m9 = {key: ColorWindowDetector({key: color})
                    for key, color in colors.items()}
        self._m12 = AdaptiveColorWindowDetector(colors)

    def detect(self, image_rgb, camera, target_id, expected_side="positive_normal"):
        primary = self._m9[target_id].detect(image_rgb, camera, target_id,
                                             expected_side)
        if primary.valid:
            return primary
        return self._m12.detect(image_rgb, camera, target_id, expected_side)


class BlenderMission(CompleteMission):
    """Reuse accepted dynamics and mission order with explicit RGB dependencies."""

    def __init__(self, *args, camera_provider: BlenderCameraProvider,
                 decor: list[dict], static_image_period_s: float = .8, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = camera_provider
        self.arm_detector = AdaptivePixelArmDetector()
        self.pixel_detector = ComplementaryWindowDetector(self.colors)
        self.decor = [(item["name"], AABB(tuple(item["minimum_m"]),
                    tuple(item["maximum_m"]))) for item in decor]
        self.minimum_decor_clearance_m = math.inf
        if static_image_period_s <= 0:
            raise ValueError("static image period must be positive")
        self.static_image_period_s = static_image_period_s

    def _guidance(self, target_id, reverse=False, delta=False):
        guidance = super()._guidance(target_id, reverse, delta)
        guidance.detector = self.pixel_detector
        return guidance

    def _observe(self, guidance, target_id, force=False):
        before = len(self.observations)
        result = super()._observe(guidance, target_id, force)
        if len(self.observations) > before:
            capture = self.renderer.last_capture
            record = self.observations[-1]
            if abs(record["time_s"] - capture["capture_time_s"]) > 1e-9:
                raise RuntimeError("image and guidance timestamps differ")
            record["image_id"] = capture["image_id"]
            record["image_sha256"] = capture["sha256"]
            record["capture_time_s"] = capture["capture_time_s"]
            record["guidance_update_time_s"] = self.time_s
            if target_id != "delta":
                self.next_image_s = self.time_s + self.static_image_period_s
        return result

    def _step(self, reference):
        # Evaluate new trajectory segments against furniture, not the saved M11 path.
        dt = self.raw["guidance"]["step_s"]
        control = self.controller.command(self.state, reference.position_m,
            self.yaw_rad, reference.velocity_m_s, reference.acceleration_m_s2)
        future = advance_state(self.state, control.wrench, self.vehicle.vehicle, dt)
        clearance = min((segment_aabb_distance(self.state.position_m,
            future.position_m, box) - self.world.envelope_m for _, box in self.decor),
            default=math.inf)
        self.minimum_decor_clearance_m = min(self.minimum_decor_clearance_m, clearance)
        if clearance <= 0:
            return self.abort("rendered_furniture_collision")
        ok = super()._step(reference)
        if ok:
            self.records[-1]["decor_clearance_m"] = clearance
        return ok

    def _fly(self, trajectory, phase, target_id=None, guidance=None):
        if not super()._fly(trajectory, phase, target_id, guidance):
            return False
        if phase == "APPROACH_DELTA":
            fix = self.last_fix
            # A prior static-window pose may point the camera for reacquisition;
            # it never authorizes a timed crossing. The M11 waiting loop requires
            # fresh RGB window and arm estimates before any commitment.
            if (fix is None or not fix.valid or fix.target_id != "delta" or
                    not self.observations or
                    self.time_s - max((item["time_s"] for item in self.observations
                        if item["target_id"] == "delta" and item["window_valid"]),
                        default=-math.inf) > 10.0):
                return self.abort("stale_delta_center_for_camera_turn")
            relative = fix.estimated_window.center_world_m-self.state.position_m
            bearing = math.atan2(relative[1], relative[0])
            return self._yaw_to(bearing, target_id)
        return True


def run_development(blender: Path, max_images: int | None, output_dir: Path,
                    static_image_period_s: float | None = None,
                    config_path: Path = Path("configs/mission_m13.toml"),
                    seed: int | None = None, phase_rad: float | None = None,
                    dropout_probability: float = 0.0,
                    forced_visual_loss: bool = False,
                    stationary_arm: bool = False,
                    max_wait_s: float | None = None,
                    trial_kind: str = "development"):
    output_dir = Path(output_dir)
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    source = config["source"]
    camera_config = config["camera"]
    development = config["development"]
    max_images = camera_config["max_images"] if max_images is None else max_images
    static_image_period_s = (camera_config["static_image_period_s"]
        if static_image_period_s is None else static_image_period_s)
    source_plan = Path(source["static_scene_plan"])
    plan_path = output_dir / "static_scene_plan.json"
    plan = scene_only_plan(source_plan, plan_path)
    raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
        source["mission_config"])
    raw["guidance"]["delta_approach_distance_m"] = config["guidance"][
        "delta_approach_distance_m"]
    if max_wait_s is not None:
        raw["guidance"]["max_wait_s"] = max_wait_s
    condition = MissionCondition(development["seed"] if seed is None else seed,
        development["arm_initial_phase_rad"] if phase_rad is None else phase_rad,
        noise_std_rgb=development["noise_std_rgb"],
        dropout_probability=dropout_probability,
        forced_visual_loss=forced_visual_loss, stationary_arm=stationary_arm)
    # Build the plant at rest at home. No M11 trace or trajectory is read.
    provider = BlenderCameraProvider(blender,
        Path(source["blender_scene"]), Path(source["blender_script"]), plan_path, output_dir,
        camera, arm, max_images)
    mission = BlenderMission(raw, vehicle, limits, camera, world, arm, colors,
                             condition, camera_provider=provider,
                             decor=plan["decor"],
                             static_image_period_s=static_image_period_s)
    provider.arm = mission.arm
    failure = None
    start = time.perf_counter()
    try:
        metrics = mission.run()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        mission.abort("render_or_budget_exception")
        metrics = mission.metrics()
    save_trial(mission, output_dir / "trial")
    result = {"schema": "m13_causal_trial_v1", "trial_kind": trial_kind,
              "configuration": {"path": str(config_path),
                  "sha256": file_sha256(config_path),
                  "static_image_period_s": static_image_period_s,
                  "max_images": max_images, "max_wait_s": raw["guidance"]["max_wait_s"]},
              "metrics": metrics,
              "detector_policy": "m9_then_m12",
              "exception": failure, "images": provider.count,
              "minimum_decor_clearance_m": mission.minimum_decor_clearance_m,
              "wall_seconds": time.perf_counter() - start,
              "complete_mission": metrics["success"]}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mission_m13.toml")
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--static-image-period-s", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase-rad", type=float)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--forced-visual-loss", action="store_true")
    parser.add_argument("--stationary-arm", action="store_true")
    parser.add_argument("--max-wait-s", type=float)
    parser.add_argument("--trial-kind", choices=("development", "final", "negative"),
                        default="development")
    parser.add_argument("--output-dir", default="outputs/mission_m13/development")
    args = parser.parse_args()
    print(json.dumps(run_development(Path(args.blender), args.max_images,
                                     Path(args.output_dir),
                                     args.static_image_period_s,
                                     Path(args.config), args.seed, args.phase_rad,
                                     args.dropout_probability,
                                     args.forced_visual_loss,
                                     args.stationary_arm, args.max_wait_s,
                                     args.trial_kind), indent=2))


if __name__ == "__main__":
    main()
