"""Source-only checks for the new causal camera boundary."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.perception.window_m10_arm import RotatingArm
from aerial_autonomy.realism.mission_m13 import (
    BlenderCameraProvider, BlenderMission, scene_only_plan,
)
from aerial_autonomy.evaluation.mission_m11 import MissionCondition
from aerial_autonomy.perception.window_m9_render import RenderSettings, WindowImageRenderer
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.trajectory.quintic import Reference


class TestM13CameraBoundary(unittest.TestCase):
    def test_scene_plan_has_no_historical_flight_or_image_jobs(self):
        source = {"settings": {"source": {"trial_dir": "forbidden"},
                               "render": {"samples": 16},
                               "appearance": {"seed": 1211}},
                  "scene": {"arm": {"initial_angle_rad": .9}},
                  "decor": [{"name": "bench", "kind": "bench",
                             "minimum_m": [0, 0, 0], "maximum_m": [1, 1, 1],
                             "minimum_extra_clearance_m": 1.1}],
                  "jobs": [{"trace_index": 123}],
                  "source_hashes_sha256": {"M11": "historical"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, output_path = root / "source.json", root / "plan.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            plan = scene_only_plan(source_path, output_path)
            self.assertEqual(plan, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertEqual(set(plan), {"schema", "settings", "scene", "decor"})
            self.assertNotIn("source", plan["settings"])
            self.assertNotIn("minimum_extra_clearance_m", plan["decor"][0])

    def test_capture_uses_current_state_time_and_trial_arm(self):
        _, _, _, camera, _, arm, _, _ = load_m11_config("configs/mission_m11.toml")
        arm = RotatingArm(arm.window, arm.shape, 1.7, -.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.blend").write_bytes(b"fixed-scene")
            (root / "plan.json").write_text("{}", encoding="utf-8")
            provider = BlenderCameraProvider(root / "blender", root / "scene.blend",
                root / "render.py", root / "plan.json", root, camera, arm, 1)
            state = VehicleState.at_rest()
            state.position_m = np.array([2.0, .3, 1.6])

            def fake_render(_command, **_kwargs):
                job = json.loads((root / "jobs" / "rgb_00000.json").read_text())
                self.assertEqual(job["position_world_m"], [2.0, .3, 1.6])
                self.assertEqual(job["time_s"], 4.5)
                self.assertEqual(job["arm_initial_angle_rad"], 1.7)
                self.assertEqual(job["arm_angular_velocity_rad_s"], -.25)
                self.assertIsNone(job["target_id"])
                path = root / "rendered" / "causal" / "rgb_00000.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(path)
                class Result:
                    returncode = 0
                    stdout = ""
                return Result()

            with patch("aerial_autonomy.realism.mission_m13.subprocess.run", fake_render):
                image = provider.render(state, 4.5, None, None)
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertEqual(provider.last_capture["capture_time_s"], 4.5)
            self.assertTrue(provider.last_capture["sha256"])
            self.assertTrue(provider.last_capture["scene_sha256"])
            self.assertTrue(provider.last_capture["plan_sha256"])
            with self.assertRaises(ValueError):
                provider.render(state, 4.4, None, None)
            with self.assertRaisesRegex(RuntimeError, "budget"):
                provider.render(state, 4.5, None, None)

    def test_new_flight_checks_rendered_furniture_at_each_plant_step(self):
        raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
            "configs/mission_m11.toml")
        mission = BlenderMission(raw, vehicle, limits, camera, world, arm,
            colors, MissionCondition(101, .9), camera_provider=None,
            decor=[{"name": "fixture", "minimum_m": [-.1, -.1, -.1],
                    "maximum_m": [.1, .1, .1]}])
        initial = mission.state.position_m.copy()
        self.assertFalse(mission._step(Reference(initial, np.zeros(3), np.zeros(3))))
        self.assertEqual(mission.failure, "rendered_furniture_collision")
        np.testing.assert_array_equal(mission.state.position_m, initial)

    def test_rgb_removal_changes_static_guidance_to_safe_failure(self):
        raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
            "configs/mission_m11.toml")
        mission = BlenderMission(raw, vehicle, limits, camera, world, arm,
            colors, MissionCondition(101, .9), camera_provider=None, decor=[])
        state = VehicleState.at_rest()
        state.position_m = np.array([0., 0., 1.6])
        renderer = WindowImageRenderer(camera, world, colors)
        rgb = renderer.render(state, RenderSettings(0.0, 1.0, False),
                              np.random.default_rng(101))
        guidance = mission._guidance("alpha")
        good = guidance.perceive(rgb, state, "alpha", 5.0)
        lost = guidance.perceive(np.zeros_like(rgb), state, "alpha", 5.0)
        self.assertTrue(good.valid)
        self.assertFalse(lost.valid)
        self.assertLess(np.linalg.norm(good.estimated_window.center_world_m -
                      world.wall_for_id("alpha").window.center_world_m), .2)


if __name__ == "__main__":
    unittest.main()
