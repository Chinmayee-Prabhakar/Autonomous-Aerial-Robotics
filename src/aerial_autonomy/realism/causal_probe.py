"""Limited causal Blender-RGB alpha crossing from an accepted M11 approach state.

Every observation is rendered from the *current newly propagated* Python plant
state. Valid pixel poses at acquisition/reacquisition set M11 trajectory entry
and exit; later frames are processed and logged without continuous replanning.
The M11 controller and collision checks then advance the same Python plant.
This is a separately labeled M12 local trial, not a replay of the successful
M11 crossing or a complete new mission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from aerial_autonomy.evaluation.mission_m11 import (
    CompleteMission, MissionCondition, load_m11_config, physical_crossings,
)
from aerial_autonomy.realism.bridge import camera_blender_pose, nearest_index, state_from_trace
from aerial_autonomy.realism.detector import AdaptiveColorWindowDetector
from aerial_autonomy.perception.window_m10_arm import arm_clearance_m


class BlenderOnDemandRenderer:
    def __init__(self, blender, plan_path, scene_path, script_path, output_dir,
                 camera, max_images):
        self.blender = str(blender)
        self.plan_path = Path(plan_path).resolve()
        self.scene_path = Path(scene_path).resolve()
        self.script_path = Path(script_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.camera = camera
        self.max_images = max_images
        self.count = 0
        self.wall_seconds = []
        (self.output_dir/'jobs').mkdir(parents=True, exist_ok=True)

    def render(self, state, time_s, settings, rng):
        if self.count >= self.max_images:
            raise RuntimeError('causal image limit reached')
        name = f'causal_{self.count:03d}'
        center, rotation = camera_blender_pose(state, self.camera)
        job = {'job_id': name, 'group': 'causal', 'time_s': float(time_s),
               'position_world_m': state.position_m.tolist(),
               'quaternion_bw': state.quaternion_bw.tolist(),
               'camera_center_world_m': center.tolist(),
               'camera_rotation_blender_to_world': rotation.tolist(),
               'target_id': 'alpha', 'expected_side': 'positive_normal',
               'profile': 'bright', 'material': 'metal',
               'occlusion': False, 'split': 'causal'}
        path = self.output_dir/'jobs'/(name+'.json')
        path.write_text(json.dumps(job, indent=2), encoding='utf-8')
        command = [self.blender, '-b', str(self.scene_path), '--python',
                   str(self.script_path), '--', '--plan', str(self.plan_path),
                   '--output-dir', str(self.output_dir), '--group', 'causal',
                   '--job-file', str(path), '--device', 'CPU']
        start = time.perf_counter()
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=90,
                                 check=False)
        elapsed = time.perf_counter()-start
        image = self.output_dir/'rendered'/'causal'/(name+'.png')
        if process.returncode != 0 or not image.is_file():
            (self.output_dir/'last_blender_error.log').write_text(
                process.stdout[-12000:], encoding='utf-8')
            raise RuntimeError(f'Blender causal frame failed: {name}, '
                               f'exit={process.returncode}')
        self.count += 1
        self.wall_seconds.append(elapsed)
        print(f'M12_CAUSAL_FRAME {name} t={time_s:.2f}s '
              f'x={state.position_m[0]:.3f} render_wall={elapsed:.2f}s',
              flush=True)
        return np.asarray(Image.open(image).convert('RGB'), dtype=np.uint8)


def run_probe(blender, max_images=60):
    root = Path('outputs/realism_m12')
    plan = json.loads((root/'scene_plan.json').read_text(encoding='utf-8'))
    trial = Path(plan['settings']['source']['trial_dir'])
    trace = np.load(trial/'physical_trace.npz')
    index = nearest_index(np.asarray(trace['time_s']), 11.5)
    raw, vehicle, limits, camera, world, arm, colors, _ = load_m11_config(
        plan['settings']['source']['mission_config'])
    mission = CompleteMission(raw, vehicle, limits, camera, world, arm, colors,
                              MissionCondition(101, arm.initial_angle_rad,
                                               noise_std_rgb=0.0))
    mission.state = state_from_trace(trace, index)
    mission.time_s = float(trace['time_s'][index])
    mission.yaw_rad = 0.0
    mission.next_image_s = mission.time_s
    mission.phase, mission.target_id = 'CAUSAL_ALPHA_INITIAL', 'alpha'
    mission.transitions = [{'time_s': mission.time_s,
                            'state': mission.phase, 'target_id': 'alpha'}]
    mission.records = [{'time_s': mission.time_s,
        'position_m': mission.state.position_m.copy(),
        'velocity_m_s': mission.state.velocity_m_s.copy(),
        'quaternion_bw': mission.state.quaternion_bw.copy(),
        'body_rate_rad_s': mission.state.body_rate_rad_s.copy(),
        'thrust_n': float(trace['thrust_n'][index]),
        'torque_nm': np.asarray(trace['torque_nm'][index]).copy(),
        'static_clearance_m': world.segment_clearance_m(
            mission.state.position_m, mission.state.position_m),
        'dynamic_clearance_m': arm_clearance_m(
            mission.state.position_m, arm.window, arm.shape,
            arm.angle_at(mission.time_s), world.envelope_m),
        'phase': mission.phase, 'target_id': 'alpha'}]
    renderer = BlenderOnDemandRenderer(blender, root/'scene_plan.json',
        root/'indoor_facility.blend', 'tools/m12/blender_render.py',
        root/'causal_alpha', camera, max_images)
    mission.renderer = renderer
    original_guidance = mission._guidance
    adaptive = AdaptiveColorWindowDetector(colors)
    def causal_guidance(*args, **kwargs):
        guidance = original_guidance(*args, **kwargs)
        guidance.detector = adaptive
        return guidance
    mission._guidance = causal_guidance
    start = time.perf_counter()
    error = None
    try:
        succeeded = bool(mission._static_window('alpha'))
    except Exception as exc:
        succeeded = False
        error = str(exc)
    elapsed = time.perf_counter()-start
    crossings = physical_crossings(mission.records, world)
    out = root/'causal_alpha'
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'm12_limited_blender_causal_alpha_v1',
              'scope': 'one alpha crossing initialized at saved M11 approach state',
              'accepted_m11_source_index': index,
              'initial_time_s': float(trace['time_s'][index]),
              'initial_position_world_m': trace['position_m'][index].tolist(),
              'rendered_from_new_plant_state': True,
              'image_source': 'Blender Cycles CPU RGB, 320x240',
              'guidance_detector': 'M12 adaptive pixel-only hue-ratio detector',
              'succeeded': succeeded, 'failure': mission.failure,
              'exception': error, 'blender_images': renderer.count,
              'valid_window_observations': sum(x['window_valid']
                                               for x in mission.observations),
              'observation_count': len(mission.observations),
              'crossings': crossings,
              'minimum_static_clearance_m': mission.min_static_clearance_m,
              'minimum_dynamic_clearance_m': mission.min_dynamic_clearance_m,
              'final_time_s': mission.time_s,
              'final_position_world_m': mission.state.position_m.tolist(),
              'wall_seconds': elapsed,
              'mean_blender_launch_and_render_seconds':
                  float(np.mean(renderer.wall_seconds)) if renderer.wall_seconds else None,
              'not_a_full_mission': True}
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/'observations.json').write_text(json.dumps(mission.observations, indent=2),
                                         encoding='utf-8')
    np.savez_compressed(out/'physical_trace.npz',
        time_s=np.array([r['time_s'] for r in mission.records]),
        position_m=np.array([r['position_m'] for r in mission.records]),
        quaternion_bw=np.array([r['quaternion_bw'] for r in mission.records]),
        thrust_n=np.array([r['thrust_n'] for r in mission.records]),
        torque_nm=np.array([r['torque_nm'] for r in mission.records]),
        static_clearance_m=np.array([r['static_clearance_m'] for r in mission.records]),
        dynamic_clearance_m=np.array([r['dynamic_clearance_m'] for r in mission.records]))
    print('M12_CAUSAL_SUMMARY', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', default=r'C:\Program Files\Blender Foundation\Blender 3.5\blender.exe')
    parser.add_argument('--max-images', type=int, default=60)
    args = parser.parse_args()
    run_probe(args.blender, args.max_images)


if __name__ == '__main__':
    main()
