"""Audit M12 visual replay against immutable M11 records and Blender exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.realism.bridge import (camera_blender_pose, nearest_index,
    project_path, state_from_trace, window_face_corners)
from aerial_autonomy.realism.detector import AdaptiveColorWindowDetector
from aerial_autonomy.simulation.dynamics import VehicleState


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(plan_path, output_dir):
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    scene = plan['scene']
    trial = project_path(plan_path, plan['settings']['source']['trial_dir'])
    trace = np.load(trial / 'physical_trace.npz')
    _, _, _, camera, world, arm, colors, _ = load_m11_config(
        project_path(plan_path, plan['settings']['source']['mission_config']))
    for path, expected in plan['source_hashes_sha256'].items():
        if sha256(project_path(plan_path, path)) != expected:
            raise AssertionError('accepted M11 source changed: ' + path)
    if plan['source_m11_head'] != plan['settings']['source']['expected_m11_head']:
        raise AssertionError('M11 checkpoint mismatch')
    if scene['world_frame'] != 'ENU' or scene['body_frame'] != 'FLU':
        raise AssertionError('coordinate frame mismatch')
    if (camera.width_px, camera.height_px) != tuple(scene['camera_image_width_height_px']):
        raise AssertionError('camera resolution mismatch')
    if not np.allclose([camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px],
                       scene['camera_fx_fy_cx_cy_px']):
        raise AssertionError('camera intrinsics mismatch')
    if abs(scene['vehicle_collision_envelope_m']-world.envelope_m) > 1e-10:
        raise AssertionError('vehicle collision envelope mismatch')
    exported_arm = scene['arm']
    for field, expected in [('pivot_right_m', arm.shape.pivot_right_m),
                            ('pivot_up_m', arm.shape.pivot_up_m),
                            ('length_m', arm.shape.length_m),
                            ('radius_m', arm.shape.arm_radius_m),
                            ('initial_angle_rad', arm.initial_angle_rad),
                            ('angular_velocity_rad_s', arm.angular_velocity_rad_s)]:
        if abs(exported_arm[field]-expected) > 1e-10:
            raise AssertionError('arm geometry mismatch: '+field)
    if len(scene['windows']) != 4:
        raise AssertionError('expected four M11 openings')
    for window in scene['windows']:
        if window['frame_thickness_m'] != .16 or window['opening_width_m'] <= 0:
            raise AssertionError('physical window geometry invalid')
        if window['opening_width_m'] >= window['outer_width_m']:
            raise AssertionError('opening exceeds partition')
    positions = np.asarray(trace['position_m'])
    intersample = float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))/2)
    min_decor = float('inf')
    for item in plan['decor']:
        lo, hi = np.asarray(item['minimum_m']), np.asarray(item['maximum_m'])
        outside = np.maximum(np.maximum(lo-positions, positions-hi), 0)
        extra = float(np.min(np.linalg.norm(outside, axis=1))-world.envelope_m-
                      intersample)
        if extra <= .1 or abs(extra-item['minimum_extra_clearance_m']) > 1e-7:
            raise AssertionError('decor clearance mismatch: '+item['name'])
        min_decor = min(min_decor, extra)
    if np.linalg.norm(positions[0, :2]) > .01 or np.linalg.norm(positions[-1, :2]) > .1:
        raise AssertionError('home/landing location mismatch')
    delta = next(w for w in scene['windows'] if w['window_id'] == 'delta')
    crossing = np.where((positions[:-1, 0] > delta['center_world_m'][0]) &
                        (positions[1:, 0] <= delta['center_world_m'][0]))[0]
    if not any(abs(positions[i, 1]) > delta['outer_width_m']/2 + world.envelope_m
               for i in crossing):
        raise AssertionError('accepted exterior bypass missing')
    max_pose_error = 0.
    ids = set()
    counts = {}
    render_audit = {}
    for group in sorted({j['group'] for j in plan['jobs']}):
        log = output_dir / (group + '_render_log.jsonl')
        latest = {x['job_id']: x for x in
                  (json.loads(line) for line in log.read_text(encoding='utf-8').splitlines())}
        render_audit[group] = latest
        counts[group] = 0
    for job in plan['jobs']:
        if job['job_id'] in ids:
            raise AssertionError('duplicate job identifier')
        ids.add(job['job_id'])
        group = job['group']
        counts[group] += 1
        index = job['trace_index']
        if abs(job['time_s']-float(trace['time_s'][index])) > 1e-8:
            raise AssertionError('saved time alignment mismatch')
        for name, value in [('position_m', job['position_world_m']),
                            ('quaternion_bw', job['quaternion_bw']),
                            ('torque_nm', job['torque_nm'])]:
            error = float(np.max(abs(np.asarray(trace[name][index])-value)))
            max_pose_error = max(max_pose_error, error)
            if error > 1e-8:
                raise AssertionError('saved state/command mismatch: '+job['job_id'])
        for key, source in [('thrust_n', 'thrust_n'),
                            ('static_clearance_m', 'static_clearance_m'),
                            ('dynamic_clearance_m', 'dynamic_clearance_m')]:
            if abs(job[key]-float(trace[source][index])) > 1e-8:
                raise AssertionError('saved physical metric mismatch')
        center, rotation = camera_blender_pose(state_from_trace(trace, index), camera)
        if (np.max(abs(center-job['camera_center_world_m'])) > 1e-9 or
                np.max(abs(rotation-job['camera_rotation_blender_to_world'])) > 1e-9):
            raise AssertionError('saved camera pose mismatch')
        image = output_dir / 'rendered' / group / (job['job_id']+'.png')
        entry = render_audit[group].get(job['job_id'])
        if entry is None or not entry['ok'] or not image.is_file():
            raise AssertionError('render missing/failed: '+job['job_id'])
        if entry.get('render_seed') != plan['settings']['appearance']['seed']:
            raise AssertionError('render seed mismatch: '+job['job_id'])
        if sha256(image) != entry['sha256']:
            raise AssertionError('render SHA mismatch: '+job['job_id'])
        expected_size = ((320, 240) if group in ('evaluation', 'onboard', 'sync')
                         else (480, 270))
        with Image.open(image) as frame:
            if frame.size != expected_size:
                raise AssertionError('render resolution mismatch')
        expected_angle = (arm.initial_angle_rad +
                          arm.angular_velocity_rad_s*job['time_s'])
        if abs(entry['arm_angle_rad']-expected_angle) > 1e-8:
            raise AssertionError('render arm clock mismatch')
    calibration = json.loads((output_dir/'camera_projection_checks.json').read_text())
    if (calibration['evaluated_views'] < 20 or
            calibration['maximum_corner_residual_px'] > .1 or
            calibration['scene_geometry']['physical_wall_boxes'] != 16):
        raise AssertionError('Blender calibration/geometry validation failed')
    geometry = calibration['scene_geometry']
    mesh = geometry['quad_visual_mesh']
    if (mesh['maximum_radius_m'] > world.envelope_m + 1e-9 or
            mesh['evaluated_vertices'] < 100 or
            not geometry['oversized_landing_gear_rejected'] or
            geometry['oversized_landing_gear_radius_m'] <= world.envelope_m or
            not 0 <= geometry['landing_visual_floor_gap_m'] <= .06 or
            not 0 <= geometry['landing_contact_proxy_floor_gap_m'] <= .03):
        raise AssertionError('evaluated quad mesh/landing containment failed')
    frames = np.load(trial/'camera_frames.npz')
    native_count = native_in_frame = 0
    native_max_residual = 0.
    for job in (j for j in plan['jobs'] if j['group'] == 'evaluation'):
        for folder in ('annotations', 'detections', 'overlays'):
            suffix = '.png' if folder == 'overlays' else '.json'
            if not (output_dir/folder/(job['job_id']+suffix)).is_file():
                raise AssertionError('perception evidence missing')
        if job['source_camera_frame'] is not None:
            native_count += 1
            f = job['source_camera_frame']
            if (abs(float(frames['time_s'][f])-job['time_s']) > 1e-8 or
                    str(frames['target_id'][f]) != job['target_id']):
                raise AssertionError('native image timestamp or target mismatch')
            physical = next(w for w in scene['windows']
                            if w['window_id'] == job['target_id'])
            corners_world, side = window_face_corners(physical,
                camera_blender_pose(state_from_trace(trace, job['trace_index']),
                                    camera)[0])
            if side != job['expected_side']:
                raise AssertionError('native visible face mismatch')
            optical = camera.world_to_camera(
                state_from_trace(trace, job['trace_index']), corners_world)
            uv, ahead, _ = camera.project(optical)
            truth = json.loads((output_dir/'annotations'/
                (job['job_id']+'_native.json')).read_text(encoding='utf-8'))
            recorded = truth['true_corners_uv']
            if recorded is None:
                if np.all(ahead):
                    raise AssertionError('missing in-front native geometry')
                usable = False
            else:
                native_max_residual = max(native_max_residual,
                    float(np.max(abs(np.asarray(recorded)-uv))))
                usable = bool(np.all(ahead) and np.all(np.isfinite(uv)) and
                    np.all((uv[:, 0] >= 0) & (uv[:, 0] < camera.width_px)) and
                    np.all((uv[:, 1] >= 0) & (uv[:, 1] < camera.height_px)))
            if bool(truth['truth_usable']) != usable or native_max_residual > 1e-8:
                raise AssertionError('native geometric annotation mismatch')
            native_in_frame += usable
    if not (output_dir/'indoor_facility.blend').is_file():
        raise AssertionError('Blender scene file missing')
    causal_audit_path = output_dir/'causal_alpha'/'independent_audit.json'
    if not causal_audit_path.is_file():
        raise AssertionError('complete M12 audit requires causal alpha evidence')
    causal = None
    causal_rgb_checked = 0
    if causal_audit_path.is_file():
        causal = json.loads(causal_audit_path.read_text(encoding='utf-8'))
        if (not causal['passed'] or len(causal['crossings']) != 1 or
                causal['crossings'][0]['window_id'] != 'alpha'):
            raise AssertionError('limited causal crossing audit failed')
        causal_root = output_dir/'causal_alpha'
        causal_trace = np.load(causal_root/'physical_trace.npz')
        causal_observations = json.loads((causal_root/'observations.json').read_text())
        causal_summary = json.loads((causal_root/'summary.json').read_text())
        causal_logs = {row['job_id']: row for row in (json.loads(line)
            for line in (causal_root/'causal_render_log.jsonl').read_text().splitlines())}
        if (len(causal_observations) != causal_summary['blender_images'] or
                len(causal_logs) != len(causal_observations) or
                sum(bool(x['window_valid']) for x in causal_observations) !=
                    causal_summary['valid_window_observations']):
            raise AssertionError('causal RGB/observation count mismatch')
        detector = AdaptiveColorWindowDetector(colors)
        for number, observation in enumerate(causal_observations):
            name = f'causal_{number:03d}'
            job = json.loads((causal_root/'jobs'/(name+'.json')).read_text())
            time_s = float(observation['time_s'])
            index = nearest_index(np.asarray(causal_trace['time_s']), time_s)
            if (abs(float(causal_trace['time_s'][index])-time_s) > 1e-8 or
                    abs(job['time_s']-time_s) > 1e-8):
                raise AssertionError('causal camera/plant time mismatch: '+name)
            state = VehicleState(np.asarray(causal_trace['position_m'][index]),
                np.zeros(3), np.asarray(causal_trace['quaternion_bw'][index]),
                np.zeros(3))
            center, rotation = camera_blender_pose(state, camera)
            if (np.max(abs(state.position_m-job['position_world_m'])) > 1e-8 or
                    np.max(abs(state.quaternion_bw-job['quaternion_bw'])) > 1e-8 or
                    np.max(abs(center-job['camera_center_world_m'])) > 1e-8 or
                    np.max(abs(rotation-job['camera_rotation_blender_to_world'])) > 1e-8):
                raise AssertionError('causal RGB pose mismatch: '+name)
            log = causal_logs[name]
            image_path = causal_root/'rendered'/'causal'/(name+'.png')
            if not log['ok'] or sha256(image_path) != log['sha256']:
                raise AssertionError('causal RGB hash mismatch: '+name)
            with Image.open(image_path) as frame:
                rgb = np.asarray(frame.convert('RGB'), dtype=np.uint8)
            detection = detector.detect(rgb, camera, 'alpha', 'positive_normal')
            recorded = observation['detected_corners_uv']
            if bool(detection.valid) != (recorded is not None):
                raise AssertionError('causal pixel validity mismatch: '+name)
            if recorded is not None and np.max(abs(detection.corners_uv-recorded)) > 1e-8:
                raise AssertionError('causal pixel corners mismatch: '+name)
            causal_rgb_checked += 1
    report = {'schema': 'm12_physical_visual_audit_v1',
              'passed': True, 'accepted_m11_source_hashes_verified': True,
              'total_render_jobs_verified': len(plan['jobs']),
              'jobs_by_group': counts,
              'maximum_saved_state_command_error': max_pose_error,
              'minimum_decor_extra_clearance_m': min_decor,
              'blender_projection_maximum_residual_px':
                  calibration['maximum_corner_residual_px'],
              'physical_wall_boxes_validated_in_blender':
                  calibration['scene_geometry']['physical_wall_boxes'],
              'exterior_bypass_verified': True,
              'limited_causal_alpha_audit_passed': bool(causal and causal['passed']),
              'causal_rgb_poses_hashes_detections_verified': causal_rgb_checked,
              'native_physical_annotations_checked': native_count,
              'native_geometrically_in_frame': native_in_frame,
              'native_projection_maximum_residual_px': native_max_residual,
              'scene_and_images': output_dir.as_posix()}
    (output_dir/'physical_visual_audit.json').write_text(json.dumps(report, indent=2),
                                                        encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='outputs/realism_m12/scene_plan.json')
    parser.add_argument('--output-dir', default='outputs/realism_m12')
    args = parser.parse_args()
    print(json.dumps(validate(args.plan, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
