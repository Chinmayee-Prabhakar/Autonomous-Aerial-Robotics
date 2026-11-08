"""Render synchronized M11 states through the M12 Blender facility.

Use Blender 3.5's bundled Python with the generated .blend as the startup file.
Operational evaluation receives only written RGB PNGs, never this scene graph.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_envelope import measure_vehicle_mesh


def options():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--group', choices=('evaluation', 'onboard', 'sync', 'mission',
                                           'arm', 'calibration', 'causal'), required=True)
    parser.add_argument('--job-file', default='')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--job-ids', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('CPU', 'CUDA', 'OPTIX'), default='CPU')
    return parser.parse_args(args)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def window_rotation(window):
    right = Vector(window['right_world'])
    up = Vector(window['up_world'])
    return Matrix((right, up, right.cross(up))).transposed()


def configure_device(device):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    if device == 'CPU':
        scene.cycles.device = 'CPU'
        return 'CPU'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    selected = [d for d in preferences.devices
                if d.type == device and 'NVIDIA' in d.name.upper()]
    if not selected:
        raise RuntimeError(f'{device} NVIDIA rendering device unavailable')
    for entry in preferences.devices:
        entry.use = entry in selected
    scene.cycles.device = 'GPU'
    return device


def apply_job(job, plan, group):
    scene = bpy.context.scene
    settings = plan['settings']
    width = settings['render']['camera_width_px'] if group in ('evaluation', 'onboard', 'sync', 'causal',
                                                               'calibration') else settings['render']['preview_width_px']
    height = settings['render']['camera_height_px'] if group in ('evaluation', 'onboard', 'sync', 'causal',
                                                                'calibration') else settings['render']['preview_height_px']
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.cycles.samples = settings['render']['samples']
    scene.cycles.seed = settings['appearance']['seed']
    quad = bpy.data.objects['m12_quadrotor_root']
    quad.location = job['position_world_m']
    quad.rotation_mode = 'QUATERNION'
    quad.rotation_quaternion = Quaternion(job['quaternion_bw'])
    arm = plan['scene']['arm']
    window = next(w for w in plan['scene']['windows'] if w['window_id'] == 'delta')
    R = window_rotation(window)
    # A causal run supplies its own arm condition. Historical M12 jobs retain
    # the scene-plan values when these optional fields are absent.
    angle = (job.get('arm_initial_angle_rad', arm['initial_angle_rad'])
             + job.get('arm_angular_velocity_rad_s', arm['angular_velocity_rad_s'])
             * job['time_s'])
    rot_z = Matrix.Rotation(angle, 4, 'Z')
    arm_root = bpy.data.objects['m12_arm_root']
    arm_root.matrix_world = (Matrix.Translation(Vector(window['center_world_m']))
                             @ R.to_4x4() @ rot_z)
    for obj in bpy.data.objects:
        if obj.get('m12_color_id'):
            name = (f"{obj['m12_color_id']}_colored_"
                    f"{job['material']}_safety_trim")
            obj.data.materials[0] = bpy.data.materials[name]
        if obj.get('m12_optional_occluder'):
            obj.hide_render = not (job['occlusion'] and obj.name.startswith(
                str(job['target_id']) + '_'))
        if obj.type == 'LIGHT' and 'm12_base_energy' in obj:
            factor = settings['appearance'][job['profile'] + '_energy_scale']
            obj.data.energy = obj['m12_base_energy'] * factor
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = (
        .28 * settings['appearance'][job['profile'] + '_energy_scale'])
    onboard = bpy.data.objects['m12_camera_onboard']
    onboard.matrix_world = (Matrix.Translation(Vector(job['camera_center_world_m']))
        @ Matrix(job['camera_rotation_blender_to_world']).to_4x4())
    external = bpy.data.objects['m12_camera_external']
    if group in ('evaluation', 'onboard', 'sync', 'causal', 'calibration'):
        scene.camera = onboard
        bpy.data.objects['facility_ceiling'].hide_render = False
    elif group == 'arm':
        scene.camera = external
        external.location = (18.7, -3.05, 2.60)
        look_at(external, (20.7, .1, 1.55))
        external.data.lens = 28.
        bpy.data.objects['facility_ceiling'].hide_render = False
    else:
        scene.camera = external
        pos = Vector(job['position_world_m'])
        # Keep the tracking camera in the vehicle's current room. A fixed
        # behind-vehicle offset could place it inside a partition at crossings.
        wall_x = [w['center_world_m'][0] for w in plan['scene']['windows']]
        closest = min(wall_x, key=lambda x: abs(pos.x - x))
        camera_x = pos.x + (1.45 if pos.x > closest + .18 else -1.45)
        if pos.x < closest - .18:
            camera_x = pos.x - 1.45
        external.location = (max(-1.45, camera_x), pos.y - 2.75,
                             min(3.65, max(2.9, pos.z + 2.0)))
        look_at(external, (pos.x + .55, pos.y, max(.5, pos.z)))
        external.data.lens = 24.
        bpy.data.objects['facility_ceiling'].hide_render = False
    scene.frame_set(int(round(job['time_s'] * 100)) + 1)
    bpy.context.view_layer.update()
    return angle


def calibration(plan, out):
    scene = bpy.context.scene
    geometry = {'physical_wall_boxes': 0, 'windows': {}}
    for window in plan['scene']['windows']:
        wid = window['window_id']
        R = window_rotation(window)
        center = Vector(window['center_world_m'])
        iw, ih = window['opening_width_m']/2, window['opening_height_m']/2
        ow, oh = window['outer_width_m']/2, window['outer_height_m']/2
        d = window['frame_thickness_m']/2
        boxes = (((-ow, -oh, -d), (-iw, oh, d)),
                 ((iw, -oh, -d), (ow, oh, d)),
                 ((-iw, -oh, -d), (iw, -ih, d)),
                 ((-iw, ih, -d), (iw, oh, d)))
        for index, (lo, hi) in enumerate(boxes):
            obj = bpy.data.objects[f'{wid}_physical_wall_{index}']
            midpoint = center + R @ ((Vector(lo)+Vector(hi))/2)
            if ((obj.location-midpoint).length > 1e-5 or
                    max(abs(obj.rotation_euler.to_matrix()[i][j]-R[i][j])
                        for i in range(3) for j in range(3)) > 1e-5 or
                    (obj.scale-(Vector(hi)-Vector(lo))).length > 1e-5 or
                    any(abs(a-b)>1e-8 for a,b in zip(obj['m11_local_min_m'],lo)) or
                    any(abs(a-b)>1e-8 for a,b in zip(obj['m11_local_max_m'],hi))):
                raise RuntimeError(f'physical wall mismatch: {obj.name}')
            geometry['physical_wall_boxes'] += 1
        geometry['windows'][wid] = {'opening_width_m': iw*2,
            'opening_height_m': ih*2, 'wall_thickness_m': d*2,
            'center_world_m': list(center)}
    if geometry['physical_wall_boxes'] != 16:
        raise RuntimeError('physical wall box count mismatch')
    for required in ('facility_floor', 'facility_ceiling',
                     'home_landing_ring', 'm12_camera_onboard',
                     'm12_quadrotor_root', 'm12_arm_root'):
        if required not in bpy.data.objects:
            raise RuntimeError('missing scene component: '+required)
    quad_root = bpy.data.objects['m12_quadrotor_root']
    measured = measure_vehicle_mesh(quad_root)
    if measured['maximum_radius_m'] > plan['scene']['vehicle_collision_envelope_m'] + 1e-9:
        raise RuntimeError('quad visual exceeds vehicle collision sphere')
    if abs(measured['maximum_radius_m'] -
           float(quad_root['visual_max_radius_m_measured'])) > 1e-8:
        raise RuntimeError('stored quad mesh bound differs from evaluated mesh')
    # Deliberately violate the radius to prove the geometry test rejects it.
    skid = next(o for o in quad_root.children if o.name.startswith('quad_landing_skid_'))
    original_z = skid.location.z
    try:
        skid.location.z = -.35
        oversized = measure_vehicle_mesh(quad_root)
        if oversized['maximum_radius_m'] <= plan['scene']['vehicle_collision_envelope_m']:
            raise RuntimeError('oversized landing-gear negative test failed')
    finally:
        skid.location.z = original_z
        bpy.context.view_layer.update()
    shaft = bpy.data.objects['arm_yellow_capsule_shaft']
    arm = plan['scene']['arm']
    shaft_local_length = max(v.co.z for v in shaft.data.vertices) - min(
        v.co.z for v in shaft.data.vertices)
    if (abs(float(shaft['m11_arm_radius_m']) - arm['radius_m']) > 1e-8 or
            abs(float(shaft_local_length) - arm['length_m']) > 1e-5):
        raise RuntimeError('visual arm size mismatch')
    if (abs(arm['pivot_right_m']) > 1e-9 or abs(arm['pivot_up_m']) > 1e-9):
        raise RuntimeError('M11 arm pivot is not centered')
    geometry['quad_visual_mesh'] = measured
    geometry['oversized_landing_gear_rejected'] = True
    geometry['oversized_landing_gear_radius_m'] = oversized['maximum_radius_m']
    geometry['arm_length_m'] = arm['length_m']
    geometry['arm_radius_m'] = arm['radius_m']
    geometry['floor_and_ceiling_present'] = True
    geometry['landing_ring_center_xy_m'] = list(
        bpy.data.objects['home_landing_ring'].location[:2])
    report = []
    for job in plan['jobs']:
        if job['group'] != 'evaluation' or job['true_corners_uv'] is None:
            continue
        apply_job(job, plan, 'calibration')
        if (bpy.data.objects['m12_arm_root'].matrix_world.translation -
                Vector(next(w['center_world_m'] for w in plan['scene']['windows']
                            if w['window_id'] == 'delta'))).length > 1e-6:
            raise RuntimeError('visual arm pivot is displaced from M11 center')
        camera = bpy.data.objects['m12_camera_onboard']
        pixels = []
        for point in job['true_corners_world_m']:
            ndc = world_to_camera_view(scene, camera, Vector(point))
            pixels.append((ndc.x * scene.render.resolution_x,
                           (1 - ndc.y) * scene.render.resolution_y))
        reference = job['true_corners_uv']
        residuals = [math.hypot(a[0] - b[0], a[1] - b[1])
                     for a, b in zip(pixels, reference)]
        report.append({'job_id': job['job_id'], 'target_id': job['target_id'],
                       'side': job['expected_side'], 'time_s': job['time_s'],
                       'python_uv': reference, 'blender_uv': pixels,
                       'corner_residual_px': residuals,
                       'maximum_residual_px': max(residuals)})
    landing = next(j for j in reversed(plan['jobs']) if j['group'] == 'mission')
    apply_job(landing, plan, 'calibration')
    landing_mesh = measure_vehicle_mesh(quad_root)
    floor = bpy.data.objects['facility_floor']
    floor_top = floor.location.z + floor.dimensions.z / 2
    visual_gap = landing_mesh['world_lowest_z_m'] - floor_top
    proxy_gap = landing['position_world_m'][2] - .25 - floor_top
    if not (0 <= visual_gap <= .06 and 0 <= proxy_gap <= .03):
        raise RuntimeError('landing display/contact proxy inconsistent with saved pose')
    geometry['landing_visual_floor_gap_m'] = visual_gap
    geometry['landing_contact_proxy_floor_gap_m'] = proxy_gap
    result = {'camera_intrinsics_px': plan['scene']['camera_fx_fy_cx_cy_px'],
              'resolution_px': plan['scene']['camera_image_width_height_px'],
              'world_frame': 'ENU equals Blender XYZ',
              'body_frame': 'FLU equals quad object local XYZ',
              'optical_RDF_from_blender_camera': [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
              'quaternion_order': 'scalar-first w,x,y,z body-to-world',
              'evaluated_views': len(report),
              'scene_geometry': geometry,
              'maximum_corner_residual_px': max(x['maximum_residual_px'] for x in report),
              'views': report}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'camera_projection_checks.json').write_text(json.dumps(result, indent=2))
    print('M12_CALIBRATION', json.dumps({'views': len(report),
          'maximum_corner_residual_px': result['maximum_corner_residual_px']}))


def render_group(plan, out, group, limit, requested, resume, external_job=None):
    jobs = [external_job] if external_job is not None else [
        j for j in plan['jobs'] if j['group'] == group]
    if requested:
        jobs = [j for j in jobs if j['job_id'] in requested]
    if limit is not None:
        jobs = jobs[:limit]
    root = out / 'rendered' / group
    root.mkdir(parents=True, exist_ok=True)
    log_path = out / f'{group}_render_log.jsonl'
    if not resume and not requested and limit is None and external_job is None:
        log_path.write_text('', encoding='utf-8')
    results = []
    for rank, job in enumerate(jobs, 1):
        image_path = root / (job['job_id'] + '.png')
        if resume and image_path.exists():
            print('M12_RESUME', rank, len(jobs), job['job_id'])
            continue
        start = time.perf_counter()
        item = {'job_id': job['job_id'], 'group': group,
                'time_s': job['time_s'],
                'file': image_path.relative_to(out).as_posix(),
                'render_seed': plan['settings']['appearance']['seed']}
        try:
            arm_angle = apply_job(job, plan, group)
            bpy.context.scene.render.filepath = str(image_path.resolve())
            bpy.ops.render.render(write_still=True)
            item.update({'ok': True, 'seconds': time.perf_counter() - start,
                         'arm_angle_rad': arm_angle,
                         'sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()})
        except Exception as exc:
            item.update({'ok': False, 'seconds': time.perf_counter() - start,
                         'error': str(exc)})
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item) + '\n')
        results.append(item)
        print('M12_RENDER', rank, len(jobs), job['job_id'], item['ok'],
              round(item['seconds'], 3), flush=True)
    print('M12_RENDER_SUMMARY', json.dumps({'group': group,
          'attempted': len(results), 'succeeded': sum(x['ok'] for x in results),
          'failed': [x['job_id'] for x in results if not x['ok']]}))


def main():
    opt = options()
    plan = json.loads(Path(opt.plan).read_text(encoding='utf-8'))
    out = Path(opt.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    actual = configure_device(opt.device)
    print('M12_RENDER_BACKEND', actual, bpy.app.version_string)
    if opt.group == 'calibration':
        calibration(plan, out)
    else:
        external_job = (json.loads(Path(opt.job_file).read_text(encoding='utf-8'))
                        if opt.job_file else None)
        if external_job is not None and opt.group != 'causal':
            raise ValueError('external jobs are reserved for causal rendering')
        render_group(plan, out, opt.group, opt.limit,
                     set(filter(None, opt.job_ids.split(','))), opt.resume,
                     external_job)


if __name__ == '__main__':
    main()
