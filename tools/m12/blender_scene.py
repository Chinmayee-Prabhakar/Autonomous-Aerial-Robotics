"""Build the M12 indoor facility from the accepted M11 exported scene.

Run with Blender 3.5's bundled Python. ENU and Blender world axes coincide.
Every colored traversal opening is a real void through the four M11 wall boxes.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_envelope import measure_vehicle_mesh
from mathutils import Matrix, Vector


def args_after_separator():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-blend', required=True)
    parser.add_argument('--preview-dir', required=True)
    return parser.parse_args(args)


def material(name, rgb, metallic=0., roughness=.75, emission=0., textured=False):
    mat = bpy.data.materials.new(name)
    mat.use_fake_user = True
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    p = nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*rgb, 1)
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = roughness
    if emission:
        p.inputs['Emission'].default_value = (*rgb, 1)
        p.inputs['Emission Strength'].default_value = emission
    if textured:
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 15
        noise.inputs['Detail'].default_value = 2
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .18
        ramp.color_ramp.elements[0].color = tuple(max(0, x*.78) for x in rgb) + (1,)
        ramp.color_ramp.elements[1].position = .82
        ramp.color_ramp.elements[1].color = tuple(min(1, x*1.13) for x in rgb) + (1,)
        mat.node_tree.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        mat.node_tree.links.new(ramp.outputs['Color'], p.inputs['Base Color'])
    return mat


def cube(name, center, dimensions, mat, parent=None, rotation=None, decorative=False):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
        obj.location = center
        obj.dimensions = dimensions
    else:
        obj.location = center
        obj.dimensions = dimensions
        if rotation is not None:
            obj.rotation_euler = rotation.to_euler()
    obj['m12_decorative_only'] = decorative
    return obj


def local_box(name, lo, hi, origin, rotation, mat, decorative=False):
    lo, hi = Vector(lo), Vector(hi)
    return cube(name, Vector(origin) + rotation @ ((lo + hi) / 2), hi - lo,
                mat, rotation=rotation, decorative=decorative)


def cylinder(name, radius, depth, center, mat, parent=None, rotation=None, vertices=20):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    obj.location = center
    if rotation is not None:
        obj.rotation_euler = rotation
    return obj


def sphere(name, radius, center, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    obj.location = center
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def window_rotation(window):
    right = Vector(window['right_world'])
    up = Vector(window['up_world'])
    normal = right.cross(up)
    return Matrix((right, up, normal)).transposed()


def add_architecture(plan, mats):
    cube('facility_floor', (12., 0., -.40), (28.2, 13.8, .30), mats['floor'])
    ceiling = cube('facility_ceiling', (12., 0., 4.42), (28.2, 13.8, .24),
                   mats['ceiling'])
    ceiling['m12_shell'] = True
    for y in (-6.9, 6.9):
        cube(f'outer_side_wall_{y:+.1f}', (12., y, 2.03),
             (28.2, .18, 4.84), mats['outer_wall'])
    for x in (-2.1, 26.1):
        cube(f'outer_end_wall_{x:+.1f}', (x, 0., 2.03),
             (.18, 13.8, 4.84), mats['outer_wall'])
    for x in (2.4, 7.5, 12.5, 17.7, 23.5):
        # Recessed overhead fixtures; all lie well above M11's flown corridor.
        cube(f'luminaire_{x:.1f}', (x, 0., 4.22), (1.1, .42, .055),
             mats['light_panel'], decorative=True)
        bpy.ops.object.light_add(type='AREA', location=(x, 0., 4.13))
        lamp = bpy.context.object
        lamp.name = f'ceiling_area_light_{x:.1f}'
        lamp.data.energy = 250
        lamp.data.shape = 'RECTANGLE'
        lamp.data.size = 2.8
        lamp.data.size_y = 2.0
        lamp.data.color = (1.0, .92, .79)
        lamp.rotation_euler = (0, 0, 0)
        lamp['m12_base_energy'] = 250.
    cube('bypass_floor_marking', (20.6, 5.3, -.245), (5.3, .10, .008),
         mats['floor_mark'], decorative=True)
    cylinder('home_landing_ring', .48, .007, (0., 0., -.244),
             mats['landing'], vertices=48)
    cylinder('home_landing_inner', .33, .009, (0., 0., -.239),
             mats['floor'], vertices=48)
    cube('outer_exit_door', (-1.99, -3.5, 1.1), (.012, 1.25, 2.25),
         mats['door'], decorative=True)
    cube('door_push_bar', (-1.98, -3.5, 1.0), (.025, .75, .045),
         mats['trim'], decorative=True)
    for x in (1.2, 6.4, 11.5, 16.6, 22.4):
        cube(f'wall_service_panel_{x:.1f}', (x, -6.79, 2.0),
             (1.35, .025, 1.35), mats['cabinet_door'], decorative=True)
        cube(f'wall_service_grille_{x:.1f}', (x, -6.77, 2.0),
             (.92, .013, .55), mats['trim'], decorative=True)
        cube(f'ceiling_cable_tray_{x:.1f}', (x, -4.8, 4.10),
             (2.7, .18, .08), mats['trim'], decorative=True)
    for y in (-1.05, 1.05):
        cube(f'flight_lane_floor_line_{y:+.2f}', (12., y, -.244),
             (26.0, .035, .008), mats['floor_mark'], decorative=True)
    for window in plan['scene']['windows']:
        wid = window['window_id']
        R = window_rotation(window)
        center = window['center_world_m']
        iw = window['opening_width_m'] / 2
        ih = window['opening_height_m'] / 2
        ow = window['outer_width_m'] / 2
        oh = window['outer_height_m'] / 2
        d = window['frame_thickness_m'] / 2
        boxes = (((-ow, -oh, -d), (-iw, oh, d)),
                 ((iw, -oh, -d), (ow, oh, d)),
                 ((-iw, -oh, -d), (iw, -ih, d)),
                 ((-iw, ih, -d), (iw, oh, d)))
        for index, (lo, hi) in enumerate(boxes):
            obj = local_box(f'{wid}_physical_wall_{index}', lo, hi,
                            center, R, mats['partition'])
            obj['m11_physical_wall'] = True
            obj['m11_window_id'] = wid
            obj['m11_local_min_m'] = lo
            obj['m11_local_max_m'] = hi
        # Colored industrial safety trims are entirely within the physical
        # wall's four solid strips. The flight passage remains exactly open.
        stripe = .16
        trims = (((-iw-stripe, -ih-stripe, -d), (-iw, ih+stripe, d)),
                 ((iw, -ih-stripe, -d), (iw+stripe, ih+stripe, d)),
                 ((-iw, -ih-stripe, -d), (iw, -ih, d)),
                 ((-iw, ih, -d), (iw, ih+stripe, d)))
        for index, (lo, hi) in enumerate(trims):
            obj = local_box(f'{wid}_safety_trim_{index}', lo, hi, center, R,
                            mats[f'{wid}_metal'])
            obj['m12_color_id'] = wid
            obj['m11_physical_wall'] = True
            for side_name, side_sign in (('front', 1), ('back', -1)):
                face_lo = (lo[0], lo[1], side_sign * (d + .0002) - .0004)
                face_hi = (hi[0], hi[1], side_sign * (d + .0002) + .0004)
                decal = local_box(f'{wid}_safety_face_{index}_{side_name}',
                    face_lo, face_hi, center, R, mats[f'{wid}_metal'])
                decal['m12_color_id'] = wid
                decal['m12_surface_decal'] = True
        # Matte metal outer trim also stays inside the M11 wall strips.
        edge = .10
        for index, (lo, hi) in enumerate((
            ((-iw-stripe-edge, -ih-stripe-edge, -d),
             (-iw-stripe, ih+stripe+edge, d)),
            ((iw+stripe, -ih-stripe-edge, -d),
             (iw+stripe+edge, ih+stripe+edge, d)))):
            local_box(f'{wid}_jamb_{index}', lo, hi, center, R, mats['trim'])
        # A small service plaque creates controlled, physically nonintruding
        # partial occlusion of the colored upper-left trim when enabled.
        plaque = local_box(f'{wid}_occluder',
            (-iw-.32, ih-.02, -d), (-iw-.01, ih+.40, d), center, R,
            mats['door'], decorative=True)
        plaque.hide_render = True
        plaque['m12_optional_occluder'] = True
    for item in plan['decor']:
        lo, hi = Vector(item['minimum_m']), Vector(item['maximum_m'])
        center = (lo + hi) / 2
        size = hi - lo
        if item['kind'] == 'bench':
            cube(item['name'] + '_top', (center.x, center.y, hi.z-.07),
                 (size.x, size.y, .12), mats['bench'], decorative=True)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cube(item['name'] + f'_leg_{sx}_{sy}',
                         (center.x+sx*(size.x/2-.08),
                          center.y+sy*(size.y/2-.07), (lo.z+hi.z)/2-.07),
                         (.065, .065, size.z-.14), mats['trim'], decorative=True)
            cube(item['name'] + '_instrument',
                 (center.x, center.y, hi.z+.13), (.32, .22, .22),
                 mats['instrument'], decorative=True)
        else:
            cube(item['name'], center, size, mats['cabinet'], decorative=True)
            cube(item['name'] + '_door', (center.x, hi.y+.004, center.z),
                 (size.x-.08, .009, size.z-.10), mats['cabinet_door'],
                 decorative=True)
            cube(item['name'] + '_handle', (center.x+.17, hi.y+.014, center.z),
                 (.025, .015, .22), mats['trim'], decorative=True)
    return ceiling


def add_arm(plan, mats):
    arm = plan['scene']['arm']
    window = next(w for w in plan['scene']['windows'] if w['window_id'] == 'delta')
    R = window_rotation(window)
    pivot = Vector(window['center_world_m'])
    root = bpy.data.objects.new('m12_arm_root', None)
    bpy.context.collection.objects.link(root)
    root.matrix_world = Matrix.Translation(pivot) @ R.to_4x4()
    length, radius = arm['length_m'], arm['radius_m']
    shaft = cylinder('arm_yellow_capsule_shaft', radius, length,
                     (length/2, 0, 0), mats['arm_yellow'], parent=root,
                     rotation=(0, math.pi/2, 0), vertices=24)
    shaft['m11_arm_radius_m'] = radius
    sphere('arm_tip_cap', radius, (length, 0, 0), mats['arm_yellow'], parent=root)
    sphere('arm_root_cap', radius, (0, 0, 0), mats['arm_yellow'], parent=root)
    cylinder('arm_magenta_hub', .065, .004, (0, 0, .103), mats['hub'],
             parent=root, vertices=24)
    return root


def add_quadrotor(mats):
    root = bpy.data.objects.new('m12_quadrotor_root', None)
    bpy.context.collection.objects.link(root)
    # The accepted camera lens is 0.08 m forward of body origin. Keep the
    # display shell behind that lens so the mounted camera sees the room.
    cube('quad_central_frame', (0, 0, 0), (.12, .16, .055),
         mats['quad_body'], parent=root)
    sphere('quad_forward_marker', .020, (.055, 0, .058),
           mats['forward_marker'], parent=root)
    for x in (-.13, .13):
        for y in (-.13, .13):
            endpoint = Vector((x, y, 0))
            cylinder(f'quad_arm_{x:+.2f}_{y:+.2f}', .013,
                     endpoint.length, endpoint/2, mats['quad_body'],
                     parent=root, rotation=endpoint.to_track_quat('Z', 'Y').to_euler(),
                     vertices=12)
            cylinder(f'quad_motor_{x:+.2f}_{y:+.2f}', .033, .045,
                     (x, y, .027), mats['motor'], parent=root, vertices=16)
            cylinder(f'quad_propeller_{x:+.2f}_{y:+.2f}', .051, .006,
                     (x, y, .055), mats['propeller'], parent=root, vertices=16)
    for y in (-.06, .06):
        cube(f'quad_landing_strut_{y:+.2f}', (0, y, -.119),
             (.015, .015, .204), mats['quad_body'], parent=root)
        cube(f'quad_landing_skid_{y:+.2f}', (0, y, -.226),
             (.12, .017, .012), mats['quad_body'], parent=root)
    root['m11_collision_envelope_m'] = .25
    measured = measure_vehicle_mesh(root)
    if measured['maximum_radius_m'] > root['m11_collision_envelope_m'] + 1e-9:
        raise RuntimeError('measured quad visual mesh exceeds collision sphere')
    root['visual_max_radius_m_measured'] = measured['maximum_radius_m']
    return root


def add_cameras(plan):
    scene = bpy.context.scene
    for name in ('m12_camera_onboard', 'm12_camera_external'):
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = name
        camera.data.type = 'PERSP'
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.sensor_width = 36.
        camera.data.lens = (plan['scene']['camera_fx_fy_cx_cy_px'][0] * 36 /
                            plan['scene']['camera_image_width_height_px'][0])
        camera.data.clip_start = .025
        camera.data.clip_end = 100.
        camera.data.shift_x = 0.
        camera.data.shift_y = 0.
    scene.camera = bpy.data.objects['m12_camera_onboard']


def set_render(plan):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = plan['settings']['render']['samples']
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (
        .33, .37, .45, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .28


def make_materials(plan):
    mats = {
        'floor': material('sealed_concrete_floor', (.30, .32, .34), roughness=.88,
                          textured=True),
        'ceiling': material('ceiling_acoustic_panel', (.57, .61, .63), roughness=.94),
        'outer_wall': material('painted_outer_wall', (.53, .58, .62), roughness=.84,
                               textured=True),
        'partition': material('painted_partition_wall', (.50, .54, .57), roughness=.82,
                              textured=True),
        'trim': material('brushed_anodized_metal', (.27, .32, .36), metallic=.72,
                         roughness=.38),
        'bench': material('laminate_lab_bench', (.27, .34, .39), roughness=.65),
        'instrument': material('lab_instrument', (.16, .21, .24), metallic=.18,
                               roughness=.55),
        'cabinet': material('industrial_cabinet', (.23, .28, .31), metallic=.38,
                            roughness=.58),
        'cabinet_door': material('industrial_cabinet_door', (.41, .47, .51),
                                 metallic=.35, roughness=.58),
        'door': material('service_door', (.26, .34, .38), metallic=.22, roughness=.67),
        'floor_mark': material('bypass_floor_stripe', (.57, .52, .22), roughness=.87),
        'landing': material('landing_ring_paint', (.76, .43, .10), roughness=.82),
        'light_panel': material('ceiling_diffuser', (.85, .82, .73), roughness=.85,
                                emission=.24),
        'arm_yellow': material('arm_safety_yellow', (.92, .66, .035), metallic=.12,
                               roughness=.55),
        'hub': material('arm_magenta_hub', (.78, .045, .66), roughness=.55),
        'quad_body': material('quad_carbon_frame', (.045, .07, .09), metallic=.20,
                              roughness=.46),
        'motor': material('quad_motor_metal', (.22, .27, .31), metallic=.78,
                          roughness=.35),
        'propeller': material('quad_propeller', (.10, .13, .15), metallic=.15,
                              roughness=.52),
        'forward_marker': material('quad_forward_red', (.85, .05, .025),
                                   roughness=.45),
    }
    for wid, rgb in plan['color_rgb_by_window'].items():
        # RGB values are specified by the accepted M11 target catalog; only
        # their physically lit material response changes for this M12 study.
        color = tuple(x / 255 for x in rgb)
        mats[f'{wid}_metal'] = material(f'{wid}_colored_metal_safety_trim',
            color, metallic=.16, roughness=.46, emission=.17)
        mats[f'{wid}_paint'] = material(f'{wid}_colored_paint_safety_trim',
            color, metallic=.02, roughness=.76, emission=.11)
    return mats


def render_previews(output, plan, ceiling):
    scene = bpy.context.scene
    cam = bpy.data.objects['m12_camera_external']
    scene.camera = cam
    scene.render.resolution_x = plan['settings']['render']['preview_width_px']
    scene.render.resolution_y = plan['settings']['render']['preview_height_px']
    scene.cycles.samples = plan['settings']['render']['preview_samples']
    ceiling.hide_render = True
    front_wall = bpy.data.objects['outer_side_wall_-6.9']
    front_wall.hide_render = True
    views = (
        ('environment_isometric.png', (10.5, -15.0, 15.), (11.8, 0., 1.4)),
        ('floor_plan.png', (12., 0., 35.), (12., 0., 0.)),
        ('interior_alpha.png', (1.1, 2.1, 2.45), (4.7, -.3, 1.45)),
        ('interior_delta.png', (17.3, 2.2, 2.5), (21., -.1, 1.55)),
    )
    for name, position, target in views:
        cam.location = position
        look_at(cam, target)
        if name.startswith('interior'):
            ceiling.hide_render = False
            front_wall.hide_render = False
        scene.render.filepath = str((output / name).resolve())
        bpy.ops.render.render(write_still=True)
    ceiling.hide_render = False
    front_wall.hide_render = False
    scene.cycles.samples = plan['settings']['render']['samples']
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240


def main():
    options = args_after_separator()
    plan = json.loads(Path(options.plan).read_text(encoding='utf-8'))
    out_blend = Path(options.output_blend).resolve()
    preview = Path(options.preview_dir).resolve()
    preview.mkdir(parents=True, exist_ok=True)
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    mats = make_materials(plan)
    set_render(plan)
    ceiling = add_architecture(plan, mats)
    add_arm(plan, mats)
    add_quadrotor(mats)
    add_cameras(plan)
    bpy.context.scene['m12_source_m11_head'] = plan['source_m11_head']
    bpy.context.scene['m12_scene_schema'] = plan['schema']
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    start = time.perf_counter()
    render_previews(preview, plan, ceiling)
    report = {'blend_path': out_blend.relative_to(preview.parent).as_posix(),
        'blender_version': bpy.app.version_string,
        'engine': bpy.context.scene.render.engine, 'device': bpy.context.scene.cycles.device,
        'architecture_window_ids': [w['window_id'] for w in plan['scene']['windows']],
        'physical_wall_boxes': len(plan['scene']['windows']) * 4,
        'decorative_prop_count': len(plan['decor']),
        'minimum_decor_extra_clearance_m': min(p['minimum_extra_clearance_m']
                                               for p in plan['decor']),
        'preview_render_seconds': time.perf_counter() - start,
        'preview_files': [(preview / view).relative_to(preview.parent).as_posix() for view in
            ('environment_isometric.png', 'floor_plan.png',
             'interior_alpha.png', 'interior_delta.png')]}
    (preview.parent / 'scene_build_report.json').write_text(json.dumps(report, indent=2))
    print('M12_SCENE_BUILD', json.dumps(report))


if __name__ == '__main__':
    main()
