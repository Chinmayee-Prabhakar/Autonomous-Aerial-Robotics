"""Measure the evaluated display mesh in the quadrotor body frame."""

import bpy


def measure_vehicle_mesh(root):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    root_inverse = root.evaluated_get(graph).matrix_world.inverted()
    maximum = 0.0
    body_lowest = float('inf')
    world_lowest = float('inf')
    furthest = None
    vertices = 0
    for child in root.children_recursive:
        if child.type != 'MESH':
            continue
        evaluated = child.evaluated_get(graph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=graph)
        try:
            body_transform = root_inverse @ evaluated.matrix_world
            for vertex in mesh.vertices:
                body_point = body_transform @ vertex.co
                world_point = evaluated.matrix_world @ vertex.co
                radius = body_point.length
                if radius > maximum:
                    maximum, furthest = radius, child.name
                body_lowest = min(body_lowest, body_point.z)
                world_lowest = min(world_lowest, world_point.z)
                vertices += 1
        finally:
            evaluated.to_mesh_clear()
    if not vertices:
        raise RuntimeError('quadrotor has no evaluated mesh vertices')
    return {'maximum_radius_m': maximum, 'body_lowest_z_m': body_lowest,
            'world_lowest_z_m': world_lowest, 'furthest_object': furthest,
            'evaluated_vertices': vertices}
