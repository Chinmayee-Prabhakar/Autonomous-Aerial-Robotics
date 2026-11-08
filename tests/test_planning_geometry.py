"""Analytic clearance and input-validation checks for M6."""

import math
import unittest

from aerial_autonomy.planning.geometry import (
    AABB, Sphere, Workspace, World, segment_aabb_distance, segment_sphere_distance,
)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = Workspace((0, 0, 0), (10, 10, 10))

    def test_invalid_workspace_and_obstacles(self) -> None:
        for low, high in [((0, 0, 0), (0, 1, 1)), ((0, 0, 0), (1, float("inf"), 1))]:
            with self.assertRaises(ValueError):
                Workspace(low, high)
        with self.assertRaises(ValueError):
            Sphere((1, 2, 3), 0)
        with self.assertRaises(ValueError):
            AABB((1, 2, 3), (2, 2, 4))
        with self.assertRaises(ValueError):
            Sphere((1, float("nan"), 3), 1)
        with self.assertRaises(ValueError):
            World(self.space, (), 0.1, -0.01)
        with self.assertRaises(ValueError):
            World(Workspace((0, 0, 0), (0.2, 1, 1)), (), 0.1)

    def test_boundary_clearance_and_vehicle_envelope(self) -> None:
        world = World(self.space, (), 0.2, 0.1)
        self.assertFalse(world.point_free((0.3, 5, 5)))
        self.assertFalse(world.point_free((0.3000000001, 5, 5)))
        self.assertTrue(world.point_free((0.31, 5, 5)))
        self.assertFalse(world.segment_free((0.31, 5, 5), (9.7, 5, 5)))
        self.assertAlmostEqual(world.segment_clearance_m((0.31, 5, 5), (9.69, 5, 5)), 0.01)

    def test_sphere_tangent_near_contact_and_endpoints(self) -> None:
        world = World(self.space, (Sphere((5, 5, 5), 1),), 0.2)
        self.assertFalse(world.segment_free((2, 6.2, 5), (8, 6.2, 5)))
        self.assertFalse(world.segment_free((2, 6.2000000001, 5), (8, 6.2000000001, 5)))
        self.assertTrue(world.segment_free((2, 6.21, 5), (8, 6.21, 5)))
        self.assertFalse(world.segment_free((6.2, 5, 5), (8, 5, 5)))
        self.assertAlmostEqual(segment_sphere_distance((0, 0, 0), (2, 0, 0), (3, 0, 0)), 1)
        self.assertAlmostEqual(segment_sphere_distance((0, 0, 0), (0, 0, 0), (3, 0, 0)), 3)

    def test_aabb_whole_segment_and_corner_distance(self) -> None:
        box = AABB((4, 4, 4), (6, 6, 6))
        world = World(self.space, (box,), 0.2)
        # Both endpoints are free; the middle of the edge crosses the box.
        self.assertTrue(world.point_free((2, 5, 5)))
        self.assertTrue(world.point_free((8, 5, 5)))
        self.assertFalse(world.segment_free((2, 5, 5), (8, 5, 5)))
        self.assertFalse(world.segment_free((2, 6.2, 5), (8, 6.2, 5)))
        self.assertTrue(world.segment_free((2, 6.21, 5), (8, 6.21, 5)))
        self.assertAlmostEqual(segment_aabb_distance((2, 7, 5), (3, 7, 5), box), math.sqrt(2))
        self.assertAlmostEqual(segment_aabb_distance((3, 7, 5), (3, 7, 5), box), math.sqrt(2))
        self.assertAlmostEqual(segment_aabb_distance((2, 5, 5), (8, 5, 5), box), 0)

    def test_outside_obstacle_and_bad_segment_input(self) -> None:
        world = World(self.space, (AABB((-4, -4, -4), (-2, -2, -2)),), 0.2)
        self.assertTrue(world.segment_free((1, 1, 1), (9, 9, 9)))
        with self.assertRaises(ValueError):
            world.segment_free((1, 2), (3, 4, 5))
        with self.assertRaises(ValueError):
            world.segment_free((1, 2, float("inf")), (3, 4, 5))


if __name__ == "__main__":
    unittest.main()
