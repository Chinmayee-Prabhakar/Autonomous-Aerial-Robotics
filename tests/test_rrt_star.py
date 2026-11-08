"""RRT* path, tree, replay and failure invariants."""

import math
from pathlib import Path
import unittest

from aerial_autonomy.evaluation.rrt_star import aggregate, load_benchmark
from aerial_autonomy.planning.geometry import AABB, Sphere, Workspace, World
from aerial_autonomy.planning.rrt_star import PlannerConfig, Tree, _select_parent, audit_result, plan_from_geometry, plan_rrt_star


class TreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(Workspace((0, 0, 0), (10, 10, 5)), (), 0.1)

    def test_parent_selection_uses_full_path_cost(self) -> None:
        tree = Tree((1, 1, 1))
        expensive = tree.add((1, 8, 1), 0)
        cheap = tree.add((5, 5, 1), 0)
        target = (7, 7, 1)
        self.assertEqual(_select_parent(tree, target, (expensive, cheap), self.world.segment_free), cheap)

    def test_rewire_propagates_descendant_costs_and_prevents_cycle(self) -> None:
        tree = Tree((1, 1, 1))
        a = tree.add((1, 8, 1), 0)
        b = tree.add((8, 8, 1), a)
        c = tree.add((8, 9, 1), b)
        new = tree.add((5, 5, 1), 0)
        old_b, old_c = tree.costs[b], tree.costs[c]
        self.assertFalse(tree.rewire(a, c, self.world))
        self.assertTrue(tree.rewire(b, new, self.world))
        self.assertLess(tree.costs[b], old_b)
        self.assertAlmostEqual(old_b - tree.costs[b], old_c - tree.costs[c])
        self.assertEqual(tree.parents[c], b)
        self.assertEqual(tree.path_indices(c), (0, new, b, c))
        self.assertTrue(tree.validate(self.world))

    def test_rewire_rejects_colliding_new_edge(self) -> None:
        world = World(self.world.workspace, (Sphere((6.5, 4.5, 1), 0.7),), 0.1)
        tree = Tree((1, 1, 1))
        a = tree.add((1, 8, 1), 0)
        b = tree.add((8, 8, 1), a)
        new = tree.add((5, 1, 1), 0)
        self.assertFalse(tree.rewire(b, new, world))
        self.assertEqual(tree.parents[b], a)


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Workspace((0, 0, 0), (10, 6, 5))
        self.config = PlannerConfig(max_iterations=500, step_m=0.7, neighbor_radius_m=1.8,
                                    goal_radius_m=1.3, goal_bias=0.15)

    def test_direct_and_coincident(self) -> None:
        world = World(self.workspace, (), 0.2)
        direct = plan_rrt_star(world, (1, 3, 1), (9, 3, 1), self.config, 11)
        self.assertEqual(direct.reason, "direct_path")
        self.assertEqual(direct.path_m, ((1, 3, 1), (9, 3, 1)))
        self.assertEqual(direct.path_length_m, 8)
        self.assertTrue(direct.path_valid)
        same = plan_rrt_star(world, (1, 3, 1), (1, 3, 1), self.config, 11)
        self.assertEqual(same.reason, "start_equals_goal")
        self.assertEqual(same.path_length_m, 0)
        self.assertEqual(len(same.path_m), 1)

    def test_detour_replay_goal_edge_cost_and_history(self) -> None:
        world = World(self.workspace, (Sphere((5, 3, 1), 1.1),), 0.2, 0.05)
        first = plan_rrt_star(world, (1, 3, 1), (9, 3, 1), self.config, 11)
        second = plan_rrt_star(world, (1, 3, 1), (9, 3, 1), self.config, 11)
        self.assertTrue(first.success, first.reason)
        self.assertEqual(first.path_m, second.path_m)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.best_cost_history_m, second.best_cost_history_m)
        self.assertTrue(first.path_valid)
        self.assertGreater(first.path_length_m, 8)
        self.assertAlmostEqual(first.path_length_m,
                               sum(math.dist(a, b) for a, b in zip(first.path_m[:-1], first.path_m[1:])))
        self.assertTrue(all(world.segment_free(a, b) for a, b in zip(first.path_m[:-1], first.path_m[1:])))
        self.assertEqual(first.path_m[-1], (9, 3, 1))
        costs = [x for x in first.best_cost_history_m if x is not None]
        self.assertTrue(all(a >= b - 1e-10 for a, b in zip(costs[:-1], costs[1:])))
        self.assertAlmostEqual(costs[-1], first.path_length_m)
        self.assertTrue(all(audit_result(world, (1, 3, 1), (9, 3, 1), first).values()))

    def test_barrier_failure_and_invalid_endpoints(self) -> None:
        barrier = World(self.workspace, (AABB((4, 0, 0), (6, 6, 5)),), 0.2)
        failed = plan_rrt_star(barrier, (1, 3, 1), (9, 3, 1),
                               PlannerConfig(max_iterations=100), 11)
        self.assertFalse(failed.success)
        self.assertEqual(failed.reason, "iteration_budget_exhausted")
        self.assertEqual(failed.path_m, ())
        self.assertIsNone(failed.path_length_m)
        self.assertTrue(all(audit_result(barrier, (1, 3, 1), (9, 3, 1), failed).values()))
        invalid_start = plan_rrt_star(barrier, (0.1, 3, 1), (9, 3, 1), self.config, 11)
        self.assertEqual(invalid_start.reason, "invalid_start")
        invalid_goal = plan_rrt_star(barrier, (1, 3, 1), (5, 3, 1), self.config, 11)
        self.assertEqual(invalid_goal.reason, "invalid_goal")

    def test_parameter_and_input_validation(self) -> None:
        for kwargs in ({"step_m": 0}, {"goal_bias": 1.1}, {"max_iterations": -1},
                       {"neighbor_radius_m": float("nan")}):
            with self.assertRaises(ValueError):
                PlannerConfig(**kwargs)
        with self.assertRaises(ValueError):
            plan_rrt_star(World(self.workspace, (), 0.2), (1, 2), (9, 3, 1), self.config, 1)

    def test_raw_geometry_returns_reason_coded_invalid_outcomes(self) -> None:
        def plan(low, high, obstacles=(), start=(1, 3, 1), goal=(9, 3, 1)):
            return plan_from_geometry(low, high, obstacles, start, goal, 0.2, 0.05, self.config, 11)
        self.assertEqual(plan((0, 0, 0), (0, 6, 5)).reason, "invalid_workspace")
        self.assertEqual(plan((0, 0, 0), (10, 6, 5), obstacles=(object(),)).reason, "invalid_geometry")
        self.assertEqual(plan((0, 0, 0), (10, 6, 5), start=(1, 2)).reason, "invalid_start_coordinates")
        self.assertEqual(plan((0, 0, 0), (10, 6, 5), goal=(1, 2)).reason, "invalid_goal_coordinates")

    def test_altitude_scenario_requires_climb_and_audits_tree(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "rrt_star.toml"
        config, seeds, scenarios, _ = load_benchmark(path)
        scenario = next(item for item in scenarios if item.name == "altitude")
        result = plan_rrt_star(scenario.world, scenario.start, scenario.goal, config, seeds[0])
        self.assertTrue(result.success)
        self.assertGreater(max(point[2] for point in result.path_m), 3.4)
        self.assertTrue(all(audit_result(scenario.world, scenario.start, scenario.goal, result).values()))

    def test_aggregate_retains_failures_without_zero_lengths(self) -> None:
        rows = [
            {"scenario": "case", "success": True, "path_length_m": 8.0,
             "first_solution_iteration": 3, "tree_nodes": 10, "runtime_s": 0.1,
             "attempted_rewires": 2, "accepted_rewires": 1, "path_valid": True, "audit_passed": True},
            {"scenario": "case", "success": False, "path_length_m": None,
             "first_solution_iteration": None, "tree_nodes": 20, "runtime_s": 0.2,
             "attempted_rewires": 3, "accepted_rewires": 2, "path_valid": False, "audit_passed": True},
        ]
        summary = aggregate(rows)[0]
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_path_length_m"], 8.0)
        self.assertEqual(summary["total_attempted_rewires"], 5)


if __name__ == "__main__":
    unittest.main()
