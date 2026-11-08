"""Seeded geometric 3D RRT* with parent optimization and cost propagation."""

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .geometry import COLLISION_TOLERANCE_M, Obstacle, Workspace, World, vector3


COST_TOLERANCE_M = 1e-10


@dataclass(frozen=True)
class PlannerConfig:
    max_iterations: int = 600
    step_m: float = 0.7
    neighbor_radius_m: float = 1.6
    goal_radius_m: float = 1.1
    goal_bias: float = 0.12

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        for name in ("step_m", "neighbor_radius_m", "goal_radius_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.goal_bias) or not 0 <= self.goal_bias <= 1:
            raise ValueError("goal_bias must be between zero and one")


@dataclass(frozen=True)
class TreeNode:
    position_m: tuple[float, float, float]
    parent: int | None
    cost_m: float


class Tree:
    """Rooted tree with collision-tested changes and descendant cost updates."""

    def __init__(self, start: Sequence[float]):
        self.positions: list[tuple[float, float, float]] = [vector3(start, "start")]
        self.parents: list[int | None] = [None]
        self.costs: list[float] = [0.0]
        self.children: list[set[int]] = [set()]

    def add(self, point: Sequence[float], parent: int) -> int:
        if not 0 <= parent < len(self.positions):
            raise ValueError("parent index outside tree")
        position = vector3(point, "new node")
        distance = math.dist(position, self.positions[parent])
        if distance <= COST_TOLERANCE_M:
            raise ValueError("duplicate/zero-length tree edge")
        index = len(self.positions)
        self.positions.append(position)
        self.parents.append(parent)
        self.costs.append(self.costs[parent] + distance)
        self.children.append(set())
        self.children[parent].add(index)
        return index

    def is_ancestor(self, possible_ancestor: int, node: int) -> bool:
        current: int | None = node
        while current is not None:
            if current == possible_ancestor:
                return True
            current = self.parents[current]
        return False

    def rewire(self, node: int, new_parent: int, world: World) -> bool:
        """Return true only for a collision-free strict improvement, never a cycle."""
        if node == 0 or node == new_parent or not (0 <= node < len(self.positions)) or not (0 <= new_parent < len(self.positions)):
            return False
        if self.is_ancestor(node, new_parent):
            return False
        new_cost = self.costs[new_parent] + math.dist(self.positions[new_parent], self.positions[node])
        if new_cost >= self.costs[node] - COST_TOLERANCE_M:
            return False
        if not world.segment_free(self.positions[new_parent], self.positions[node]):
            return False
        old_parent = self.parents[node]
        assert old_parent is not None
        self.children[old_parent].remove(node)
        self.children[new_parent].add(node)
        self.parents[node] = new_parent
        delta = new_cost - self.costs[node]
        stack = [node]
        while stack:
            current = stack.pop()
            self.costs[current] += delta
            stack.extend(self.children[current])
        return True

    def path_indices(self, node: int) -> tuple[int, ...]:
        indices = []
        while node is not None:
            indices.append(node)
            node = self.parents[node]
        return tuple(reversed(indices))

    def snapshot(self) -> tuple[TreeNode, ...]:
        return tuple(TreeNode(point, parent, cost)
                     for point, parent, cost in zip(self.positions, self.parents, self.costs))

    def validate(self, world: World) -> bool:
        if self.parents[0] is not None or abs(self.costs[0]) > COST_TOLERANCE_M:
            return False
        for index in range(1, len(self.positions)):
            parent = self.parents[index]
            if parent is None or self.is_ancestor(index, parent):
                return False
            if index not in self.children[parent]:
                return False
            expected = self.costs[parent] + math.dist(self.positions[parent], self.positions[index])
            if not math.isclose(self.costs[index], expected, rel_tol=1e-10, abs_tol=1e-9):
                return False
            if not world.segment_free(self.positions[parent], self.positions[index]):
                return False
        return True


@dataclass(frozen=True)
class PlanningResult:
    success: bool
    reason: str
    seed: int
    iterations: int
    first_solution_iteration: int | None
    path_m: tuple[tuple[float, float, float], ...]
    path_length_m: float | None
    best_cost_history_m: tuple[float | None, ...]
    nodes: tuple[TreeNode, ...]
    attempted_rewires: int
    accepted_rewires: int
    collision_checks: int
    path_valid: bool
    min_clearance_m: float | None
    runtime_s: float


def audit_result(world: World, start: Sequence[float], goal: Sequence[float],
                 result: PlanningResult) -> dict[str, bool]:
    """Recompute tree, path and history invariants from an immutable snapshot.

    This pass uses no planning-time edge decisions, tree child sets or caches.
    It includes failed trials, whose stored trees must also remain consistent.
    """
    nodes = result.nodes
    root_ok = bool(nodes and nodes[0].parent is None and
                   nodes[0].position_m == vector3(start, "start") and
                   abs(nodes[0].cost_m) <= COST_TOLERANCE_M)
    cycles_absent = root_ok
    costs_valid = root_ok
    tree_edges_valid = root_ok
    for index in range(1, len(nodes)):
        node = nodes[index]
        parent = node.parent
        if parent is None or not 0 <= parent < len(nodes):
            cycles_absent = costs_valid = tree_edges_valid = False
            continue
        seen = {index}
        cursor: int | None = parent
        while cursor is not None:
            if cursor in seen or not 0 <= cursor < len(nodes):
                cycles_absent = False
                break
            seen.add(cursor)
            cursor = nodes[cursor].parent
        expected = nodes[parent].cost_m + math.dist(nodes[parent].position_m, node.position_m)
        costs_valid &= math.isclose(node.cost_m, expected, rel_tol=1e-10, abs_tol=1e-9)
        tree_edges_valid &= world.segment_clearance_m(nodes[parent].position_m, node.position_m) > COLLISION_TOLERANCE_M
    finite_history = [value for value in result.best_cost_history_m if value is not None]
    history_length_ok = (len(result.best_cost_history_m) == result.iterations + 1 or
                         (result.reason in ("invalid_start", "invalid_goal") and not result.best_cost_history_m))
    history_monotone = (history_length_ok and
                        all(a >= b - 1e-9 for a, b in zip(finite_history[:-1], finite_history[1:])))
    if result.success:
        path = result.path_m
        endpoints_valid = bool(path and path[0] == vector3(start, "start") and
                               path[-1] == vector3(goal, "goal"))
        edges = list(zip(path[:-1], path[1:]))
        path_edges_valid = endpoints_valid and all(
            world.segment_clearance_m(a, b) > COLLISION_TOLERANCE_M for a, b in edges)
        length = sum(math.dist(a, b) for a, b in edges)
        path_cost_valid = (result.path_length_m is not None and
                           math.isclose(length, result.path_length_m, rel_tol=1e-10, abs_tol=1e-9) and
                           bool(finite_history) and
                           math.isclose(finite_history[-1], length, rel_tol=1e-10, abs_tol=1e-9))
        goal_edge_valid = (len(path) == 1 and vector3(start, "start") == vector3(goal, "goal")) or bool(edges and
                          world.segment_clearance_m(*edges[-1]) > COLLISION_TOLERANCE_M)
    else:
        endpoints_valid = not result.path_m
        path_edges_valid = not result.path_m
        path_cost_valid = result.path_length_m is None and not finite_history
        goal_edge_valid = not result.path_m
    return {"root_valid": root_ok, "cycles_absent": cycles_absent,
            "tree_costs_valid": costs_valid, "tree_edges_valid": tree_edges_valid,
            "path_endpoints_valid": endpoints_valid, "path_edges_valid": path_edges_valid,
            "path_cost_valid": path_cost_valid, "goal_edge_valid": goal_edge_valid,
            "best_history_monotone": history_monotone}


def plan_from_geometry(workspace_min: Sequence[float], workspace_max: Sequence[float],
                       obstacles: Sequence[Obstacle], start: Sequence[float], goal: Sequence[float],
                       vehicle_radius_m: float, clearance_m: float,
                       config: PlannerConfig, seed: int) -> PlanningResult:
    """Validate raw geometry and return stable reason codes for invalid maps.

    Constructed ``World`` objects also work directly with ``plan_rrt_star``.
    """
    began = perf_counter()

    def invalid(reason: str) -> PlanningResult:
        return PlanningResult(False, reason, seed, 0, None, (), None, (), (),
                              0, 0, 0, False, None, perf_counter() - began)

    try:
        workspace = Workspace(workspace_min, workspace_max)
    except (TypeError, ValueError):
        return invalid("invalid_workspace")
    try:
        world = World(workspace, tuple(obstacles), vehicle_radius_m, clearance_m)
    except (TypeError, ValueError):
        return invalid("invalid_geometry")
    try:
        vector3(start, "start")
    except (TypeError, ValueError):
        return invalid("invalid_start_coordinates")
    try:
        vector3(goal, "goal")
    except (TypeError, ValueError):
        return invalid("invalid_goal_coordinates")
    return plan_rrt_star(world, start, goal, config, seed)


def _select_parent(tree: Tree, point: tuple[float, float, float], candidates: Sequence[int],
                   edge_free: Callable[[Sequence[float], Sequence[float]], bool]) -> int | None:
    """Try candidates in ascending full root-to-point cost, with stable ties."""
    ranked = sorted(candidates, key=lambda i: (tree.costs[i] + math.dist(tree.positions[i], point), i))
    for index in ranked:
        if math.dist(tree.positions[index], point) > COST_TOLERANCE_M and edge_free(tree.positions[index], point):
            return index
    return None


def plan_rrt_star(world: World, start: Sequence[float], goal: Sequence[float],
                  config: PlannerConfig, seed: int) -> PlanningResult:
    """Return ordered geometric waypoints or a reason-coded finite-budget failure.

    The fixed 3D neighbor radius is a finite-run engineering choice. It does not
    assert asymptotic optimality. The goal is connected only by a checked edge.
    """
    began = perf_counter()
    if not isinstance(config, PlannerConfig) or not isinstance(world, World):
        raise ValueError("world and config must be validated geometry/config objects")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    start_p, goal_p = vector3(start, "start"), vector3(goal, "goal")
    tree = Tree(start_p)
    checks = 0

    def edge_free(a: Sequence[float], b: Sequence[float]) -> bool:
        nonlocal checks
        checks += 1
        return world.segment_free(a, b)

    def result(reason: str, iterations: int, first: int | None,
               history: list[float | None], goal_parent: int | None) -> PlanningResult:
        if goal_parent is None:
            path: tuple[tuple[float, float, float], ...] = ()
            length = None
            valid = False
            clearance = None
        else:
            path = tuple(tree.positions[i] for i in tree.path_indices(goal_parent))
            if path[-1] != goal_p:
                path += (goal_p,)
            length = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
            # Independent final pass; it does not use the planner's checked-edge cache.
            clearances = [world.segment_clearance_m(a, b) for a, b in zip(path[:-1], path[1:])]
            if not clearances:
                clearances = [world.segment_clearance_m(path[0], path[0])]
            clearance = min(clearances)
            valid = all(c > COLLISION_TOLERANCE_M for c in clearances)
            if not valid or not tree.validate(world):
                raise RuntimeError("final path/tree failed independent validation")
        return PlanningResult(goal_parent is not None, reason, seed, iterations, first,
                              path, length, tuple(history), tree.snapshot(), attempts,
                              accepted, checks, valid, clearance, perf_counter() - began)

    attempts = accepted = 0
    if not edge_free(start_p, start_p):
        return result("invalid_start", 0, None, [], None)
    if not edge_free(goal_p, goal_p):
        return result("invalid_goal", 0, None, [], None)
    if start_p == goal_p:
        return result("start_equals_goal", 0, 0, [0.0], 0)
    if edge_free(start_p, goal_p):
        return result("direct_path", 0, 0, [math.dist(start_p, goal_p)], 0)

    rng = np.random.default_rng(seed)
    lower = np.asarray(world.workspace.minimum) + world.envelope_m + 2 * COLLISION_TOLERANCE_M
    upper = np.asarray(world.workspace.maximum) - world.envelope_m - 2 * COLLISION_TOLERANCE_M
    goal_connections: set[int] = set()
    history: list[float | None] = [None]
    first: int | None = None
    best_cost = math.inf
    best_parent: int | None = None

    for iteration in range(1, config.max_iterations + 1):
        sample = np.asarray(goal_p) if rng.random() < config.goal_bias else rng.uniform(lower, upper)
        positions = np.asarray(tree.positions)
        distances = np.linalg.norm(positions - sample, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= COST_TOLERANCE_M:
            history.append(None if best_parent is None else best_cost)
            continue
        direction = sample - positions[nearest]
        point = vector3(positions[nearest] + direction * min(1.0, config.step_m / distances[nearest]), "steered point")
        if not edge_free(point, point):
            history.append(None if best_parent is None else best_cost)
            continue
        near_distances = np.linalg.norm(positions - point, axis=1)
        nearby = [i for i, distance in enumerate(near_distances) if distance <= config.neighbor_radius_m]
        if nearest not in nearby:
            nearby.append(nearest)
        parent = _select_parent(tree, point, nearby, edge_free)
        if parent is None:
            history.append(None if best_parent is None else best_cost)
            continue
        new_index = tree.add(point, parent)
        for index in nearby:
            if index == parent or index == 0 or tree.is_ancestor(index, new_index):
                continue
            candidate_cost = tree.costs[new_index] + math.dist(point, tree.positions[index])
            if candidate_cost < tree.costs[index] - COST_TOLERANCE_M:
                attempts += 1
                checks += 1
                if tree.rewire(index, new_index, world):
                    accepted += 1
        if math.dist(point, goal_p) <= config.goal_radius_m and edge_free(point, goal_p):
            goal_connections.add(new_index)
            if first is None:
                first = iteration
        if goal_connections:
            candidate_parent = min(goal_connections,
                                   key=lambda i: (tree.costs[i] + math.dist(tree.positions[i], goal_p), i))
            candidate_cost = tree.costs[candidate_parent] + math.dist(tree.positions[candidate_parent], goal_p)
            if candidate_cost < best_cost - COST_TOLERANCE_M:
                best_cost, best_parent = candidate_cost, candidate_parent
        history.append(None if best_parent is None else best_cost)
    return result("success" if best_parent is not None else "iteration_budget_exhausted",
                  config.max_iterations, first, history, best_parent)
