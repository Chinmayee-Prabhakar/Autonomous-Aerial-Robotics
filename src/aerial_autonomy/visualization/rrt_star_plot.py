"""Saved 3D planning, projection and convergence figures for M6."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from aerial_autonomy.planning.geometry import AABB, Sphere, World
from aerial_autonomy.planning.rrt_star import PlanningResult


def _box_faces(box: AABB) -> list[list[tuple[float, float, float]]]:
    lo, hi = np.asarray(box.minimum), np.asarray(box.maximum)
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return [[corners[i] for i in face] for face in
            ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))]


def save_plan_figure(world: World, start: tuple[float, ...], goal: tuple[float, ...],
                     result: PlanningResult, scenario: str, path: str | Path,
                     show_tree: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13, 6.2))
    ax = figure.add_subplot(121, projection="3d")
    side = figure.add_subplot(222)
    top = figure.add_subplot(224)
    lo, hi = np.asarray(world.workspace.minimum), np.asarray(world.workspace.maximum)
    for y in (lo[1], hi[1]):
        for z in (lo[2], hi[2]):
            ax.plot([lo[0], hi[0]], [y, y], [z, z], color="0.65", lw=0.8)
    for x in (lo[0], hi[0]):
        for z in (lo[2], hi[2]):
            ax.plot([x, x], [lo[1], hi[1]], [z, z], color="0.65", lw=0.8)
        for y in (lo[1], hi[1]):
            ax.plot([x, x], [y, y], [lo[2], hi[2]], color="0.65", lw=0.8)
    for obstacle in world.obstacles:
        if isinstance(obstacle, Sphere):
            u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:13j]
            center = np.asarray(obstacle.center)
            radius = obstacle.radius_m + world.envelope_m
            xs = center[0] + radius * np.cos(u) * np.sin(v)
            ys = center[1] + radius * np.sin(u) * np.sin(v)
            zs = center[2] + radius * np.cos(v)
            ax.plot_surface(xs, ys, zs, color="tomato", alpha=0.22, linewidth=0)
            side.add_patch(plt.Circle((center[0], center[2]), radius, color="tomato", alpha=0.25))
            top.add_patch(plt.Circle((center[0], center[1]), radius, color="tomato", alpha=0.25))
        else:
            faces = _box_faces(obstacle)
            ax.add_collection3d(Poly3DCollection(faces, facecolors="darkorange", alpha=0.32,
                                                  edgecolors="saddlebrown", linewidths=0.4))
            from matplotlib.patches import Rectangle
            side.add_patch(Rectangle((obstacle.minimum[0], obstacle.minimum[2]),
                                     obstacle.maximum[0] - obstacle.minimum[0],
                                     obstacle.maximum[2] - obstacle.minimum[2], color="darkorange", alpha=0.3))
            top.add_patch(Rectangle((obstacle.minimum[0], obstacle.minimum[1]),
                                    obstacle.maximum[0] - obstacle.minimum[0],
                                    obstacle.maximum[1] - obstacle.minimum[1], color="darkorange", alpha=0.3))
    if show_tree:
        for node in result.nodes[1::max(1, len(result.nodes) // 180)]:
            if node.parent is not None:
                parent = result.nodes[node.parent].position_m
                ax.plot([parent[0], node.position_m[0]], [parent[1], node.position_m[1]],
                        [parent[2], node.position_m[2]], color="steelblue", alpha=0.16, lw=0.55)
    for point, color, label in ((start, "seagreen", "start"), (goal, "purple", "goal")):
        ax.scatter(*point, color=color, s=60, label=label, depthshade=False)
        side.scatter(point[0], point[2], color=color, s=35)
        top.scatter(point[0], point[1], color=color, s=35)
    if result.success:
        points = np.asarray(result.path_m)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="navy", lw=3, label="RRT* path")
        side.plot(points[:, 0], points[:, 2], color="navy", lw=2)
        top.plot(points[:, 0], points[:, 1], color="navy", lw=2)
        caption = f"{result.path_length_m:.2f} m | min clearance {result.min_clearance_m:.4f} m"
    else:
        caption = f"No complete path: {result.reason}"
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel="East x (m)", ylabel="North y (m)", zlabel="Up z (m)")
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=25, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    for axis, ordinate, label in ((side, 2, "Up z (m)"), (top, 1, "North y (m)")):
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[ordinate], hi[ordinate])
        axis.set_xlabel("East x (m)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    side.set_title("Side projection")
    top.set_title("Top projection")
    figure.suptitle(f"{scenario} | seed {result.seed} | {caption}\n"
                   f"Spherical vehicle envelope + safety margin = {world.envelope_m:.2f} m")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def save_convergence_figure(result: PlanningResult, scenario: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    values = [np.nan if value is None else value for value in result.best_cost_history_m]
    axis.plot(range(len(values)), values, color="navy", lw=2)
    axis.set(xlabel="Planning iteration", ylabel="Best complete path length (m)",
             title=f"{scenario} | seed {result.seed} | RRT* best feasible cost")
    if not result.success:
        axis.text(0.5, 0.5, "No complete path found", transform=axis.transAxes,
                  ha="center", va="center", color="firebrick")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
