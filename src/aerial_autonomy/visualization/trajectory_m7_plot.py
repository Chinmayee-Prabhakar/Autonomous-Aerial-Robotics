"""M7 research plots of commanded and actual rigid-body flight."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from aerial_autonomy.planning.geometry import AABB, Sphere


def _draw_inflated_obstacles(ax, world) -> None:
    """Draw the same 0.25 m flight-envelope geometry in plots and GIFs."""
    for obstacle in world.obstacles:
        if isinstance(obstacle, Sphere):
            u, v = np.meshgrid(np.linspace(0, 2*np.pi, 24), np.linspace(0, np.pi, 12))
            radius = obstacle.radius_m + world.envelope_m
            cx, cy, cz = obstacle.center
            ax.plot_surface(cx+radius*np.cos(u)*np.sin(v), cy+radius*np.sin(u)*np.sin(v),
                            cz+radius*np.cos(v), color="tab:red", alpha=.15)
        elif isinstance(obstacle, AABB):
            lo = np.asarray(obstacle.minimum)-world.envelope_m
            hi = np.asarray(obstacle.maximum)+world.envelope_m
            for z in (lo[2], hi[2]):
                xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
                ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
                ax.plot(xs, ys, [z]*5, color="tab:red", alpha=.6)
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    ax.plot([x,x], [y,y], [lo[2],hi[2]], color="tab:red", alpha=.6)


def save_flight_animation(world, trajectory, trace, folder: Path) -> None:
    """Render a visual replay of accepted saved data; numerical checks remain separate."""
    folder.mkdir(parents=True, exist_ok=True)
    waypoints = np.asarray(trajectory.waypoints_m)
    reference = trace["reference_position_m"]
    actual = trace["actual_position_m"]
    time = trace["time_s"]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_inflated_obstacles(ax, world)
    ax.plot([], [], [], color="tab:red", linewidth=2,
            label=f"obstacles inflated by {world.envelope_m:.2f} m")
    ax.plot(*waypoints.T, "k.--", linewidth=1, markersize=5, label="M6 waypoints")
    ax.plot(*reference.T, color="tab:blue", alpha=.7, linewidth=2, label="M7 reference")
    ax.scatter(*waypoints[0], marker="^", color="green", s=65, label="start")
    ax.scatter(*waypoints[-1], marker="*", color="purple", s=85, label="goal")
    trail, = ax.plot([], [], [], color="tab:orange", linewidth=2, label="actual flight")
    vehicle, = ax.plot([], [], [], marker="o", linestyle="None", markersize=9,
                       markerfacecolor="gold", markeredgecolor="black",
                       label="current vehicle center")
    ax.set(xlim=(world.workspace.minimum[0], world.workspace.maximum[0]),
           ylim=(world.workspace.minimum[1], world.workspace.maximum[1]),
           zlim=(world.workspace.minimum[2], world.workspace.maximum[2]),
           xlabel="East (m)", ylabel="North (m)", zlabel="Up (m)")
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    frames = np.unique(np.linspace(0, len(time)-1, 48, dtype=int))

    def update(frame):
        portion = actual[:frame+1]
        trail.set_data(portion[:,0], portion[:,1])
        trail.set_3d_properties(portion[:,2])
        vehicle.set_data([portion[-1,0]], [portion[-1,1]])
        vehicle.set_3d_properties([portion[-1,2]])
        ax.set_title(f"Actual flight, t={time[frame]:.1f} s")
        return trail, vehicle

    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    animation.save(folder/"flight.gif", writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)


def save_trial_figures(world, trajectory, trace, limits, vehicle, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    waypoints = np.asarray(trajectory.waypoints_m)
    reference = trace["reference_position_m"]
    actual = trace["actual_position_m"]
    time = trace["time_s"]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_inflated_obstacles(ax, world)
    ax.plot(*waypoints.T, "o--", color="black", label="M6 waypoints")
    ax.plot(*reference.T, color="tab:blue", linewidth=2, label="M7 timed reference")
    ax.plot(*actual.T, color="tab:orange", linewidth=2, label="M1 actual flight")
    ax.scatter(*waypoints[0], marker="^", color="green", s=75, label="start")
    ax.scatter(*waypoints[-1], marker="*", color="purple", s=95, label="goal")
    ax.set(xlabel="East (m)", ylabel="North (m)", zlabel="Up (m)",
           title="Planned waypoints, timed reference, actual flight")
    ax.set_xlim(world.workspace.minimum[0], world.workspace.maximum[0])
    ax.set_ylim(world.workspace.minimum[1], world.workspace.maximum[1])
    ax.set_zlim(world.workspace.minimum[2], world.workspace.maximum[2])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(folder/"tracking_3d.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label, ax in zip(range(3), ("East", "North", "Up"), axes):
        ax.plot(time, reference[:,axis], label="reference", color="tab:blue")
        ax.plot(time, actual[:,axis], label="actual", color="tab:orange")
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(folder/"position.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, trace["position_error_m"], color="tab:orange")
    ax.set(xlabel="Time (s)", ylabel="Position error (m)", title="Euclidean reference tracking error")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(folder/"tracking_error.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    specific_force = (vehicle.mass_kg*(trace["reference_acceleration_m_s2"]
                      + np.array([0,0,vehicle.gravity_m_s2]))
                      + vehicle.linear_drag_n_s_m*trace["reference_velocity_m_s"])
    series = [(np.linalg.norm(trace["reference_velocity_m_s"],axis=1),
               trace["actual_speed_m_s"], limits.max_speed_m_s, "Speed (m/s)"),
              (np.linalg.norm(trace["reference_acceleration_m_s2"],axis=1),
               trace["actual_acceleration_norm_m_s2"], limits.max_acceleration_m_s2, "Acceleration (m/s²)"),
              (np.linalg.norm(specific_force,axis=1),
               trace["thrust_n"], limits.max_thrust_n, "Thrust (N)"),
              (np.rad2deg(np.arctan2(np.linalg.norm(specific_force[:,:2],axis=1),
                                    specific_force[:,2])),
               trace["actual_tilt_deg"], limits.max_tilt_deg, "Tilt (deg)")]
    for ax,(reference_values,actual_values,ceiling,label) in zip(axes,series):
        ax.plot(time, reference_values, color="tab:blue", label="reference")
        ax.plot(time, actual_values, color="tab:orange", label="actual")
        ax.axhline(ceiling, color="tab:red", linestyle=":", label="reference limit")
        ax.set_ylabel(label)
        ax.grid(True, alpha=.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(folder/"feasibility.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, trace["reference_clearance_m"], label="reference point", color="tab:blue")
    ax.plot(time, trace["actual_clearance_m"], label="actual point", color="tab:orange")
    ax.axhline(0, color="tab:red", linestyle="--", label=f"envelope contact ({world.envelope_m:.2f} m)")
    ax.set(xlabel="Time (s)", ylabel="Signed clearance beyond envelope (m)",
           title="Obstacle and workspace clearance")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(folder/"clearance.png", dpi=150)
    plt.close(fig)

    if folder.parent.name in ("detour", "altitude"):
        save_flight_animation(world, trajectory, trace, folder)
