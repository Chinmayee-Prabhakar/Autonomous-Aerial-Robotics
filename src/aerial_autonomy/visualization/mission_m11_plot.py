"""M11 figures rendered exclusively from saved mission traces and observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw

from aerial_autonomy.perception.window_m10_arm import RotatingArm
from aerial_autonomy.perception.window_m10_arm import ArmShape
from aerial_autonomy.simulation.rotation import quaternion_to_matrix
from aerial_autonomy.visualization.window_m10_plot import _quad


def _load(folder):
    folder = Path(folder)
    trace = np.load(folder / "physical_trace.npz")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    transitions = json.loads((folder / "transitions.json").read_text(encoding="utf-8"))
    return trace, metrics, transitions


def _base_scene(ax, world, arm, time_s, closeup=False):
    for wall in world.walls:
        window = wall.window
        panels = []
        for box in wall.local_boxes():
            lo, hi = np.asarray(box.minimum), np.asarray(box.maximum)
            local = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0],
                              [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
            panels.append(window.center_world_m +
                          local @ window.rotation_window_to_world.T)
        if closeup and window.model.window_id != "delta":
            continue
        ax.add_collection3d(Poly3DCollection(panels, facecolor="#8e9caa",
            edgecolor="#a2acb4", alpha=.12))
        corners = window.world_corners_m()
        loop = np.vstack((corners, corners[0]))
        ax.plot(*loop.T, color="#e8795d" if window.model.window_id == "delta"
                else "#537eb0", linewidth=2.2)
        ax.text(*(window.center_world_m + np.array([0., 0., 1.3])),
                window.model.window_id, fontsize=8)
    a, b = arm.endpoints_world(float(time_s))
    ax.plot(*np.array([a, b]).T, color="#e3a21a", linewidth=7,
            solid_capstyle="round")
    ax.scatter(*a, color="#e629d8", s=30)
    if closeup:
        ax.set_xlim(19.8, 22.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(.45, 2.8)
        ax.set_box_aspect((2.7, 3.0, 2.35))
    else:
        ax.set_xlim(-.7, 23.0); ax.set_ylim(-6.0, 6.0); ax.set_zlim(0., 3.3)
        ax.set_box_aspect((23.7, 12., 3.3))
    ax.set_xlabel("east x (m)"); ax.set_ylabel("north y (m)"); ax.set_zlabel("up z (m)")
    ax.view_init(elev=23, azim=-66)


def _save_gif(images, path, duration_ms=180):
    if not images:
        raise ValueError("no saved frames to animate")
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0, optimize=True)
    for item in images:
        item.close()


def _front_arm(ax, shape, angle_rad):
    pivot, tip = shape.endpoints_local(angle_rad)
    centerline = tip[:2]-pivot[:2]
    across = np.array([-centerline[1], centerline[0]]) / shape.length_m
    across *= shape.arm_radius_m
    corners = np.array([pivot[:2]-across, tip[:2]-across,
                        tip[:2]+across, pivot[:2]+across])
    ax.add_patch(Polygon(corners, closed=True, facecolor="#eab32f",
                         edgecolor="#704315", linewidth=1.2, zorder=5))
    for endpoint in (pivot, tip):
        ax.add_patch(Circle(endpoint[:2], shape.arm_radius_m,
                            facecolor="#eab32f", edgecolor="#704315", zorder=5))
    ax.add_patch(Circle(pivot[:2], .065, facecolor="#d528ce",
                        edgecolor="white", linewidth=.6, zorder=7))


def _front_window(ax, window, corridor):
    width, height = window.model.width_m, window.model.height_m
    ax.add_patch(Rectangle((-width/2, -height/2), width, height,
                           fill=False, edgecolor="#ce654c", linewidth=5))
    ax.axvline(corridor, color="#267eaa", linestyle="--", linewidth=1.6,
               label=f"crossing corridor {corridor:+.2f} m")
    ax.set(xlim=(-1.55, 1.55), ylim=(-1.25, 1.25),
           xlabel="window right (m)", ylabel="window up (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.15)


def _front_on_animation(trace, metrics, arm, corridor, indices, path):
    """Physical window-plane projection of saved plant states and arm clock."""
    t, position, attitude = (trace["time_s"], trace["position_m"],
                              trace["quaternion_bw"])
    window = arm.window
    frames = []
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=100)
    for index in indices:
        ax.clear()
        _front_window(ax, window, corridor)
        _front_arm(ax, arm.shape, arm.angle_at(float(t[index])))
        local = window.opening_coordinates(position[index])
        ax.add_patch(Circle(local[:2], metrics.get("modeled_envelope_m", .25),
                            facecolor="#1a8caf", edgecolor="#08445a", alpha=.28,
                            linewidth=1.5, zorder=6))
        ax.plot(local[0], local[1], marker="o", color="#073449", markersize=6, zorder=7)
        forward_world = quaternion_to_matrix(attitude[index]) @ np.array([1., 0., 0.])
        forward_local = window.rotation_window_to_world.T @ forward_world
        ax.arrow(local[0], local[1], .22*forward_local[0], .22*forward_local[1],
                 color="#073449", width=.012, head_width=.065, zorder=8)
        ax.set_title(f"Centered delta arm | actual t={t[index]:.2f} s | "
                     f"plane distance={local[2]:+.2f} m", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    _save_gif(frames, path, duration_ms=160)
    return [float(t[i]) for i in indices]


def _geometry_comparison(window, original_shape, original_corridor,
                         revised_shape, revised_corridor, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=130, sharex=True, sharey=True)
    for ax, shape, corridor, title in (
        (axes[0], original_shape, original_corridor, "Original submitted M11"),
        (axes[1], revised_shape, revised_corridor, "Centered enlarged M11")):
        _front_window(ax, window, corridor)
        ax.add_patch(Circle((shape.pivot_right_m, shape.pivot_up_m),
                            shape.length_m+shape.arm_radius_m, fill=False,
                            edgecolor="#d7a16a", linestyle=":", linewidth=1.2))
        _front_arm(ax, shape, .9)
        ax.add_patch(Circle((corridor, 0.), .25, facecolor="#2086aa",
                            edgecolor="#0c4663", alpha=.22))
        margin = min(window.model.width_m/2-abs(shape.pivot_right_m)-
                     shape.length_m-shape.arm_radius_m,
                     window.model.height_m/2-abs(shape.pivot_up_m)-
                     shape.length_m-shape.arm_radius_m)
        ax.set_title(f"{title}\npivot {shape.pivot_right_m:+.2f} m | "
                     f"arm {shape.length_m:.2f} × r{shape.arm_radius_m:.2f} m | "
                     f"frame margin {margin:.3f} m", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Same 2.70 × 2.05 m opening and 0.25 m vehicle envelope; arm at 0.9 rad")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _render_3d(trace, metrics, world, arm, indices, path, closeup=False,
               landing=False, reason=None):
    t = trace["time_s"]
    p = trace["position_m"]
    q = trace["quaternion_bw"]
    phase = trace["phase"]
    delta_time = next((e["plane_crossing_time_s"] for e in metrics["crossing_events"]
                       if e["window_id"] == "delta" and e["classification"] == "opening"),
                      math.inf)
    bypass_time = next((e["plane_crossing_time_s"] for e in metrics["crossing_events"]
                        if e["classification"] == "external_plane_bypass"), math.inf)
    frames = []
    fig = plt.figure(figsize=(9.2, 6.0), dpi=85)
    for index in indices:
        fig.clear()
        ax = fig.add_subplot(projection="3d")
        if landing:
            ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0); ax.set_zlim(0., 1.6)
            ax.set_box_aspect((2., 2., 1.6))
            ax.set_xlabel("east x (m)"); ax.set_ylabel("north y (m)")
            ax.set_zlabel("up z (m)"); ax.view_init(elev=22, azim=-65)
            theta = np.linspace(0, 2*math.pi, 80)
            ax.plot(.2*np.cos(theta), .2*np.sin(theta), np.zeros(80),
                    color="#158474", linewidth=2)
        else:
            _base_scene(ax, world, arm, t[index], closeup)
        if closeup:
            near = (t[:index+1] >= t[index]-8.)
            ax.plot(*p[:index+1][near].T, color="#167ea4", linewidth=2)
        elif landing:
            near = t[:index+1] >= t[index]-5.
            ax.plot(*p[:index+1][near].T, color="#167ea4", linewidth=2)
        else:
            for mask, color, label in (
                ((t[:index+1] <= delta_time), "#198bb4", "outbound"),
                (((t[:index+1] > delta_time) & (t[:index+1] <= bypass_time)),
                 "#a363ce", "far side"),
                ((t[:index+1] > bypass_time), "#e19336", "return")):
                if np.any(mask):
                    ax.plot(*p[:index+1][mask].T, color=color, linewidth=2,
                            label=label)
        _quad(ax, {"position_m": p[index], "quaternion_bw": q[index]})
        title = f"M11 actual plant | t={t[index]:.1f} s | {phase[index]}"
        if reason:
            title += f" | safe abort: {reason}"
        ax.set_title(title, fontsize=10)
        if not closeup and not landing:
            ax.legend(loc="upper left", fontsize=7)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    _save_gif(frames, path)
    return [float(t[i]) for i in indices]


def _overview(trace, metrics, world, arm, path):
    t, p = trace["time_s"], trace["position_m"]
    fig = plt.figure(figsize=(12, 7), dpi=130)
    ax = fig.add_subplot(projection="3d")
    _base_scene(ax, world, arm, metrics["duration_s"])
    delta = next(e["plane_crossing_time_s"] for e in metrics["crossing_events"]
                 if e["window_id"] == "delta" and e["classification"] == "opening")
    bypass = next(e["plane_crossing_time_s"] for e in metrics["crossing_events"]
                  if e["classification"] == "external_plane_bypass")
    for mask, color, label in (((t <= delta), "#1283ab", "outbound"),
                               (((t > delta) & (t <= bypass)), "#9b5fc1", "far-side bypass"),
                               ((t > bypass), "#e39631", "reverse return and landing")):
        ax.plot(*p[mask].T, color=color, linewidth=2.5, label=label)
        selected = np.flatnonzero(mask)
        for fraction in (.25, .55, .82):
            if len(selected) > 30:
                j = selected[int(fraction*(len(selected)-2))]
                direction = p[j+20]-p[j]
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    ax.quiver(*p[j], *(direction/norm*.75), color=color,
                              arrow_length_ratio=.35, linewidth=1.8)
    for event in metrics["crossing_events"]:
        if event["classification"] != "opening":
            continue
        point = p[int(np.searchsorted(t, event["plane_crossing_time_s"]))]
        ax.scatter(*point, s=30, color="#0c3647")
    ax.scatter(*p[0], color="#3ac979", s=70, label="start / landing")
    ax.set_title("M11 complete physical route: seven openings and exterior delta bypass")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _timeline(metrics, path):
    events = [e for e in metrics["crossing_events"] if e["classification"] == "opening"]
    fig, ax = plt.subplots(figsize=(11, 4), dpi=130)
    for i, event in enumerate(events):
        ax.plot([event["entry_time_s"], event["exit_time_s"]], [i, i],
                linewidth=8, solid_capstyle="round",
                color="#cc7842" if event["direction"] == "return" else "#2183ab")
        ax.scatter(event["plane_crossing_time_s"], i, color="#172a38", s=25)
    ax.set_yticks(range(len(events)), [f"{e['window_id']} {e['direction']}" for e in events])
    ax.invert_yaxis(); ax.set_xlabel("simulation time (s)")
    ax.set_title("Actual full-envelope opening traversal intervals")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _series(trace, metrics, transitions, folder, acceptance):
    t = trace["time_s"]
    position = trace["position_m"]
    ref = trace["reference_position_m"]
    speed = np.linalg.norm(trace["velocity_m_s"], axis=1)
    error = np.linalg.norm(position-ref, axis=1)
    acceleration = np.gradient(trace["velocity_m_s"], t, axis=0, edge_order=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    q = trace["quaternion_bw"]
    tilt_deg = np.degrees(np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1)))
    applied_thrust = trace["thrust_n"].astype(float).copy()
    applied_thrust[0] = np.nan  # The initial state precedes the first plant command.
    fig, axes = plt.subplots(7, 1, figsize=(13, 13), sharex=True, dpi=120)
    for axis, values, title, limit in zip(axes,
        (error, trace["static_clearance_m"], trace["dynamic_clearance_m"],
         speed, acceleration_norm, applied_thrust, tilt_deg),
        ("position tracking error (m)", "static extra clearance (m)",
         "arm extra clearance (m)", "actual speed (m/s)",
         "actual acceleration (m/s²)", "applied thrust (N)", "actual tilt (deg)"),
        (None, 0, 0, acceptance["maximum_speed_m_s"],
         acceptance["maximum_acceleration_m_s2"],
         acceptance["maximum_thrust_n"], acceptance["maximum_tilt_deg"])):
        axis.plot(t, values, linewidth=1)
        if limit is not None:
            axis.axhline(limit, color="#b54738", linestyle="--", linewidth=1)
        axis.set_ylabel(title); axis.grid(alpha=.2)
    axes[5].axhline(acceptance["minimum_thrust_n"], color="#b54738",
                    linestyle="--", linewidth=1)
    axes[-1].set_xlabel("simulation time (s)")
    fig.tight_layout(); fig.savefig(folder / "safety_tracking.png"); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, dpi=120)
    axes[0].plot(t[1:], trace["thrust_n"][1:]); axes[0].axhline(16, color="red", linestyle="--")
    axes[0].axhline(5, color="red", linestyle="--")
    axes[0].set_ylabel("collective thrust (N)")
    torque = trace["torque_nm"]
    for j in range(3):
        axes[j+1].plot(t[1:], torque[1:, j]); axes[j+1].set_ylabel(f"body torque {j} (N m)")
        bound = (.4, .4, .2)[j]
        axes[j+1].axhline(bound, color="red", linestyle="--", linewidth=.8)
        axes[j+1].axhline(-bound, color="red", linestyle="--", linewidth=.8)
    for axis in axes:
        for transition in transitions:
            if transition["state"] in ("TRAVERSE_STATIC", "TRAVERSE_DELTA", "BYPASS_DELTA", "LAND"):
                axis.axvline(transition["time_s"], color="#a67851", alpha=.35, linewidth=.8)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("simulation time (s)")
    saturation = (trace["thrust_n"][1:] >= 20-1e-8) | np.any(
        abs(torque[1:]) >= np.array([.4, .4, .2])-1e-8, axis=1)
    if np.any(saturation):
        for axis in axes:
            for when in t[1:][saturation]:
                axis.axvline(when, color="#d53132", alpha=.25, linewidth=.7)
    fig.suptitle(f"Controller-generated wrench applied to the plant | "
                 f"saturated steps: {metrics['control_saturation_step_count']}")
    fig.tight_layout(); fig.savefig(folder / "control_inputs.png"); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, dpi=120)
    for j, label in enumerate(("east", "north", "up")):
        axes[j].plot(t, position[:, j], label="actual")
        axes[j].plot(t, ref[:, j], linestyle="--", label="reference")
        axes[j].set_ylabel(f"{label} (m)"); axes[j].grid(alpha=.2)
    axes[0].legend()
    axes[3].plot(t, acceleration_norm, label="actual acceleration")
    axes[3].axhline(1.8, color="red", linestyle="--", label="limit")
    axes[3].set_ylabel("acceleration (m/s²)"); axes[3].set_xlabel("simulation time (s)")
    axes[3].legend(); axes[3].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "reference_actual.png"); plt.close(fig)


def _camera_animation(folder, path, reason=None):
    source = np.load(folder / "camera_frames.npz")
    images = []
    for i, rgb in enumerate(source["image_rgb"]):
        canvas = Image.fromarray(rgb).resize((640, 480))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 639, 45), fill=(8, 17, 28))
        draw.text((8, 5), f"M11 actual camera t={source['time_s'][i]:.1f}s | "
                  f"{source['phase'][i]} | {source['target_id'][i]}", fill="white")
        draw.text((8, 26), "green: detected window corners | yellow: detected arm",
                  fill="#d2e5ee")
        if reason:
            draw.rectangle((0, 430, 639, 479), fill=(45, 16, 18))
            draw.text((8, 434), f"Safe abort: {reason}", fill="#ffb2a1")
        corners = source["corners_uv"][i]
        if np.all(np.isfinite(corners)):
            for u, v in corners:
                draw.ellipse((2*u-4, 2*v-4, 2*u+4, 2*v+4),
                             outline="#36f79c", width=2)
        angle = source["arm_angle_rad"][i]
        if np.isfinite(angle):
            draw.text((8, 453), f"image arm angle {math.degrees(angle):+.1f} deg",
                      fill="#ffe177")
        images.append(canvas)
    _save_gif(images, path, duration_ms=175)
    return [float(x) for x in source["time_s"]]


def render_saved_trial(folder, world, arm, acceptance, corridor,
                       original_shape, original_corridor):
    """Never synthesize states: every drawn pose comes from physical_trace.npz."""
    folder = Path(folder)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    trace, metrics, transitions = _load(folder)
    arm = RotatingArm(arm.window, arm.shape,
        metrics["condition"]["initial_phase_rad"],
        0.0 if metrics["condition"]["stationary_arm"] else arm.angular_velocity_rad_s)
    _overview(trace, metrics, world, arm, figures / "route_overview.png")
    _timeline(metrics, figures / "crossing_timeline.png")
    _series(trace, metrics, transitions, figures, acceptance)
    t = trace["time_s"]
    event_times = [x["plane_crossing_time_s"] for x in metrics["crossing_events"]]
    indices = sorted(set(np.linspace(0, len(t)-1, 65, dtype=int).tolist() +
        [int(np.searchsorted(t, x)) for event in event_times for x in
         (max(0, event-.6), event, min(t[-1], event+.6))]))
    mission_times = _render_3d(trace, metrics, world, arm, indices,
                               figures / "mission_3d.gif")
    delta = next(e["plane_crossing_time_s"] for e in metrics["crossing_events"]
                 if e["window_id"] == "delta" and e["classification"] == "opening")
    close_indices = sorted(set(np.searchsorted(t, np.linspace(delta-3., delta+3., 31))))
    close_times = _render_3d(trace, metrics, world, arm, close_indices,
                             figures / "delta_closeup.gif", closeup=True)
    front_times = _front_on_animation(trace, metrics, arm, corridor, close_indices,
                                      figures / "delta_front_on.gif")
    _geometry_comparison(arm.window, original_shape, original_corridor,
                         arm.shape, corridor, figures / "arm_geometry_comparison.png")
    landing_indices = sorted(set(np.searchsorted(t,
        np.linspace(max(0, t[-1]-4), t[-1], 22))))
    landing_times = _render_3d(trace, metrics, world, arm, landing_indices,
                               figures / "landing_closeup.gif", landing=True)
    camera_times = _camera_animation(folder, figures / "camera_mission.gif")
    def nearest(times, target):
        return int(np.argmin(abs(np.asarray(times)-target)))
    timeline = {"mission_frame_times_s": mission_times,
                "camera_frame_times_s": camera_times,
                "delta_closeup_frame_times_s": close_times,
                "delta_front_on_frame_times_s": front_times,
                "landing_frame_times_s": landing_times,
                "events": [{"window_id": e["window_id"],
                    "direction": e.get("direction"), "classification": e["classification"],
                    "time_s": e["plane_crossing_time_s"],
                    "mission_frame": nearest(mission_times, e["plane_crossing_time_s"]),
                    "camera_frame": (nearest(camera_times, e["plane_crossing_time_s"])
                        if abs(camera_times[nearest(camera_times, e["plane_crossing_time_s"])]-
                               e["plane_crossing_time_s"]) <= 1.5 else None),
                    "nearest_camera_observation_time_s": camera_times[
                        nearest(camera_times, e["plane_crossing_time_s"])]}
                    for e in metrics["crossing_events"]]}
    (figures / "animation_timeline.json").write_text(json.dumps(timeline, indent=2),
                                                       encoding="utf-8")
    return figures


def render_failure_saved_trial(folder, world, arm):
    folder = Path(folder)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    trace, metrics, transitions = _load(folder)
    if metrics["success"] or metrics["failure_reason"] is None:
        raise ValueError("a saved failure trace is required")
    arm = RotatingArm(arm.window, arm.shape,
        metrics["condition"]["initial_phase_rad"],
        0.0 if metrics["condition"]["stationary_arm"] else arm.angular_velocity_rad_s)
    t = trace["time_s"]
    indices = sorted(set(np.linspace(0, len(t)-1, 45, dtype=int).tolist() +
        np.searchsorted(t, np.linspace(max(0, t[-1]-8), t[-1], 18)).tolist()))
    _render_3d(trace, metrics, world, arm, indices,
               figures / "safe_abort_3d.gif", reason=metrics["failure_reason"])
    if (folder / "camera_frames.npz").exists():
        _camera_animation(folder, figures / "safe_abort_camera.gif",
                          reason=metrics["failure_reason"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, dpi=120)
    axes[0].plot(t, trace["position_m"][:, 0], label="east")
    axes[0].plot(t, trace["position_m"][:, 1], label="north")
    axes[0].legend(); axes[0].set_ylabel("position (m)")
    axes[1].plot(t, trace["static_clearance_m"], label="static")
    axes[1].plot(t, trace["dynamic_clearance_m"], label="moving arm")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_ylabel("extra clearance (m)"); axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.suptitle(f"Controlled hold and abort: {metrics['failure_reason']}")
    fig.tight_layout(); fig.savefig(figures / "safe_abort_evidence.png"); plt.close(fig)
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mission_m11.toml")
    parser.add_argument("--output-dir", default="outputs/mission_m11")
    parser.add_argument("--original-config", default="configs/mission_m11_original.toml")
    args = parser.parse_args(argv)
    from aerial_autonomy.evaluation.mission_m11 import load_m11_config
    raw, _, _, _, world, arm, _, _ = load_m11_config(args.config)
    root = Path(args.output_dir)
    hero = root / f"trial_{raw['experiments']['hero_seed']}"
    with Path(args.original_config).open("rb") as source:
        original = tomllib.load(source)
    prior = original["arm"]
    original_shape = ArmShape(prior["pivot_right_m"], prior["pivot_up_m"],
                              prior["length_m"], prior["arm_radius_m"])
    print(render_saved_trial(hero, world, arm, raw["acceptance"],
        raw["guidance"]["delta_corridor_right_m"], original_shape,
        original["guidance"]["delta_corridor_right_m"]))
    print(render_failure_saved_trial(root / "failure_blocked_arm", world, arm))


if __name__ == "__main__":
    main()
