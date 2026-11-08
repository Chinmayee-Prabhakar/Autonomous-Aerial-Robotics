"""Render and label the actual saved M13 causal flight for visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.realism.bridge import camera_blender_pose
from aerial_autonomy.simulation.dynamics import VehicleState


def _caption(image: Image.Image, lines: list[str]) -> Image.Image:
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    height = 17 * len(lines) + 8
    draw.rectangle((0, 0, frame.width, height), fill=(14, 20, 28))
    for n, line in enumerate(lines):
        draw.text((8, 4 + n * 17), line, fill=(240, 245, 247))
    return frame


def _gif(frames: list[Image.Image], path: Path, delay_ms: int = 150):
    if not frames:
        raise ValueError("cannot write an empty visual sequence")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=delay_ms, loop=0, optimize=False)


def _plot(trace, metrics, output: Path):
    t = trace["time_s"]
    p = trace["position_m"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    for axis, name in enumerate("xyz"):
        axes[0].plot(t, p[:, axis], label=name)
    axes[0].set_ylabel("position (m)"); axes[0].legend(ncol=3)
    axes[1].plot(t, np.linalg.norm(p-trace["reference_position_m"], axis=1),
                 label="tracking error")
    axes[1].set_ylabel("tracking (m)")
    axes[2].plot(t, trace["thrust_n"], label="collective thrust", color="C0")
    torque_axis = axes[2].twinx()
    for axis, name in enumerate("xyz"):
        torque_axis.plot(t, trace["torque_nm"][:, axis],
                         label=f"torque {name}", color=f"C{axis+1}")
    axes[2].set_ylabel("thrust (N)")
    torque_axis.set_ylabel("body torque (N m)")
    lines = axes[2].lines + torque_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], ncol=4)
    axes[3].plot(t, trace["static_clearance_m"], label="wall/frame")
    axes[3].plot(t, trace["dynamic_clearance_m"], label="moving arm")
    axes[3].axhline(0, color="black", linewidth=.8)
    axes[3].set_ylabel("extra clearance (m)")
    axes[3].set_xlabel("M13 simulation time (s)"); axes[3].legend()
    for event in metrics["crossing_events"]:
        at = event.get("plane_crossing_time_s")
        if at is not None:
            for ax in axes:
                ax.axvline(at, color="gray", linewidth=.6, alpha=.45)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def present(trial_dir: Path, output_dir: Path, blender: Path,
            negative_dir: Path | None = None):
    trial_dir, output_dir = Path(trial_dir), Path(output_dir)
    summary = json.loads((trial_dir / "summary.json").read_text())
    metrics = summary["metrics"]
    if not metrics["success"]:
        raise ValueError("full-mission visual requires a successful new M13 flight")
    trace = np.load(trial_dir / "trial" / "physical_trace.npz")
    captures = [json.loads(line) for line in (trial_dir / "captures.jsonl").read_text().splitlines()]
    observations = json.loads((trial_dir / "trial" / "operational_observations.json").read_text())
    plan = json.loads((trial_dir / "static_scene_plan.json").read_text())
    raw, _, _, camera, _, arm, _, _ = load_m11_config("configs/mission_m11.toml")
    phase = metrics["condition"]["initial_phase_rad"]
    rate = 0.0 if metrics["condition"]["stationary_arm"] else arm.angular_velocity_rad_s
    t = trace["time_s"]
    events = metrics["crossing_events"]
    sample_times = np.linspace(float(t[0]), float(t[-1]), 56).tolist()
    sample_times += [float(event["plane_crossing_time_s"]) for event in events
                     if event.get("plane_crossing_time_s") is not None]
    indices = sorted({int(np.argmin(abs(t-value))) for value in sample_times})
    delta = next(event for event in events if event["window_id"] == "delta"
                 and event["classification"] == "opening")
    arm_times = np.linspace(delta["entry_time_s"] - 3,
                            delta["exit_time_s"] + 3, 24)
    arm_indices = sorted({int(np.argmin(abs(t-value))) for value in arm_times})
    jobs = []
    for group, selected in (("mission", indices), ("arm", arm_indices)):
        for rank, i in enumerate(selected):
            state = VehicleState(trace["position_m"][i], trace["velocity_m_s"][i],
                                 trace["quaternion_bw"][i], trace["body_rate_rad_s"][i])
            center, rotation = camera_blender_pose(state, camera)
            jobs.append({"job_id": f"{group}_{rank:03d}", "group": group,
                "time_s": float(t[i]), "trace_index": i,
                "position_world_m": state.position_m.tolist(),
                "quaternion_bw": state.quaternion_bw.tolist(),
                "camera_center_world_m": center.tolist(),
                "camera_rotation_blender_to_world": rotation.tolist(),
                "target_id": None, "profile": "bright", "material": "metal",
                "occlusion": False, "arm_initial_angle_rad": phase,
                "arm_angular_velocity_rad_s": rate})
    plan["jobs"] = jobs
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "presentation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    for group in ("mission", "arm"):
        command = [str(blender), "-b", str(Path("outputs/realism_m12/indoor_facility.blend")),
                   "--python", str(Path("tools/m12/blender_render.py")), "--",
                   "--plan", str(plan_path), "--output-dir", str(output_dir),
                   "--group", group, "--device", "CPU"]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (output_dir / f"{group}_blender.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Blender {group} presentation render failed")
    external = []
    for rank, i in enumerate(indices):
        image = Image.open(output_dir / "rendered" / "mission" / f"mission_{rank:03d}.png")
        external.append(_caption(image, [
            f"NEW M13 flight  t={float(t[i]):.2f}s  {str(trace['phase'][i])}",
            f"target={str(trace['target_id'][i]) or '-'}  frame={rank:03d}"]))
    _gif(external, output_dir / "full_indoor_m13.gif")
    bypass_time = next(event["plane_crossing_time_s"] for event in events
                       if event["classification"] == "external_plane_bypass")
    before_rank = min(range(len(indices)),
                      key=lambda n: abs(float(t[indices[n]])-(bypass_time-3.4)))
    after_rank = min(range(len(indices)),
                     key=lambda n: abs(float(t[indices[n]])-(bypass_time+3.5)))
    bypass_sheet = Image.new("RGB", (2*external[0].width, external[0].height),
                             (20, 20, 20))
    bypass_sheet.paste(external[before_rank], (0, 0))
    bypass_sheet.paste(external[after_rank], (external[0].width, 0))
    bypass_sheet.save(output_dir / "bypass_view.png")
    external[-1].save(output_dir / "landing_view.png")
    arm_frames = []
    for rank, i in enumerate(arm_indices):
        image = Image.open(output_dir / "rendered" / "arm" / f"arm_{rank:03d}.png")
        arm_frames.append(_caption(image, [
            f"NEW M13 delta  t={float(t[i]):.2f}s  {str(trace['phase'][i])}",
            f"arm phase from current trial; frame={rank:03d}"]))
    _gif(arm_frames, output_dir / "delta_tracking_crossing.gif")
    onboard = []
    selected_captures = list(range(0, len(captures), max(1, len(captures)//65)))
    for n in selected_captures:
        item, observation = captures[n], observations[n]
        image = Image.open(item["image_path"].replace("\\", "/"))
        onboard.append(_caption(image, [
            f"ACTUAL M13 RGB  t={item['capture_time_s']:.2f}s  {observation['phase']}",
            f"{observation['target_id']}  pose={observation['window_valid']}  "
            f"arm={observation['tracker_valid']}  image={item['image_id']}"]))
    _gif(onboard, output_dir / "onboard_rgb_used.gif")
    reverse = [n for n, obs in enumerate(observations) if
               obs["target_id"] in ("gamma", "beta", "alpha") and
               obs["time_s"] > bypass_time]
    # Retain true observed frames only; no interpolated onboard view is presented.
    picks = [reverse[n] for n in np.linspace(0, len(reverse)-1,
             min(12, len(reverse))).round().astype(int)] if reverse else []
    if picks:
        tiles = [_caption(Image.open(captures[n]["image_path"].replace("\\", "/")), [
            f"{observations[n]['target_id']}  t={captures[n]['capture_time_s']:.2f}s"])
            for n in picks]
        sheet = Image.new("RGB", (4*tiles[0].width, 3*tiles[0].height), (20, 20, 20))
        for n, tile in enumerate(tiles):
            sheet.paste(tile, ((n % 4)*tile.width, (n // 4)*tile.height))
        sheet.save(output_dir / "reverse_window_observations.png")
    _plot(trace, metrics, output_dir / "controls_tracking_clearance.png")
    if negative_dir is not None:
        negative_dir = Path(negative_dir)
        negative = json.loads((negative_dir / "summary.json").read_text())
        abort_captures = [json.loads(line) for line in
            (negative_dir / "captures.jsonl").read_text().splitlines()]
        if abort_captures:
            raw_image = Image.open(abort_captures[0]["image_path"].replace("\\", "/"))
            zeroed = Image.new("RGB", raw_image.size, (0, 0, 0))
            left = _caption(raw_image, ["Blender raw RGB", "target present in scene"])
            right = _caption(zeroed, ["Forced sensor loss", "no pose passed to guidance"])
            sheet = Image.new("RGB", (left.width + right.width,
                                      max(left.height, right.height)), (20, 20, 20))
            sheet.paste(left, (0, 0)); sheet.paste(right, (left.width, 0))
            draw = ImageDraw.Draw(sheet)
            draw.text((8, sheet.height-20),
                f"Result: {negative['metrics']['failure_reason']}; crossings=0",
                fill=(255, 240, 150))
            sheet.save(output_dir / "safe_abort_visual_loss.png")
    report = {"schema": "m13_new_flight_visuals_v1", "source_trial": str(trial_dir),
              "mission_frames": len(indices), "arm_frames": len(arm_indices),
              "onboard_observed_frames": len(onboard),
              "critical_events": events,
              "files": [p.name for p in output_dir.iterdir() if p.is_file()]}
    (output_dir / "visual_manifest.json").write_text(json.dumps(report, indent=2),
                                                       encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe")
    parser.add_argument("--negative-dir", default="outputs/mission_m13/negative_visual_loss")
    args = parser.parse_args()
    print(json.dumps(present(Path(args.trial_dir), Path(args.output_dir),
                             Path(args.blender), Path(args.negative_dir)), indent=2))


if __name__ == "__main__":
    main()
