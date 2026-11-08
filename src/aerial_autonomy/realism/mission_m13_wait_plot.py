"""Present the separate M13 wait-then-cross development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def plot_wait(trial_dir: Path, output: Path) -> None:
    trial_dir, output = Path(trial_dir), Path(output)
    summary = json.loads((trial_dir / "summary.json").read_text())
    choices = json.loads((trial_dir / "trial" / "timing_choices.json").read_text())
    captures = [json.loads(line) for line in (trial_dir / "captures.jsonl").read_text().splitlines()]
    by_id = {item["image_id"]: item for item in captures}
    delta = next(item for item in summary["metrics"]["crossing_events"]
                 if item["window_id"] == "delta" and item["classification"] == "opening")
    ids = ["rgb_00095", "rgb_00104", "rgb_00109", "rgb_00120"]
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[2.2, 1.1])
    for i, image_id in enumerate(ids):
        record = by_id[image_id]
        path = trial_dir / "rendered" / "causal" / f"{image_id}.png"
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(Image.open(path))
        ax.set_title(f"Actual RGB {image_id}\nt={record['capture_time_s']:.2f} s")
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    times = [item["time_s"] for item in choices]
    accepted = [1 if item["valid"] else 0 for item in choices]
    ax.step(times, accepted, where="post", color="tab:blue", label="Predicted safe interval")
    ax.scatter(times, accepted, c=["tab:green" if ok else "tab:red" for ok in accepted], zorder=3)
    ax.axvline(66.79, color="tab:green", linestyle="--", label="Committed after 2.8 s wait")
    ax.axvline(delta["plane_crossing_time_s"], color="black", linestyle=":",
               label=f"Physical delta plane crossing {delta['plane_crossing_time_s']:.2f} s")
    ax.set(xlabel="Simulation time (s)", ylabel="Safe interval", ylim=(-.15, 1.3))
    ax.set_yticks([0, 1], ["rejected", "available"])
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("M13 development phase 4.0: observed wait, then delta crossing\n"
                 "RGB panels are saved onboard observations; crossing time is from physical trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    manifest_path = output.parent / "visual_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if output.name not in manifest["files"]:
            manifest["files"].append(output.name)
        manifest["waiting_development_trial"] = str(trial_dir).replace("\\", "/")
        manifest["waiting_development_physical_crossing_s"] = delta["plane_crossing_time_s"]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    plot_wait(Path(args.trial_dir), Path(args.output))
