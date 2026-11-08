"""Matplotlib diagnostics for the hover experiment."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from aerial_autonomy.evaluation.hover import HoverTrace


def save_hover_plot(trace: "HoverTrace", path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for axis, component, label in zip(axes[0], range(3), ("x", "y", "z")):
        axis.plot(trace.time_s, trace.desired_position_m[:, component], "--", label="desired")
        axis.plot(trace.time_s, trace.position_m[:, component], label="actual")
        axis.set_ylabel(f"{label} (m)")
        axis.grid(alpha=0.3)
    actual_deg = np.rad2deg(trace.euler_rad)
    desired_deg = np.rad2deg(trace.desired_euler_rad)
    for axis, component, label in zip(axes[1], range(3), ("roll", "pitch", "yaw")):
        axis.plot(trace.time_s, desired_deg[:, component], "--", label="desired")
        axis.plot(trace.time_s, actual_deg[:, component], label="actual")
        axis.set_ylabel(f"{label} (deg)")
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.3)
    axes[0, 0].legend(loc="best")
    figure.suptitle("Deterministic takeoff and hover | ENU world, FLU body")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

