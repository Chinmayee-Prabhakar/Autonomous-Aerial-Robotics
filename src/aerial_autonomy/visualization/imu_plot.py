"""A2 comparisons of true and measured body-frame IMU signals."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from aerial_autonomy.evaluation.imu import ImuScenarioTrace


def save_imu_plot(trace: "ImuScenarioTrace", path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for component, axis_name in enumerate(("x", "y", "z")):
        gyro_ax = axes[0, component]
        gyro_ax.plot(trace.time_s, trace.body_rate_rad_s[:, component], label="truth", lw=1.5)
        gyro_ax.plot(trace.measurement_time_s, trace.measured_gyro_rad_s[:, component],
                     label="measured", lw=0.7, alpha=0.8)
        gyro_ax.set_ylabel(f"gyro {axis_name} (rad/s)")
        gyro_ax.grid(alpha=0.3)

        accel_ax = axes[1, component]
        accel_ax.plot(trace.time_s, trace.specific_force_body_m_s2[:, component], label="truth", lw=1.5)
        accel_ax.plot(trace.measurement_time_s, trace.measured_accel_m_s2[:, component],
                      label="measured", lw=0.7, alpha=0.8)
        accel_ax.set_ylabel(f"accel {axis_name} (m/s²)")
        accel_ax.set_xlabel("time (s)")
        accel_ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best")
    figure.suptitle(f"IMU truth vs measurement: {trace.name.replace('_', ' ')} | body FLU")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
