"""Static M3/M4 attitude and error comparisons."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_attitude_plot(path: Path, title: str, time_s: np.ndarray,
                       true_euler_deg: np.ndarray, estimated_euler_deg: np.ndarray,
                       geodesic_error_deg: np.ndarray, quaternion_norm_error: np.ndarray,
                       accel_norm_m_s2: np.ndarray, gyro_norm_rad_s: np.ndarray,
                       comparators: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                       primary_label: str = "Madgwick") -> None:
    """Show attitude and error versus time for identical-IMU comparisons."""
    comparators = comparators or {}
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    labels = ("roll", "pitch", "relative yaw")
    for index, ax in enumerate((axes[0, 0], axes[0, 1], axes[1, 0])):
        truth = true_euler_deg[:, index].copy()
        estimate = estimated_euler_deg[:, index].copy()
        if index == 2:
            truth -= truth[0]
            estimate -= estimate[0]
        ax.plot(time_s, truth, label="truth", linewidth=1.5)
        ax.plot(time_s, estimate, label=primary_label, linewidth=1.0)
        for label, (other_euler, _) in comparators.items():
            other = other_euler[:, index].copy()
            if index == 2:
                other -= other[0]
            ax.plot(time_s, other, label=label.replace("_", " "), linewidth=0.9, alpha=0.8)
        ax.set_ylabel(f"{labels[index]} (deg)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[1, 1].plot(time_s, geodesic_error_deg, label=primary_label)
    for label, (_, other_error) in comparators.items():
        axes[1, 1].plot(time_s, other_error, label=label.replace("_", " "),
                        linewidth=0.9, alpha=0.8)
    axes[1, 1].legend()
    axes[1, 1].set_ylabel("orientation error (deg)")
    accel_axis = axes[2, 0]
    gyro_axis = accel_axis.twinx()
    accel_line = accel_axis.plot(time_s, accel_norm_m_s2, color="tab:blue", label="|accelerometer|")
    gyro_line = gyro_axis.plot(time_s, gyro_norm_rad_s, color="tab:orange", label="|gyro|")
    accel_axis.set_ylabel("accelerometer norm (m/s²)")
    gyro_axis.set_ylabel("gyro norm (rad/s)")
    accel_axis.legend(accel_line + gyro_line, ["|accelerometer|", "|gyro|"], loc="best")
    axes[2, 1].plot(time_s, quaternion_norm_error)
    axes[2, 1].set_ylabel("|quaternion norm − 1|")
    for ax in axes[2]:
        ax.set_xlabel("time (s)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_sensitivity_plot(path: Path, aggregate: list[dict]) -> None:
    """Show trial means and standard deviations for fixed sweep settings."""
    cases = list(dict.fromkeys(item["case"] for item in aggregate))
    scenarios = ("hover", "lateral_waypoint")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, scenario in zip(axes, scenarios):
        rows = [next(item for item in aggregate if item["case"] == case
                     and item["scenario"] == scenario) for case in cases]
        position = np.arange(len(cases))
        madgwick_mean = [row["geodesic_rmse_deg_mean"] for row in rows]
        madgwick_std = [row["geodesic_rmse_deg_std"] for row in rows]
        gyro_mean = [row["gyro_only_geodesic_rmse_deg_mean"] for row in rows]
        gyro_std = [row["gyro_only_geodesic_rmse_deg_std"] for row in rows]
        axis.bar(position - 0.19, madgwick_mean, 0.38, yerr=madgwick_std,
                 capsize=3, label="Madgwick")
        axis.bar(position + 0.19, gyro_mean, 0.38, yerr=gyro_std,
                 capsize=3, label="gyro only")
        axis.set_xticks(position, cases, rotation=25)
        axis.set_title(scenario.replace("_", " "))
        axis.set_ylabel("geodesic RMSE (deg)")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    figure.suptitle("Madgwick bias/noise sensitivity: mean ± SD across seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
