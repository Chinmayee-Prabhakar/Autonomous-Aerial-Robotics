"""Covariance and innovation figures for the attitude-only UKF."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_covariance_plot(path: Path, time_s: np.ndarray, covariance_rad2: np.ndarray,
                         heading_std_deg: np.ndarray, minimum_eigenvalue_rad2: np.ndarray,
                         innovation_norm: np.ndarray, covariance_condition: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    std_deg = np.rad2deg(np.sqrt(np.maximum(np.diagonal(covariance_rad2, axis1=1, axis2=2), 0)))
    for index, label in enumerate(("body-local x", "body-local y", "body-local z")):
        axes[0, 0].plot(time_s, std_deg[:, index], label=label)
    axes[0, 0].plot(time_s, heading_std_deg, label="local gravity-axis heading", linestyle="--")
    axes[0, 0].set_ylabel("attitude standard deviation (deg)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(time_s, minimum_eigenvalue_rad2)
    axes[0, 1].set_ylabel("minimum covariance eigenvalue (rad²)")
    axes[1, 0].plot(time_s, innovation_norm)
    axes[1, 0].set_ylabel("normalized-force innovation norm")
    axes[1, 1].semilogy(time_s, covariance_condition)
    axes[1, 1].set_ylabel("covariance condition number")
    for axis in axes[1]:
        axis.set_xlabel("time (s)")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    figure.suptitle("UKF local uncertainty and measurement diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_sensitivity_plot(path: Path, aggregate: list[dict]) -> None:
    """Plot all three methods' mean geodesic errors with across-seed spread."""
    cases = list(dict.fromkeys(item["case"] for item in aggregate))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for axis, scenario in zip(axes, ("hover", "lateral_waypoint")):
        rows = [next(row for row in aggregate if row["case"] == case
                     and row["scenario"] == scenario) for case in cases]
        position = np.arange(len(cases))
        for offset, key, label in ((-0.25, "ukf", "UKF"),
                                   (0.0, "madgwick", "Madgwick"),
                                   (0.25, "gyro_only", "gyro only")):
            axis.bar(position + offset,
                     [row[f"{key}_geodesic_rmse_deg_mean"] for row in rows], 0.24,
                     yerr=[row[f"{key}_geodesic_rmse_deg_std"] for row in rows],
                     capsize=2, label=label)
        axis.set_xticks(position, cases, rotation=25)
        axis.set_title(scenario.replace("_", " "))
        axis.set_ylabel("geodesic RMSE (deg)")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Shared-input estimator sensitivity: mean ± SD over three seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
