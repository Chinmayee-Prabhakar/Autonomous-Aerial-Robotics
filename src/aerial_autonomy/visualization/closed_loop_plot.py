"""Static research plots for controlled attitude-feedback flight experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerial_autonomy.simulation.rotation import quaternion_to_euler


COLORS = {"truth": "#1f77b4", "madgwick": "#2ca02c", "ukf": "#d62728"}


def _finish(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_scenario(directory: Path, scenario, trials: dict) -> None:
    """Use true state for flight plots and separate estimated attitude traces."""
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_index, axis in enumerate(axes):
        first = next(iter(trials.values())).evaluation
        axis.plot(first["timestamp_s"], first["desired_position_m"][:, axis_index],
                  "k--", label="desired")
        for mode, trial in trials.items():
            data = trial.evaluation
            axis.plot(data["timestamp_s"], data["position_m"][:, axis_index],
                      color=COLORS[mode], label=mode)
        axis.set_ylabel(("east", "north", "up")[axis_index] + " (m)")
        axis.grid(alpha=.25)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(scenario.name + ": desired and actual position")
    _finish(figure, directory / "position.png")

    figure, axes = plt.subplots(3, 3, figsize=(13, 9), sharex="col")
    for column, (mode, trial) in enumerate(trials.items()):
        data, estimate = trial.evaluation, trial.estimator
        true_euler = np.rad2deg(np.array([quaternion_to_euler(q)
                                          for q in data["true_quaternion_bw"]]))
        estimated = estimate["euler_deg"].copy()
        if len(true_euler):
            true_euler[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(true_euler[:, 2])))
            estimated[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(estimated[:, 2])))
            true_euler[:, 2] -= true_euler[0, 2]
            estimated[:, 2] -= estimated[0, 2]
        for row, label in enumerate(("roll", "pitch", "relative yaw")):
            axes[row, column].plot(data["timestamp_s"], true_euler[:, row],
                                   "k--", label="true")
            axes[row, column].plot(data["timestamp_s"], estimated[:, row],
                                   color=COLORS[mode], label="estimated")
            axes[row, column].set_ylabel(label + " (deg)")
            axes[row, column].grid(alpha=.25)
        axes[0, column].set_title(mode)
        axes[0, column].legend(fontsize=8)
        axes[-1, column].set_xlabel("time (s)")
    figure.suptitle(scenario.name + ": true versus controller feedback attitude")
    _finish(figure, directory / "attitude.png")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for mode, trial in trials.items():
        data, estimate = trial.evaluation, trial.estimator
        axes[0].plot(data["timestamp_s"], data["attitude_estimation_geodesic_error_deg"],
                     color=COLORS[mode], label=mode)
        axes[1].plot(data["timestamp_s"], data["position_error_m"],
                     color=COLORS[mode], label=mode)
        if trial.metrics["failure_time_s"] is not None:
            axes[1].axvline(trial.metrics["failure_time_s"], color=COLORS[mode],
                            linestyle=":", alpha=.7)
    axes[0].set_ylabel("attitude geodesic error (deg)")
    axes[1].set_ylabel("position tracking error (m)")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle(scenario.name + ": estimation and tracking errors")
    _finish(figure, directory / "errors.png")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for mode, trial in trials.items():
        data = trial.evaluation
        time_s = data["timestamp_s"]
        axes[0].plot(time_s, data["applied_thrust_n"], color=COLORS[mode], label=mode)
        axes[1].plot(time_s, np.linalg.norm(data["applied_torque_nm"], axis=1),
                     color=COLORS[mode], label=mode)
        for axis, mask, values in ((axes[0], data["thrust_saturated"], data["applied_thrust_n"]),
                                   (axes[1], data["torque_saturated"],
                                    np.linalg.norm(data["applied_torque_nm"], axis=1))):
            axis.scatter(time_s[mask], values[mask], s=5, color=COLORS[mode], alpha=.5)
    axes[0].set_ylabel("applied thrust (N)")
    axes[1].set_ylabel("applied torque norm (N m)")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle(scenario.name + ": control effort; dots mark saturation")
    _finish(figure, directory / "control.png")

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    for mode, trial in trials.items():
        position = trial.evaluation["position_m"]
        axis.plot(position[:, 0], position[:, 1], position[:, 2],
                  color=COLORS[mode], label=mode)
    target = next(iter(trials.values())).evaluation["desired_position_m"]
    axis.plot(target[:, 0], target[:, 1], target[:, 2], "k--", label="desired")
    axis.set(xlabel="east (m)", ylabel="north (m)", zlabel="up (m)")
    axis.legend()
    figure.suptitle(scenario.name + ": 3D flight path")
    _finish(figure, directory / "trajectory_3d.png")


def plot_aggregate(path: Path, aggregate: list[dict]) -> None:
    """Full-horizon mean and population SD, with failure counts visible."""
    scenarios = sorted({row["scenario"] for row in aggregate})
    conditions = ["ideal", "noise_only", "bias_only", "nominal", "stressed"]
    figure, axes = plt.subplots(len(scenarios), 3, figsize=(16, 4 * len(scenarios)))
    axes = np.atleast_2d(axes)
    for row_index, scenario in enumerate(scenarios):
        for column, field, unit in ((0, "position_rmse_m", "m"),
                                    (1, "attitude_geodesic_rmse_deg", "deg")):
            axis = axes[row_index, column]
            for mode in COLORS:
                selected = [next(item for item in aggregate if item["scenario"] == scenario
                                 and item["condition"] == condition and item["mode"] == mode)
                            for condition in conditions]
                x = np.arange(len(conditions)) + (list(COLORS).index(mode) - 1) * .22
                means = [item[field + "_full_horizon_mean"] for item in selected]
                stds = [item[field + "_full_horizon_std"] for item in selected]
                axis.bar(x, [np.nan if value is None else value for value in means],
                         width=.21, yerr=[0 if value is None else value for value in stds],
                         color=COLORS[mode], alpha=.8, label=mode)
            axis.set_xticks(range(len(conditions)), conditions, rotation=20)
            axis.set_ylabel(f"{scenario}: {field.replace('_', ' ')} ({unit})")
            axis.grid(axis="y", alpha=.25)
            if row_index == 0:
                axis.legend()
        axis = axes[row_index, 2]
        for mode in COLORS:
            selected = [next(item for item in aggregate if item["scenario"] == scenario
                             and item["condition"] == condition and item["mode"] == mode)
                        for condition in conditions]
            x = np.arange(len(conditions)) + (list(COLORS).index(mode) - 1) * .22
            axis.bar(x, [item["failures"] for item in selected], width=.21,
                     color=COLORS[mode], alpha=.8, label=mode)
        axis.set_xticks(range(len(conditions)), conditions, rotation=20)
        axis.set_ylabel(f"{scenario}: failures per {aggregate[0]['trials']} seeds")
        axis.set_ylim(0, max(item["trials"] for item in aggregate) + .5)
        axis.grid(axis="y", alpha=.25)
        if row_index == 0:
            axis.legend()
    figure.suptitle("M5 paired sensor-condition sensitivity; full-horizon runs only")
    _finish(figure, path)


def plot_error_relationship(path: Path, trials: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for mode in COLORS:
        selected = [row for row in trials if row["mode"] == mode and row["completed_horizon"]]
        axis.scatter([row["attitude_geodesic_rmse_deg"] for row in selected],
                     [row["position_rmse_m"] for row in selected],
                     color=COLORS[mode], label=mode, alpha=.65, s=24)
    axis.set(xlabel="attitude geodesic RMSE (deg)", ylabel="position RMSE (m)")
    axis.grid(alpha=.25)
    axis.legend()
    figure.suptitle("M5 full-horizon paired trials; association is descriptive")
    _finish(figure, path)
