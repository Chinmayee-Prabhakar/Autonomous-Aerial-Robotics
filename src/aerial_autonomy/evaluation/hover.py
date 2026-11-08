"""Deterministic takeoff to a fixed one-metre hover target."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from aerial_autonomy.config import ExperimentConfig, load_config
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.logging_setup import configure_logging
from aerial_autonomy.simulation.dynamics import VehicleState, advance_state
from aerial_autonomy.simulation.rotation import quaternion_to_euler
from aerial_autonomy.visualization.hover_plot import save_hover_plot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoverTrace:
    time_s: np.ndarray
    position_m: np.ndarray
    desired_position_m: np.ndarray
    euler_rad: np.ndarray
    desired_euler_rad: np.ndarray


def run_hover(config: ExperimentConfig) -> HoverTrace:
    """The seed fixes future stochastic additions; this baseline has no noise."""
    np.random.default_rng(config.demo.seed)
    steps = round(config.demo.duration_s / config.demo.step_s)
    time_s = np.arange(steps + 1) * config.demo.step_s
    position = np.zeros((steps + 1, 3))
    attitude = np.zeros((steps + 1, 3))
    desired_attitude = np.zeros((steps + 1, 3))
    desired_position = np.repeat(config.demo.target_position_m[None, :], steps + 1, axis=0)
    state = VehicleState.at_rest()
    controller = CascadedHoverController(config.vehicle, config.controller)
    target_yaw = np.deg2rad(config.demo.target_yaw_deg)

    for index in range(steps + 1):
        position[index] = state.position_m
        attitude[index] = quaternion_to_euler(state.quaternion_bw)
        result = controller.command(state, config.demo.target_position_m, target_yaw)
        desired_attitude[index] = result.desired_euler_rad
        if index < steps:
            state = advance_state(state, result.wrench, config.vehicle, config.demo.step_s)
    return HoverTrace(time_s, position, desired_position, attitude, desired_attitude)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hover.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hover.png"))
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    trace = run_hover(config)
    save_hover_plot(trace, args.output)
    final_error = np.linalg.norm(trace.position_m[-1] - config.demo.target_position_m)
    LOGGER.info("Final position: %s m", np.array2string(trace.position_m[-1], precision=4))
    LOGGER.info("Final position error: %.4f m", final_error)
    LOGGER.info("Plot saved: %s", args.output.resolve())


if __name__ == "__main__":
    main()

