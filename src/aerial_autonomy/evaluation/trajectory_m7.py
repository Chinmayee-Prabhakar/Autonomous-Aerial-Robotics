"""Reproducible M7 planning, reference generation, and M1 closed-loop flight."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tomllib

import numpy as np

from aerial_autonomy.config import load_config as load_vehicle_config
from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.rrt_star import load_benchmark
from aerial_autonomy.planning.geometry import COLLISION_TOLERANCE_M
from aerial_autonomy.planning.rrt_star import plan_rrt_star
from aerial_autonomy.simulation.dynamics import VehicleState, advance_state, world_acceleration
from aerial_autonomy.simulation.rotation import quaternion_to_euler, quaternion_to_matrix
from aerial_autonomy.trajectory.quintic import Trajectory, TrajectoryLimits, generate_trajectory
from aerial_autonomy.visualization.trajectory_m7_plot import save_trial_figures


def load_m7_config(path: str | Path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    planner, _, scenarios, _ = load_benchmark(raw["inputs"]["planner_config"])
    vehicle = load_vehicle_config(raw["inputs"]["vehicle_config"])
    limits = TrajectoryLimits(**raw["trajectory"])
    plan = raw["planning"]
    seeds = tuple(plan["seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("planning seeds must be unique nonnegative integers")
    if abs(float(plan["vehicle_radius_m"]) - scenarios[0].world.vehicle_radius_m) > 1e-12:
        raise ValueError("M7 must preserve the M6 vehicle radius")
    if float(plan["clearance_m"]) < scenarios[0].world.clearance_m:
        raise ValueError("M7 clearance cannot be smaller than M6")
    if limits.max_thrust_n > vehicle.controller.max_total_thrust_n or limits.max_tilt_deg > vehicle.controller.max_tilt_deg:
        raise ValueError("reference limits exceed controller limits")
    sim = raw["simulation"]
    if (any(not np.isfinite(float(sim[key])) or float(sim[key]) <= 0 for key in
            ("step_s", "settle_s", "max_position_error_m", "final_error_m", "settle_mean_error_m"))
            or not np.isfinite(float(sim["yaw_deg"])) or
            not np.isclose(float(sim["step_s"]), vehicle.demo.step_s)):
        raise ValueError("simulation limits/step invalid")
    names = tuple(raw["demonstration"]["scenarios"])
    if len(set(names)) != len(names) or any(name not in {x.name for x in scenarios} for name in names):
        raise ValueError("unknown or duplicate scenario")
    if raw["demonstration"]["visual_seed"] not in seeds:
        raise ValueError("visual seed must be predefined")
    return raw, planner, scenarios, vehicle, limits, seeds, names


def fly(trajectory: Trajectory, world, vehicle, simulation: dict,
        limits: TrajectoryLimits | None = None):
    """Truth feedback; plant and controller run once per 0.01 s interval."""
    dt = float(simulation["step_s"])
    horizon = trajectory.duration_s + float(simulation["settle_s"])
    steps = int(np.ceil(horizon/dt))
    controller = CascadedHoverController(vehicle.vehicle, vehicle.controller)
    state = VehicleState.at_rest()
    state.position_m = np.asarray(trajectory.waypoints_m[0], dtype=float)
    fields = {key: [] for key in ("time_s", "reference_position_m", "reference_velocity_m_s",
             "reference_acceleration_m_s2", "actual_position_m", "actual_velocity_m_s",
             "actual_acceleration_m_s2", "actual_euler_rad", "actual_body_rate_rad_s",
             "thrust_n", "torque_nm", "desired_euler_rad", "position_error_m",
             "actual_clearance_m", "reference_clearance_m", "actual_speed_m_s",
             "actual_acceleration_norm_m_s2", "actual_tilt_deg")}
    failure = None
    min_swept_clearance = float("inf")
    saturations = {"thrust": 0, "torque": 0, "tilt": 0, "body_rate": 0}
    yaw = float(np.deg2rad(simulation["yaw_deg"]))
    for index in range(steps+1):
        time_s = min(index*dt, horizon)
        reference = trajectory.evaluate(min(time_s, trajectory.duration_s))
        command = controller.command(state, reference.position_m, yaw,
                                     reference.velocity_m_s, reference.acceleration_m_s2)
        acceleration = world_acceleration(state, command.wrench, vehicle.vehicle)
        tilt = float(np.rad2deg(np.arccos(np.clip(quaternion_to_matrix(state.quaternion_bw)[2, 2], -1, 1))))
        clearance = world.segment_clearance_m(state.position_m, state.position_m)
        ref_clearance = world.segment_clearance_m(reference.position_m, reference.position_m)
        error = float(np.linalg.norm(state.position_m-reference.position_m))
        row = (time_s, reference.position_m, reference.velocity_m_s, reference.acceleration_m_s2,
               state.position_m, state.velocity_m_s, acceleration,
               quaternion_to_euler(state.quaternion_bw), state.body_rate_rad_s,
               command.wrench.thrust_n, command.wrench.torque_nm, command.desired_euler_rad,
               error, clearance, ref_clearance, float(np.linalg.norm(state.velocity_m_s)),
               float(np.linalg.norm(acceleration)), tilt)
        for key, value in zip(fields, row):
            fields[key].append(np.array(value, copy=True))
        saturations["thrust"] += int(command.wrench.thrust_n >= vehicle.controller.max_total_thrust_n-1e-9)
        saturations["torque"] += int(np.any(abs(command.wrench.torque_nm) >= vehicle.controller.max_torque_nm-1e-9))
        saturations["tilt"] += int(np.rad2deg(np.linalg.norm(command.desired_euler_rad[:2])) >= vehicle.controller.max_tilt_deg-1e-7)
        saturations["body_rate"] += int(np.linalg.norm(state.body_rate_rad_s) >= vehicle.controller.max_body_rate_rad_s-1e-7)
        if clearance <= COLLISION_TOLERANCE_M:
            failure = "actual_collision"
        elif error > float(simulation["max_position_error_m"]):
            failure = "tracking_error_limit"
        elif not np.all(np.isfinite(np.concatenate((state.position_m, state.velocity_m_s,
                                                   state.quaternion_bw, state.body_rate_rad_s)))):
            failure = "nonfinite_state"
        if failure or index == steps:
            break
        next_state = advance_state(state, command.wrench, vehicle.vehicle, dt)
        swept = world.segment_clearance_m(state.position_m, next_state.position_m)
        min_swept_clearance = min(min_swept_clearance, swept)
        if swept <= COLLISION_TOLERANCE_M:
            failure = "actual_swept_collision"
            state = next_state
            continue
        state = next_state
    trace = {key: np.asarray(value) for key, value in fields.items()}
    complete = failure is None and len(trace["time_s"]) == steps+1
    final_error = float(np.linalg.norm(trace["actual_position_m"][-1]-trajectory.waypoints_m[-1]))
    settle_mask = trace["time_s"] >= trajectory.duration_s
    settle_mean = float(np.mean(trace["position_error_m"][settle_mask])) if np.any(settle_mask) else None
    if complete and (final_error > float(simulation["final_error_m"])
                     or settle_mean > float(simulation["settle_mean_error_m"])):
        failure = "did_not_settle"
    actual_limits_satisfied = True
    if limits is not None:
        actual_limits_satisfied = bool(
            np.max(trace["actual_speed_m_s"]) <= limits.max_speed_m_s + 1e-9
            and np.max(trace["actual_acceleration_norm_m_s2"]) <= limits.max_acceleration_m_s2 + 1e-9
            and np.min(trace["thrust_n"]) >= limits.min_thrust_n - 1e-9
            and np.max(trace["thrust_n"]) <= limits.max_thrust_n + 1e-9
            and np.max(trace["actual_tilt_deg"]) <= limits.max_tilt_deg + 1e-9)
        if failure is None and not actual_limits_satisfied:
            failure = "actual_dynamic_limit"
    minimum = min(float(np.min(trace["actual_clearance_m"])), min_swept_clearance)
    metrics = {"flight_success": failure is None, "failure_reason": failure,
               "completed_horizon": complete, "max_position_error_m": float(np.max(trace["position_error_m"])),
               "rms_position_error_m": float(np.sqrt(np.mean(trace["position_error_m"]**2))),
               "final_position_error_m": final_error, "settle_mean_error_m": settle_mean,
               "min_actual_clearance_m": minimum,
               "actual_collision": minimum <= COLLISION_TOLERANCE_M,
               "actual_dynamic_limits_satisfied": actual_limits_satisfied,
               "max_actual_speed_m_s": float(np.max(trace["actual_speed_m_s"])),
               "max_actual_acceleration_m_s2": float(np.max(trace["actual_acceleration_norm_m_s2"])),
               "max_actual_tilt_deg": float(np.max(trace["actual_tilt_deg"])),
               "max_applied_thrust_n": float(np.max(trace["thrust_n"])),
               "saturation_updates": saturations, "control_updates": len(trace["time_s"]),
               "simulated_duration_s": float(trace["time_s"][-1])}
    return trace, metrics


def run(config_path: str | Path, output_dir: str | Path, figures: bool = True) -> list[dict]:
    raw, planner, scenarios, vehicle, limits, seeds, names = load_m7_config(config_path)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"config_snapshot.json").write_text(json.dumps(raw, indent=2)+"\n", encoding="utf-8")
    rows = []
    for scenario in scenarios:
        if scenario.name not in names:
            continue
        world = scenario.world
        planning_world = replace(world, clearance_m=float(raw["planning"]["clearance_m"]))
        for seed in seeds:
            plan = plan_rrt_star(planning_world, scenario.start, scenario.goal, planner, seed)
            trajectory, feasibility = (generate_trajectory(plan.path_m, world, vehicle.vehicle, limits)
                                       if plan.success else (None, {"success": False,
                                             "reason": "planner_"+plan.reason}))
            trace = None
            flight = {"flight_success": False, "failure_reason": "flight_not_started"}
            if trajectory is not None:
                trace, flight = fly(trajectory, world, vehicle, raw["simulation"], limits)
            result = {"scenario": scenario.name, "seed": seed, "planner_success": plan.success,
                      "planner_reason": plan.reason, "path_length_m": plan.path_length_m,
                      "planning_envelope_m": planning_world.envelope_m,
                      "flight_collision_envelope_m": world.envelope_m,
                      "min_planning_clearance_m": plan.min_clearance_m,
                      "trajectory_duration_s": trajectory.duration_s if trajectory else None,
                      "reference_feasibility": feasibility, **flight}
            rows.append(result)
            trial_dir = folder/scenario.name/f"seed_{seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir/"metrics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            (trial_dir/"waypoints.json").write_text(json.dumps({"waypoints_m": plan.path_m,
                    "segment_durations_s": trajectory.durations_s if trajectory else None,
                    "world_envelope_m": world.envelope_m}, indent=2)+"\n", encoding="utf-8")
            if trace is not None:
                np.savez_compressed(trial_dir/"flight.npz", **trace)
                if figures and seed == raw["demonstration"]["visual_seed"]:
                    save_trial_figures(world, trajectory, trace, limits, vehicle.vehicle, trial_dir)
            print(f"{scenario.name} seed {seed}: ref={feasibility['reason']}, "
                  f"flight={flight['failure_reason']}, clearance={flight.get('min_actual_clearance_m')}", flush=True)
    aggregate = {name: {"trials": sum(x["scenario"] == name for x in rows),
                        "reference_valid": sum(x["scenario"] == name and x["reference_feasibility"]["success"] for x in rows),
                        "flight_success": sum(x["scenario"] == name and x["flight_success"] for x in rows)}
                 for name in names}
    (folder/"trials.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    (folder/"aggregate.json").write_text(json.dumps(aggregate, indent=2)+"\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/trajectory_m7.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trajectory_m7"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output_dir, not args.no_figures)


if __name__ == "__main__":
    main()
