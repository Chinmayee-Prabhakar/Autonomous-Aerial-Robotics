"""Reproducible M6 geometric planner benchmark and independent path checks."""

import argparse
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import statistics
import tomllib

from aerial_autonomy.planning.geometry import AABB, Sphere, Workspace, World, vector3
from aerial_autonomy.planning.rrt_star import PlannerConfig, PlanningResult, audit_result, plan_rrt_star
from aerial_autonomy.visualization.rrt_star_plot import save_convergence_figure, save_plan_figure


@dataclass(frozen=True)
class Scenario:
    name: str
    world: World
    start: tuple[float, float, float]
    goal: tuple[float, float, float]


def load_benchmark(path: str | Path) -> tuple[PlannerConfig, tuple[int, ...], tuple[Scenario, ...], dict]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    planner = raw["planner"]
    config = PlannerConfig(**{key: planner[key] for key in
                              ("max_iterations", "step_m", "neighbor_radius_m", "goal_radius_m", "goal_bias")})
    seeds = tuple(planner["seeds"])
    if not seeds or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    scenarios = []
    for entry in raw["scenarios"]:
        obstacles = []
        for item in entry["obstacles"]:
            if item["type"] == "sphere":
                obstacles.append(Sphere(item["center"], item["radius_m"]))
            elif item["type"] == "box":
                obstacles.append(AABB(item["minimum"], item["maximum"]))
            else:
                raise ValueError(f"unknown obstacle type: {item['type']}")
        world = World(Workspace(entry["workspace_min"], entry["workspace_max"]),
                      tuple(obstacles), planner["vehicle_radius_m"], planner["clearance_m"])
        scenarios.append(Scenario(entry["name"], world, vector3(entry["start"], "start"),
                                  vector3(entry["goal"], "goal")))
    if not scenarios or len(set(s.name for s in scenarios)) != len(scenarios):
        raise ValueError("scenario names must be unique and nonempty")
    return config, seeds, tuple(scenarios), raw


def result_record(scenario: Scenario, result: PlanningResult, audit: dict[str, bool] | None = None) -> dict:
    """A flat record that preserves failed trials with a missing path length."""
    return {
        "scenario": scenario.name, "seed": result.seed, "success": result.success,
        "reason": result.reason, "iterations": result.iterations,
        "first_solution_iteration": result.first_solution_iteration,
        "path_length_m": result.path_length_m, "tree_nodes": len(result.nodes),
        "attempted_rewires": result.attempted_rewires,
        "accepted_rewires": result.accepted_rewires,
        "collision_checks": result.collision_checks,
        "path_valid": result.path_valid, "min_clearance_m": result.min_clearance_m,
        "runtime_s": result.runtime_s, "audit_passed": all(audit.values()) if audit is not None else None,
    }


def aggregate(records: list[dict]) -> list[dict]:
    output = []
    for scenario in dict.fromkeys(row["scenario"] for row in records):
        subset = [row for row in records if row["scenario"] == scenario]
        successes = [row for row in subset if row["success"]]
        lengths = [row["path_length_m"] for row in successes]
        output.append({
            "scenario": scenario, "runs": len(subset), "successes": len(successes),
            "success_rate": len(successes) / len(subset),
            "mean_path_length_m": statistics.mean(lengths) if lengths else None,
            "std_path_length_m": statistics.stdev(lengths) if len(lengths) > 1 else (0.0 if lengths else None),
            "mean_first_solution_iteration": statistics.mean(row["first_solution_iteration"] for row in successes)
            if successes else None,
            "mean_tree_nodes": statistics.mean(row["tree_nodes"] for row in subset),
            "mean_runtime_s": statistics.mean(row["runtime_s"] for row in subset),
            "total_attempted_rewires": sum(row["attempted_rewires"] for row in subset),
            "total_accepted_rewires": sum(row["accepted_rewires"] for row in subset),
            "all_successful_paths_valid": all(row["path_valid"] for row in successes),
            "all_tree_and_path_audits_passed": all(row["audit_passed"] for row in subset),
        })
    return output


def run_benchmark(config_path: str | Path, output_dir: str | Path,
                  selected_scenarios: tuple[str, ...] = ()) -> tuple[list[dict], list[dict]]:
    config, seeds, scenarios, raw = load_benchmark(config_path)
    if selected_scenarios:
        unknown = set(selected_scenarios) - {s.name for s in scenarios}
        if unknown:
            raise ValueError(f"unknown scenarios: {sorted(unknown)}")
        scenarios = tuple(s for s in scenarios if s.name in selected_scenarios)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    records: list[dict] = []
    for scenario in scenarios:
        folder = output / scenario.name
        folder.mkdir(parents=True, exist_ok=True)
        representative: PlanningResult | None = None
        for seed in seeds:
            result = plan_rrt_star(scenario.world, scenario.start, scenario.goal, config, seed)
            audit = audit_result(scenario.world, scenario.start, scenario.goal, result)
            if not all(audit.values()):
                raise RuntimeError(f"independent audit failed for {scenario.name} seed {seed}: {audit}")
            record = result_record(scenario, result, audit)
            records.append(record)
            payload = {"metadata": {"scenario": scenario.name, "start_m": scenario.start,
                                     "goal_m": scenario.goal, "seed": seed,
                                     "vehicle_radius_m": scenario.world.vehicle_radius_m,
                                     "safety_clearance_m": scenario.world.clearance_m,
                                     "planner": asdict(config)},
                       "result": asdict(result), "independent_audit": audit}
            (folder / f"seed_{seed}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"{scenario.name:9} seed {seed:2}  {result.reason:26} "
                  f"length={result.path_length_m}  nodes={len(result.nodes)}  {result.runtime_s:.2f}s", flush=True)
            if representative is None or (not representative.success and result.success):
                representative = result
        assert representative is not None
        save_plan_figure(scenario.world, scenario.start, scenario.goal, representative,
                         scenario.name, folder / "plan_3d.png")
        save_convergence_figure(representative, scenario.name, folder / "convergence.png")
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = aggregate(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["M6 geometric RRT* benchmark", "", f"Seeds: {list(seeds)}",
             f"Iteration budget: {config.max_iterations}", ""]
    for item in summary:
        lines.append(f"{item['scenario']}: {item['successes']}/{item['runs']} successes; "
                     f"mean path length={item['mean_path_length_m']} m; "
                     f"mean runtime={item['mean_runtime_s']:.2f} s")
    lines += ["", "Failed lengths are missing. Finite-budget failures do not prove infeasibility.",
              "Waypoints are geometric only; dynamic feasibility belongs to M7."]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rrt_star.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rrt_star"))
    parser.add_argument("--scenarios", nargs="*", default=())
    args = parser.parse_args()
    run_benchmark(args.config, args.output_dir, tuple(args.scenarios))


if __name__ == "__main__":
    main()
