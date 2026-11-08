"""Stop-at-waypoint quintic references on certified geometric edges (ENU/FLU)."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.polynomial import Polynomial

from aerial_autonomy.config import VehicleConfig
from aerial_autonomy.planning.geometry import COLLISION_TOLERANCE_M, World, vector3


@dataclass(frozen=True)
class TrajectoryLimits:
    max_speed_m_s: float
    max_acceleration_m_s2: float
    max_tilt_deg: float
    min_thrust_n: float
    max_thrust_n: float
    initial_segment_s: float
    max_segment_s: float
    stretch_factor: float = 1.25
    retiming_target_speed_m_s: float | None = None
    retiming_target_acceleration_m_s2: float | None = None

    def __post_init__(self) -> None:
        values = vars(self)
        if any(x is not None and not math.isfinite(x) for x in values.values()):
            raise ValueError("trajectory limits must be finite")
        if (self.max_speed_m_s <= 0 or self.max_acceleration_m_s2 <= 0
                or not 0 < self.max_tilt_deg < 90 or self.min_thrust_n < 0
                or self.max_thrust_n <= self.min_thrust_n or self.initial_segment_s <= 0
                or self.max_segment_s < self.initial_segment_s or self.stretch_factor <= 1):
            raise ValueError("invalid trajectory limits")
        if (self.retiming_target_speed_m_s is not None and
                not 0 < self.retiming_target_speed_m_s <= self.max_speed_m_s):
            raise ValueError("retiming speed target exceeds physical limit")
        if (self.retiming_target_acceleration_m_s2 is not None and
                not 0 < self.retiming_target_acceleration_m_s2 <= self.max_acceleration_m_s2):
            raise ValueError("retiming acceleration target exceeds physical limit")


@dataclass(frozen=True)
class Reference:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    waypoints_m: tuple[tuple[float, float, float], ...]
    durations_s: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return sum(self.durations_s)

    def evaluate(self, time_s: float) -> Reference:
        if not math.isfinite(time_s) or time_s < 0 or time_s > self.duration_s + 1e-9:
            raise ValueError("time outside trajectory")
        elapsed = 0.0
        for index, duration in enumerate(self.durations_s):
            if time_s <= elapsed + duration or index == len(self.durations_s) - 1:
                u = float(np.clip((time_s - elapsed) / duration, 0, 1))
                displacement = np.asarray(self.waypoints_m[index + 1]) - np.asarray(self.waypoints_m[index])
                s = 10*u**3 - 15*u**4 + 6*u**5
                ds = 30*u**2 - 60*u**3 + 30*u**4
                dds = 60*u - 180*u**2 + 120*u**3
                return Reference(np.asarray(self.waypoints_m[index]) + displacement*s,
                                 displacement*ds/duration, displacement*dds/duration**2)
            elapsed += duration
        raise AssertionError("unreachable")


def reference_clearance(world: World, waypoints: Sequence[Sequence[float]]) -> float:
    """Exact complete-curve clearance: quintic scaling is monotone on each edge."""
    if len(waypoints) < 2:
        raise ValueError("path needs at least two waypoints")
    return min(world.segment_clearance_m(a, b) for a, b in zip(waypoints[:-1], waypoints[1:]))


def _segment_metrics(length: float, unit: np.ndarray, duration: float,
                     vehicle: VehicleConfig) -> dict[str, float]:
    speed = length * 15/8 / duration
    acceleration = length * (10*math.sqrt(3)/3) / duration**2
    # Force includes the M1 linear drag required to track the reference.
    q = (vehicle.mass_kg * length / duration**2
         * Polynomial([0, 60, -180, 120])
         + vehicle.linear_drag_n_s_m * length / duration
         * Polynomial([0, 0, 30, -60, 30]))
    roots = q.deriv().roots()
    points = [0.0, 1.0] + [float(z.real) for z in roots
                            if abs(z.imag) < 1e-9 and 0 < z.real < 1]
    samples = [float(q(u)) for u in points]
    low, high = min(samples), max(samples)
    gravity_force = vehicle.mass_kg * vehicle.gravity_m_s2
    def thrust(value: float) -> float:
        return float(np.linalg.norm(gravity_force*np.array([0., 0., 1.]) + value*unit))
    def tilt(value: float) -> float:
        force = gravity_force*np.array([0., 0., 1.]) + value*unit
        return math.degrees(math.atan2(float(np.linalg.norm(force[:2])), float(force[2])))
    min_q = float(np.clip(-gravity_force*unit[2], low, high))
    return {"max_speed_m_s": speed, "max_acceleration_m_s2": acceleration,
            "min_thrust_n": thrust(min_q),
            "max_thrust_n": max(thrust(low), thrust(high)),
            "max_tilt_deg": max(tilt(low), tilt(high)),
            "min_vertical_force_n": gravity_force + min(low*unit[2], high*unit[2])}


def generate_trajectory(waypoints: Sequence[Sequence[float]], world: World,
                        vehicle: VehicleConfig, limits: TrajectoryLimits) -> tuple[Trajectory | None, dict]:
    """Retiming never changes geometry; failures carry explicit reason codes."""
    report: dict = {"success": False, "reason": None, "method": "stop_at_waypoint_quintic",
                    "continuous_collision_checked": False, "envelope_m": world.envelope_m}
    try:
        path = tuple(vector3(point, "waypoint") for point in waypoints)
        if len(path) < 2 or any(math.dist(a, b) <= 1e-9 for a, b in zip(path[:-1], path[1:])):
            raise ValueError("invalid_path")
    except (ValueError, TypeError):
        report["reason"] = "invalid_path"
        return None, report
    clearance = reference_clearance(world, path)
    report.update(min_reference_clearance_m=clearance,
                  continuous_collision_checked=True)
    if clearance <= COLLISION_TOLERANCE_M:
        report["reason"] = "reference_collision"
        return None, report
    target_speed = limits.retiming_target_speed_m_s or limits.max_speed_m_s
    target_acceleration = limits.retiming_target_acceleration_m_s2 or limits.max_acceleration_m_s2
    durations = []
    metrics = []
    for a, b in zip(path[:-1], path[1:]):
        displacement = np.asarray(b) - np.asarray(a)
        length = float(np.linalg.norm(displacement))
        duration = limits.initial_segment_s
        while True:
            item = _segment_metrics(length, displacement/length, duration, vehicle)
            feasible = (item["max_speed_m_s"] <= target_speed + 1e-10
                        and item["max_acceleration_m_s2"] <= target_acceleration + 1e-10
                        and item["min_thrust_n"] >= limits.min_thrust_n - 1e-10
                        and item["max_thrust_n"] <= limits.max_thrust_n + 1e-10
                        and item["max_tilt_deg"] <= limits.max_tilt_deg + 1e-10
                        and item["min_vertical_force_n"] > 0)
            if feasible:
                durations.append(duration)
                metrics.append(item)
                break
            if duration >= limits.max_segment_s - 1e-12:
                report.update(reason="dynamic_limits_unmet", failed_segment=len(durations),
                              failed_metrics=item)
                return None, report
            duration = min(duration*limits.stretch_factor, limits.max_segment_s)
    report.update(success=True, reason="feasible", durations_s=durations,
                  retiming_target_speed_m_s=target_speed,
                  retiming_target_acceleration_m_s2=target_acceleration,
                  path_length_m=sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:])),
                  max_speed_m_s=max(x["max_speed_m_s"] for x in metrics),
                  max_acceleration_m_s2=max(x["max_acceleration_m_s2"] for x in metrics),
                  min_thrust_n=min(x["min_thrust_n"] for x in metrics),
                  max_thrust_n=max(x["max_thrust_n"] for x in metrics),
                  max_tilt_deg=max(x["max_tilt_deg"] for x in metrics),
                  start_end_velocity_acceleration_zero=True,
                  position_velocity_acceleration_continuous=True)
    return Trajectory(path, tuple(durations)), report
