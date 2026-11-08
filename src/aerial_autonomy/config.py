"""Typed parameters loaded from a small, standard-library TOML file."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


def _vector3(values: list[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite numbers")
    return result


@dataclass(frozen=True)
class VehicleConfig:
    mass_kg: float
    inertia_kg_m2: np.ndarray
    gravity_m_s2: float
    linear_drag_n_s_m: float

    def __post_init__(self) -> None:
        if self.mass_kg <= 0 or self.gravity_m_s2 <= 0 or self.linear_drag_n_s_m < 0:
            raise ValueError("vehicle mass and gravity must be positive; drag cannot be negative")
        if self.inertia_kg_m2.shape != (3,) or np.any(self.inertia_kg_m2 <= 0):
            raise ValueError("inertia must have three positive diagonal entries")


@dataclass(frozen=True)
class ControllerConfig:
    position_kp_s2: np.ndarray
    velocity_kd_s: np.ndarray
    attitude_kp_s: float
    rate_kp_s: float
    max_tilt_deg: float
    max_body_rate_rad_s: float
    max_total_thrust_n: float
    max_torque_nm: np.ndarray

    def __post_init__(self) -> None:
        if np.any(self.position_kp_s2 < 0) or np.any(self.velocity_kd_s < 0):
            raise ValueError("position and velocity gains must be nonnegative")
        if self.attitude_kp_s <= 0 or self.rate_kp_s <= 0:
            raise ValueError("attitude and rate gains must be positive")
        if not 0 < self.max_tilt_deg < 90 or self.max_body_rate_rad_s <= 0:
            raise ValueError("tilt and body-rate limits must be positive")
        if self.max_total_thrust_n <= 0 or np.any(self.max_torque_nm <= 0):
            raise ValueError("thrust and torque limits must be positive")


@dataclass(frozen=True)
class DemoConfig:
    duration_s: float
    step_s: float
    target_position_m: np.ndarray
    target_yaw_deg: float
    seed: int

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.step_s <= 0:
            raise ValueError("duration and time step must be positive")
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("duration must be an integer multiple of step")


@dataclass(frozen=True)
class ExperimentConfig:
    vehicle: VehicleConfig
    controller: ControllerConfig
    demo: DemoConfig


@dataclass(frozen=True)
class ImuConfig:
    sample_hz: float
    accel_bias_m_s2: np.ndarray
    gyro_bias_rad_s: np.ndarray
    accel_noise_std_m_s2: np.ndarray
    gyro_noise_std_rad_s: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_hz) or self.sample_hz <= 0:
            raise ValueError("IMU sample_hz must be finite and positive")
        for name in ("accel_bias_m_s2", "gyro_bias_rad_s", "accel_noise_std_m_s2", "gyro_noise_std_rad_s"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite numbers")
            object.__setattr__(self, name, value)
        if np.any(self.accel_noise_std_m_s2 < 0) or np.any(self.gyro_noise_std_rad_s < 0):
            raise ValueError("IMU noise standard deviations cannot be negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("IMU seed must be a nonnegative integer")


@dataclass(frozen=True)
class ManeuverConfig:
    switch_time_s: float
    target_position_m: np.ndarray
    target_yaw_deg: float


@dataclass(frozen=True)
class ImuExperimentConfig:
    base: ExperimentConfig
    imu: ImuConfig
    maneuver: ManeuverConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    vehicle = raw["vehicle"]
    controller = raw["controller"]
    demo = raw["demo"]
    return ExperimentConfig(
        vehicle=VehicleConfig(
            mass_kg=float(vehicle["mass_kg"]),
            inertia_kg_m2=_vector3(vehicle["inertia_kg_m2"], "inertia_kg_m2"),
            gravity_m_s2=float(vehicle["gravity_m_s2"]),
            linear_drag_n_s_m=float(vehicle["linear_drag_n_s_m"]),
        ),
        controller=ControllerConfig(
            position_kp_s2=_vector3(controller["position_kp_s2"], "position_kp_s2"),
            velocity_kd_s=_vector3(controller["velocity_kd_s"], "velocity_kd_s"),
            attitude_kp_s=float(controller["attitude_kp_s"]),
            rate_kp_s=float(controller["rate_kp_s"]),
            max_tilt_deg=float(controller["max_tilt_deg"]),
            max_body_rate_rad_s=float(controller["max_body_rate_rad_s"]),
            max_total_thrust_n=float(controller["max_total_thrust_n"]),
            max_torque_nm=_vector3(controller["max_torque_nm"], "max_torque_nm"),
        ),
        demo=DemoConfig(
            duration_s=float(demo["duration_s"]),
            step_s=float(demo["step_s"]),
            target_position_m=_vector3(demo["target_position_m"], "target_position_m"),
            target_yaw_deg=float(demo["target_yaw_deg"]),
            seed=int(demo["seed"]),
        ),
    )


def load_imu_experiment(path: str | Path) -> ImuExperimentConfig:
    """Reuse the A1 plant/controller fields and add IMU and waypoint settings."""
    base = load_config(path)
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    sensor = raw["imu"]
    maneuver = raw["maneuver"]
    switch_time = float(maneuver["switch_time_s"])
    if not np.isfinite(switch_time) or not 0 < switch_time < base.demo.duration_s:
        raise ValueError("maneuver switch_time_s must fall strictly inside the demo duration")
    target_yaw = float(maneuver["target_yaw_deg"])
    if not np.isfinite(target_yaw):
        raise ValueError("maneuver target_yaw_deg must be finite")
    return ImuExperimentConfig(
        base=base,
        imu=ImuConfig(
            sample_hz=float(sensor["sample_hz"]),
            accel_bias_m_s2=_vector3(sensor["accel_bias_m_s2"], "accel_bias_m_s2"),
            gyro_bias_rad_s=_vector3(sensor["gyro_bias_rad_s"], "gyro_bias_rad_s"),
            accel_noise_std_m_s2=_vector3(sensor["accel_noise_std_m_s2"], "accel_noise_std_m_s2"),
            gyro_noise_std_rad_s=_vector3(sensor["gyro_noise_std_rad_s"], "gyro_noise_std_rad_s"),
            seed=sensor["seed"],
        ),
        maneuver=ManeuverConfig(
            switch_time_s=switch_time,
            target_position_m=_vector3(maneuver["target_position_m"], "maneuver target_position_m"),
            target_yaw_deg=target_yaw,
        ),
    )

