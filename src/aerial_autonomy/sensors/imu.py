"""Body-frame gyroscope and specific-force accelerometer observations."""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.config import ImuConfig, VehicleConfig
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench, world_acceleration
from aerial_autonomy.simulation.rotation import quaternion_to_matrix


@dataclass(frozen=True)
class ImuMeasurement:
    timestamp_s: float
    accel_m_s2: np.ndarray
    gyro_rad_s: np.ndarray


def specific_force_body(quaternion_bw: np.ndarray, acceleration_world_m_s2: np.ndarray,
                        gravity_m_s2: float) -> np.ndarray:
    """R_bw^T (a_world - g_world), with ENU gravity along negative z."""
    gravity_world = np.array([0.0, 0.0, -gravity_m_s2])
    return quaternion_to_matrix(quaternion_bw).T @ (acceleration_world_m_s2 - gravity_world)


def ideal_imu(state: VehicleState, wrench: Wrench, vehicle: VehicleConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return true body specific force (m/s²) and body rate (rad/s)."""
    acceleration_world = world_acceleration(state, wrench, vehicle)
    force_body = specific_force_body(state.quaternion_bw, acceleration_world, vehicle.gravity_m_s2)
    return force_body, state.body_rate_rad_s.copy()


class SimulatedImu:
    """Seeded discrete-time IMU; only `ImuMeasurement` is estimator-visible."""

    def __init__(self, config: ImuConfig, plant_step_s: float):
        if not np.isfinite(plant_step_s) or plant_step_s <= 0:
            raise ValueError("plant_step_s must be finite and positive")
        ratio = (1.0 / config.sample_hz) / plant_step_s
        stride = round(ratio)
        if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("IMU period must be an integer multiple of plant step; interpolation is unsupported")
        self.config = config
        self.plant_step_s = plant_step_s
        self.sample_stride = stride
        self._rng = np.random.default_rng(config.seed)

    def should_sample(self, step_index: int) -> bool:
        return step_index >= 0 and step_index % self.sample_stride == 0

    def sample(self, step_index: int, state: VehicleState, wrench: Wrench,
               vehicle: VehicleConfig) -> ImuMeasurement:
        if not self.should_sample(step_index):
            raise ValueError("step_index is outside the configured IMU sampling grid")
        force_body, body_rate = ideal_imu(state, wrench, vehicle)
        accel_noise = self._rng.normal(0.0, self.config.accel_noise_std_m_s2, size=3)
        gyro_noise = self._rng.normal(0.0, self.config.gyro_noise_std_rad_s, size=3)
        return ImuMeasurement(
            timestamp_s=step_index * self.plant_step_s,
            accel_m_s2=force_body + self.config.accel_bias_m_s2 + accel_noise,
            gyro_rad_s=body_rate + self.config.gyro_bias_rad_s + gyro_noise,
        )
