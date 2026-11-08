"""Six-axis Madgwick attitude update in this project's ENU/FLU convention.

The unit Hamilton quaternion maps body FLU vectors to world ENU vectors.
This module consumes only timestamped IMU measurements; it never uses plant
state, world acceleration, or evaluation truth.
"""

from dataclasses import dataclass

import numpy as np

from aerial_autonomy.sensors.imu import ImuMeasurement
from aerial_autonomy.simulation.rotation import normalize_quaternion, quaternion_product


_EPS = 1e-12


@dataclass(frozen=True)
class MadgwickConfig:
    beta_per_s: float
    initial_quaternion_bw: np.ndarray
    accel_norm_gate_m_s2: float | None = None
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta_per_s) or self.beta_per_s < 0:
            raise ValueError("beta_per_s must be finite and nonnegative")
        q = np.asarray(self.initial_quaternion_bw, dtype=float)
        if q.shape != (4,):
            raise ValueError("initial_quaternion_bw must have four finite components")
        object.__setattr__(self, "initial_quaternion_bw", normalize_quaternion(q))
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        if self.accel_norm_gate_m_s2 is not None:
            if not np.isfinite(self.accel_norm_gate_m_s2) or self.accel_norm_gate_m_s2 < 0:
                raise ValueError("accel_norm_gate_m_s2 must be finite and nonnegative")


@dataclass(frozen=True)
class AttitudeUpdate:
    timestamp_s: float
    quaternion_bw: np.ndarray
    delta_time_s: float
    accel_norm_m_s2: float
    correction_used: bool
    accel_status: str
    gradient_norm: float


def gravity_direction_and_jacobian(quaternion_bw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict R_bw.T @ e_z and its polynomial Jacobian for a unit quaternion.

    q=(w,x,y,z), h=[2(xz-wy), 2(yz+wx), 1-2(x²+y²)]. The third row of
    R_bw gives the +up specific-force direction in body coordinates.
    """
    q = np.asarray(quaternion_bw, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion_bw must have four finite components")
    w, x, y, z = q
    direction = np.array([2 * (x * z - w * y),
                          2 * (y * z + w * x),
                          1 - 2 * (x * x + y * y)])
    jacobian = np.array([
        [-2 * y, 2 * z, -2 * w, 2 * x],
        [2 * x, 2 * w, 2 * z, 2 * y],
        [0.0, -4 * x, -4 * y, 0.0],
    ])
    return direction, jacobian


class MadgwickImu:
    """Stateful gyro integration plus normalized accelerometer gradient step.

    The first reading establishes the time origin and returns the configured
    orientation. Later readings use their own positive timestamp difference.
    Invalid/near-zero accelerometer data cause explicit gyro-only fallback;
    invalid gyro data or timestamps raise ValueError without changing state.
    """

    def __init__(self, config: MadgwickConfig):
        self.config = config
        self._quaternion_bw = config.initial_quaternion_bw.copy()
        self._last_timestamp_s: float | None = None

    @property
    def quaternion_bw(self) -> np.ndarray:
        return self._quaternion_bw.copy()

    def update(self, reading: ImuMeasurement) -> AttitudeUpdate:
        time_s = float(reading.timestamp_s)
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        gyro = np.asarray(reading.gyro_rad_s, dtype=float)
        if gyro.shape != (3,) or not np.all(np.isfinite(gyro)):
            raise ValueError("gyro_rad_s must be a finite three-vector")
        accel = np.asarray(reading.accel_m_s2, dtype=float)
        if accel.shape != (3,):
            raise ValueError("accel_m_s2 must be a three-vector")
        if self._last_timestamp_s is not None and time_s <= self._last_timestamp_s:
            raise ValueError("IMU timestamps must increase strictly; zero or negative interval")

        accel_norm = float(np.linalg.norm(accel))
        if not np.isfinite(accel_norm):
            accel_status = "nonfinite"
        elif accel_norm <= _EPS:
            accel_status = "near_zero"
        elif (self.config.accel_norm_gate_m_s2 is not None
              and abs(accel_norm - self.config.gravity_m_s2) > self.config.accel_norm_gate_m_s2):
            accel_status = "magnitude_gated"
        else:
            accel_status = "valid"

        if self._last_timestamp_s is None:
            self._last_timestamp_s = time_s
            return AttitudeUpdate(time_s, self.quaternion_bw, 0.0, accel_norm,
                                  False, "initial_sample", 0.0)

        dt_s = time_s - self._last_timestamp_s
        q = self._quaternion_bw
        # Body rates right-multiply q_bw: q_dot = 1/2 q_bw ⊗ [0, omega_body].
        q_dot = 0.5 * quaternion_product(q, np.r_[0.0, gyro])
        gradient_norm = 0.0
        correction_used = False
        if accel_status == "valid" and self.config.beta_per_s > 0:
            predicted, jacobian = gravity_direction_and_jacobian(q)
            residual = predicted - accel / accel_norm
            gradient = jacobian.T @ residual
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > _EPS:
                q_dot -= self.config.beta_per_s * gradient / gradient_norm
                correction_used = True
            else:
                accel_status = "zero_gradient"
        elif accel_status == "valid":
            accel_status = "gain_zero"

        updated = normalize_quaternion(q + dt_s * q_dot)
        self._quaternion_bw = updated
        self._last_timestamp_s = time_s
        return AttitudeUpdate(time_s, updated.copy(), dt_s, accel_norm,
                              correction_used, accel_status, gradient_norm)
