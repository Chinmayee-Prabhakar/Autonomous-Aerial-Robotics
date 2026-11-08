"""Attitude-only unscented filter on SO(3), using body-to-world Hamilton quaternions.

Local errors multiply on the right: q_i = q_mean ⊗ Exp(delta_i).  The three
coordinates of delta and their covariance are body-local rotation radians.
Only timestamped body gyro and specific-force measurements enter this module.
"""

from dataclasses import dataclass

import numpy as np

from aerial_autonomy.sensors.imu import ImuMeasurement
from aerial_autonomy.simulation.rotation import (
    normalize_quaternion, quaternion_conjugate, quaternion_product,
    quaternion_to_matrix,
)


_DIM = 3
_EPS = 1e-12


def _vector3(value: np.ndarray, name: str, *, strictly_positive: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    if np.any(vector < 0) or (strictly_positive and np.any(vector <= 0)):
        raise ValueError(f"{name} must contain {'positive' if strictly_positive else 'nonnegative'} values")
    return vector.copy()


def _canonical(q: np.ndarray) -> np.ndarray:
    unit = normalize_quaternion(np.asarray(q, dtype=float))
    # q and -q are the same rotation; choose one representative even at pi.
    if unit[0] < 0 or (abs(unit[0]) <= _EPS and unit[np.argmax(abs(unit))] < 0):
        unit = -unit
    return unit


def rotation_vector_to_quaternion(rotation_rad: np.ndarray) -> np.ndarray:
    """SO(3) exponential map, with a Taylor limit around zero."""
    vector = np.asarray(rotation_rad, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_rad must be a finite three-vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        scale = 0.5 - angle * angle / 48.0
    else:
        scale = np.sin(angle / 2.0) / angle
    return normalize_quaternion(np.r_[np.cos(angle / 2.0), scale * vector])


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm; antipodal quaternion signs give one result."""
    unit = _canonical(quaternion)
    size = float(np.linalg.norm(unit[1:]))
    if size < 1e-8:
        return 2.0 * unit[1:] / unit[0]
    return 2.0 * np.arctan2(size, unit[0]) * unit[1:] / size


def relative_rotation_vector(reference_bw: np.ndarray, other_bw: np.ndarray) -> np.ndarray:
    """Right-local rotation taking reference_bw to other_bw."""
    return quaternion_to_rotation_vector(quaternion_product(
        quaternion_conjugate(normalize_quaternion(reference_bw)),
        normalize_quaternion(other_bw)))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def right_jacobian_so3(rotation_rad: np.ndarray) -> np.ndarray:
    """Transport right-local covariance after applying an attitude correction."""
    vector = np.asarray(rotation_rad, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_rad must be a finite three-vector")
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    if angle < 1e-5:
        return np.eye(3) - 0.5 * cross + (cross @ cross) / 6.0
    return (np.eye(3) - ((1 - np.cos(angle)) / angle ** 2) * cross
            + ((angle - np.sin(angle)) / angle ** 3) * (cross @ cross))


def stabilize_covariance(covariance: np.ndarray, floor_rad2: float) -> tuple[np.ndarray, float]:
    """Symmetrize and floor eigenvalues only when required for Cholesky."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("attitude covariance must be a finite 3x3 matrix")
    if not np.isfinite(floor_rad2) or floor_rad2 <= 0:
        raise ValueError("covariance floor must be finite and positive")
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    if values[0] >= floor_rad2:
        return symmetric, float(np.linalg.norm(symmetric - matrix, ord="fro"))
    corrected = (vectors * np.maximum(values, floor_rad2)) @ vectors.T
    corrected = (corrected + corrected.T) / 2.0
    return corrected, float(np.linalg.norm(corrected - matrix, ord="fro"))


def unscented_weights(alpha: float, beta: float, kappa: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Scaled UT weights for a three-dimensional local attitude error."""
    if (not np.all(np.isfinite([alpha, beta, kappa])) or alpha <= 0
            or beta < 0 or kappa <= -_DIM):
        raise ValueError("invalid scaled unscented-transform parameters")
    scale = alpha * alpha * (_DIM + kappa)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("sigma-point scale must be positive and finite")
    mean = np.full(2 * _DIM + 1, 1.0 / (2.0 * scale))
    mean[0] = (scale - _DIM) / scale
    covariance = mean.copy()
    covariance[0] += 1.0 - alpha * alpha + beta
    return scale, mean, covariance


def local_sigma_points(covariance: np.ndarray, scale: float) -> np.ndarray:
    """Return the central and ± Cholesky-column errors in radians."""
    root = np.linalg.cholesky(scale * covariance)
    return np.vstack((np.zeros(3), root.T, -root.T))


def quaternion_sigma_mean(quaternions: np.ndarray, mean_weights: np.ndarray,
                          tolerance_rad: float, max_iterations: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Iterate the weighted local logarithm until the manifold mean converges."""
    values = np.asarray(quaternions, dtype=float)
    weights = np.asarray(mean_weights, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 4 or weights.shape != (len(values),)
            or not np.all(np.isfinite(weights)) or not np.isclose(np.sum(weights), 1.0)):
        raise ValueError("invalid quaternion sigma set or mean weights")
    mean = normalize_quaternion(values[0])
    for iteration in range(1, max_iterations + 1):
        errors = np.array([relative_rotation_vector(mean, q) for q in values])
        change = weights @ errors
        if np.linalg.norm(change) <= tolerance_rad:
            return mean, errors, iteration
        mean = normalize_quaternion(quaternion_product(mean, rotation_vector_to_quaternion(change)))
    raise RuntimeError("quaternion sigma-point mean did not converge")


@dataclass(frozen=True)
class QuaternionUkfConfig:
    initial_quaternion_bw: np.ndarray
    initial_attitude_std_rad: np.ndarray
    gyro_noise_std_rad_s: np.ndarray  # angular-rate standard deviation per sample
    accel_direction_std: np.ndarray  # unitless Cartesian approximation after normalization
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    covariance_floor_rad2: float = 1e-12
    mean_tolerance_rad: float = 1e-11
    mean_max_iterations: int = 20

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_quaternion_bw, dtype=float)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_quaternion_bw must be a finite scalar-first four-vector")
        object.__setattr__(self, "initial_quaternion_bw",
                           normalize_quaternion(initial))
        for name, positive in (("initial_attitude_std_rad", False),
                               ("gyro_noise_std_rad_s", False),
                               ("accel_direction_std", True)):
            object.__setattr__(self, name, _vector3(getattr(self, name), name,
                                                    strictly_positive=positive))
        unscented_weights(self.alpha, self.beta, self.kappa)
        if not np.isfinite(self.covariance_floor_rad2) or self.covariance_floor_rad2 <= 0:
            raise ValueError("covariance_floor_rad2 must be finite and positive")
        if not np.isfinite(self.mean_tolerance_rad) or self.mean_tolerance_rad <= 0:
            raise ValueError("mean_tolerance_rad must be finite and positive")
        if not isinstance(self.mean_max_iterations, int) or self.mean_max_iterations < 1:
            raise ValueError("mean_max_iterations must be a positive integer")


@dataclass(frozen=True)
class UkfAttitudeUpdate:
    timestamp_s: float
    quaternion_bw: np.ndarray
    covariance_rad2: np.ndarray
    delta_time_s: float
    accel_status: str
    innovation_norm: float
    innovation_condition: float
    mean_iterations: int
    sigma_max_norm_deviation: float
    covariance_symmetry_residual: float
    covariance_min_eigenvalue_rad2: float
    covariance_condition: float
    numerical_corrections: int
    max_correction_norm_rad2: float


class QuaternionUkf:
    """Six-axis attitude UKF; no bias state or truth-data interface."""

    def __init__(self, config: QuaternionUkfConfig):
        self.config = config
        self._q = config.initial_quaternion_bw.copy()
        initial = np.diag(config.initial_attitude_std_rad ** 2)
        self._P, correction = stabilize_covariance(initial, config.covariance_floor_rad2)
        self._correction_count = int(correction > 1e-15)
        self._max_correction = correction
        self._last_timestamp_s: float | None = None
        self._scale, self._wm, self._wc = unscented_weights(
            config.alpha, config.beta, config.kappa)

    @property
    def quaternion_bw(self) -> np.ndarray:
        return self._q.copy()

    @property
    def covariance_rad2(self) -> np.ndarray:
        return self._P.copy()

    def _record_stabilization(self, matrix: np.ndarray) -> np.ndarray:
        corrected, change = stabilize_covariance(matrix, self.config.covariance_floor_rad2)
        if change > 1e-15:
            self._correction_count += 1
            self._max_correction = max(self._max_correction, change)
        return corrected

    def _result(self, time_s: float, dt_s: float, status: str, innovation: float,
                innovation_condition: float, iterations: int, sigma_deviation: float) -> UkfAttitudeUpdate:
        symmetry = float(np.linalg.norm(self._P - self._P.T, ord="fro"))
        eigenvalues = np.linalg.eigvalsh(self._P)
        return UkfAttitudeUpdate(
            time_s, self._q.copy(), self._P.copy(), dt_s, status, innovation,
            innovation_condition, iterations, sigma_deviation, symmetry,
            float(eigenvalues[0]), float(np.linalg.cond(self._P)),
            self._correction_count, self._max_correction)

    def update(self, reading: ImuMeasurement) -> UkfAttitudeUpdate:
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
            raise ValueError("IMU timestamps must increase strictly")
        accel_norm = float(np.linalg.norm(accel))
        status = ("nonfinite" if not np.isfinite(accel_norm) else
                  "near_zero" if accel_norm <= _EPS else "valid")
        if self._last_timestamp_s is None:
            self._last_timestamp_s = time_s
            return self._result(time_s, 0.0, "initial_sample", 0.0, float("nan"), 0, 0.0)

        dt_s = time_s - self._last_timestamp_s
        errors = local_sigma_points(self._P, self._scale)
        increment = rotation_vector_to_quaternion(gyro * dt_s)
        propagated = np.array([normalize_quaternion(quaternion_product(
            quaternion_product(self._q, rotation_vector_to_quaternion(error)), increment))
            for error in errors])
        sigma_deviation = float(np.max(np.abs(np.linalg.norm(propagated, axis=1) - 1.0)))
        predicted_q, predicted_errors, iterations = quaternion_sigma_mean(
            propagated, self._wm, self.config.mean_tolerance_rad,
            self.config.mean_max_iterations)
        predicted_P = sum(weight * np.outer(error, error)
                          for weight, error in zip(self._wc, predicted_errors))
        # Gyro noise is specified in rad/s per sample, so angle variance grows as dt².
        predicted_P += np.diag((self.config.gyro_noise_std_rad_s * dt_s) ** 2)
        predicted_P = self._record_stabilization(predicted_P)
        innovation = 0.0
        innovation_condition = float("nan")
        updated_q, updated_P = predicted_q, predicted_P
        if status == "valid":
            # Regenerate sigma points from P_pred so the accelerometer update
            # sees uncertainty added by the gyro process model as well.
            measurement_errors = local_sigma_points(predicted_P, self._scale)
            measurement_quaternions = np.array([
                normalize_quaternion(quaternion_product(
                    predicted_q, rotation_vector_to_quaternion(error)))
                for error in measurement_errors])
            sigma_deviation = max(sigma_deviation, float(np.max(np.abs(
                np.linalg.norm(measurement_quaternions, axis=1) - 1.0))))
            directions = np.array([quaternion_to_matrix(q).T @ [0, 0, 1]
                                   for q in measurement_quaternions])
            mean_direction = self._wm @ directions
            deviations = directions - mean_direction
            noise = np.diag(self.config.accel_direction_std ** 2)
            innovation_covariance = noise + sum(
                weight * np.outer(delta, delta)
                for weight, delta in zip(self._wc, deviations))
            cross = sum(weight * np.outer(error, delta)
                        for weight, error, delta in zip(
                            self._wc, measurement_errors, deviations))
            gain = np.linalg.solve(innovation_covariance, cross.T).T
            residual = accel / accel_norm - mean_direction
            innovation = float(np.linalg.norm(residual))
            innovation_condition = float(np.linalg.cond(innovation_covariance))
            correction = gain @ residual
            updated_q = normalize_quaternion(quaternion_product(
                predicted_q, rotation_vector_to_quaternion(correction)))
            local_P = predicted_P - gain @ innovation_covariance @ gain.T
            transport = right_jacobian_so3(correction)
            updated_P = self._record_stabilization(transport @ local_P @ transport.T)
        self._q, self._P = updated_q, updated_P
        self._last_timestamp_s = time_s
        return self._result(time_s, dt_s, status, innovation,
                            innovation_condition, iterations, sigma_deviation)
