"""Scalar-first Hamilton quaternions mapping body FLU vectors into world ENU."""

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion norm is zero or invalid")
    return np.asarray(q, dtype=float) / norm


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic ZYX yaw-pitch-roll, angles in radians."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return normalize_quaternion(np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ]))


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Return intrinsic ZYX roll, pitch, yaw in radians."""
    r = quaternion_to_matrix(q)
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])


def integrate_quaternion(q: np.ndarray, body_rate_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
    """Integrate a constant body angular rate over one step with a rotation increment."""
    angle = float(np.linalg.norm(body_rate_rad_s) * dt_s)
    if angle < 1e-12:
        delta = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = body_rate_rad_s / np.linalg.norm(body_rate_rad_s)
        delta = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    return normalize_quaternion(quaternion_product(q, delta))

