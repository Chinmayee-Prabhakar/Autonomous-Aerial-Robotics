"""Minimal six-degree-of-freedom rigid-body quadrotor dynamics.

World: ENU (x east, y north, z up). Body: FLU (x forward, y left, z up).
Thrust acts along +body z. This model commands total thrust and body torque
directly; motor allocation, contact, aerodynamics beyond linear drag, and
sensor noise are deliberately outside the first milestone.
"""

from dataclasses import dataclass
import numpy as np

from aerial_autonomy.config import VehicleConfig
from aerial_autonomy.simulation.rotation import integrate_quaternion, quaternion_to_matrix


@dataclass
class VehicleState:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    quaternion_bw: np.ndarray
    body_rate_rad_s: np.ndarray

    @classmethod
    def at_rest(cls) -> "VehicleState":
        return cls(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))


@dataclass(frozen=True)
class Wrench:
    thrust_n: float
    torque_nm: np.ndarray


def world_acceleration(state: VehicleState, wrench: Wrench, vehicle: VehicleConfig) -> np.ndarray:
    """Instantaneous ENU acceleration from thrust, linear drag, and gravity."""
    body_z_world = quaternion_to_matrix(state.quaternion_bw)[:, 2]
    force_world = wrench.thrust_n * body_z_world - vehicle.linear_drag_n_s_m * state.velocity_m_s
    return force_world / vehicle.mass_kg - np.array([0.0, 0.0, vehicle.gravity_m_s2])


def advance_state(state: VehicleState, wrench: Wrench, vehicle: VehicleConfig, dt_s: float) -> VehicleState:
    """Semi-implicit fixed-step update of translation, rotation, and attitude."""
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("time step must be finite and positive")
    if wrench.thrust_n < 0 or not np.isfinite(wrench.thrust_n):
        raise ValueError("thrust must be finite and nonnegative")
    if np.asarray(wrench.torque_nm).shape != (3,) or not np.all(np.isfinite(wrench.torque_nm)):
        raise ValueError("torque must be a finite three-vector")

    acceleration = world_acceleration(state, wrench, vehicle)
    velocity = state.velocity_m_s + dt_s * acceleration
    position = state.position_m + dt_s * velocity

    inertia = vehicle.inertia_kg_m2
    angular_momentum = inertia * state.body_rate_rad_s
    angular_acceleration = (wrench.torque_nm - np.cross(state.body_rate_rad_s, angular_momentum)) / inertia
    body_rate = state.body_rate_rad_s + dt_s * angular_acceleration
    quaternion = integrate_quaternion(state.quaternion_bw, body_rate, dt_s)
    return VehicleState(position, velocity, quaternion, body_rate)

