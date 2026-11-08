"""Position PD -> desired attitude/rate -> torque controller."""

from dataclasses import dataclass
import numpy as np

from aerial_autonomy.config import ControllerConfig, VehicleConfig
from aerial_autonomy.simulation.dynamics import VehicleState, Wrench
from aerial_autonomy.simulation.rotation import (
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_product,
)


@dataclass(frozen=True)
class ControlResult:
    wrench: Wrench
    desired_euler_rad: np.ndarray


class CascadedHoverController:
    def __init__(self, vehicle: VehicleConfig, gains: ControllerConfig):
        if gains.max_total_thrust_n <= vehicle.mass_kg * vehicle.gravity_m_s2:
            raise ValueError("maximum thrust must exceed weight for takeoff")
        self.vehicle = vehicle
        self.gains = gains

    def command(self, state: VehicleState, target_position_m: np.ndarray, target_yaw_rad: float,
                target_velocity_m_s: np.ndarray | None = None,
                target_acceleration_m_s2: np.ndarray | None = None) -> ControlResult:
        """Use the supplied feedback state; M5 varies only its attitude field."""
        gains = self.gains
        velocity_reference = (np.zeros(3) if target_velocity_m_s is None
                              else np.asarray(target_velocity_m_s, dtype=float))
        acceleration_reference = (np.zeros(3) if target_acceleration_m_s2 is None
                                  else np.asarray(target_acceleration_m_s2, dtype=float))
        if (velocity_reference.shape != (3,) or acceleration_reference.shape != (3,)
                or not np.all(np.isfinite(velocity_reference))
                or not np.all(np.isfinite(acceleration_reference))):
            raise ValueError("trajectory velocity and acceleration must be finite three-vectors")
        acceleration_cmd = (
            gains.position_kp_s2 * (target_position_m - state.position_m)
            + gains.velocity_kd_s * (velocity_reference - state.velocity_m_s)
            + acceleration_reference
        )
        if target_velocity_m_s is not None:
            acceleration_cmd += self.vehicle.linear_drag_n_s_m / self.vehicle.mass_kg * velocity_reference
        specific_force = acceleration_cmd + np.array([0.0, 0.0, self.vehicle.gravity_m_s2])
        specific_force[2] = max(specific_force[2], 0.1 * self.vehicle.gravity_m_s2)

        # Limit tilt by limiting the horizontal-to-vertical force ratio.
        horizontal = float(np.linalg.norm(specific_force[:2]))
        horizontal_limit = specific_force[2] * np.tan(np.deg2rad(gains.max_tilt_deg))
        if horizontal > horizontal_limit:
            specific_force[:2] *= horizontal_limit / horizontal

        cy, sy = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
        forward_force = cy * specific_force[0] + sy * specific_force[1]
        left_force = -sy * specific_force[0] + cy * specific_force[1]
        force_norm = float(np.linalg.norm(specific_force))
        roll = -np.arcsin(np.clip(left_force / force_norm, -1.0, 1.0))
        pitch = np.arctan2(forward_force, specific_force[2])
        desired_euler = np.array([roll, pitch, target_yaw_rad])
        desired_quaternion = euler_to_quaternion(*desired_euler)

        attitude_error = normalize_quaternion(quaternion_product(
            quaternion_conjugate(state.quaternion_bw), desired_quaternion
        ))
        if attitude_error[0] < 0:
            attitude_error = -attitude_error
        desired_rate = np.clip(
            2 * gains.attitude_kp_s * attitude_error[1:],
            -gains.max_body_rate_rad_s,
            gains.max_body_rate_rad_s,
        )
        inertia = self.vehicle.inertia_kg_m2
        rate_error = desired_rate - state.body_rate_rad_s
        torque = inertia * gains.rate_kp_s * rate_error + np.cross(
            state.body_rate_rad_s, inertia * state.body_rate_rad_s
        )
        torque = np.clip(torque, -gains.max_torque_nm, gains.max_torque_nm)
        thrust = float(np.clip(self.vehicle.mass_kg * force_norm, 0.0, gains.max_total_thrust_n))
        return ControlResult(Wrench(thrust, torque), desired_euler)

