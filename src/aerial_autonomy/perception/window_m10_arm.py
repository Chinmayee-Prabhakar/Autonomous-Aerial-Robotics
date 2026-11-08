"""Stationary frame with a time-varying capsule in its opening.

Local axes are window right, up, normal. Angle zero points right and positive
angle turns toward up about the fixed window normal. The capsule centerline
runs from the pivot to the tip and has radius ``arm_radius_m``.
"""

from dataclasses import dataclass
import math
import numpy as np

from aerial_autonomy.perception.window_geometry import StaticWindow


@dataclass(frozen=True)
class ArmShape:
    pivot_right_m: float
    pivot_up_m: float
    length_m: float
    arm_radius_m: float

    def __post_init__(self):
        if any(not math.isfinite(x) for x in vars(self).values()):
            raise ValueError("arm geometry must be finite")
        if self.length_m <= 0 or self.arm_radius_m <= 0:
            raise ValueError("arm length and radius must be positive")

    def endpoints_local(self, angle_rad: float) -> tuple[np.ndarray,np.ndarray]:
        pivot=np.array([self.pivot_right_m,self.pivot_up_m,0.],dtype=float)
        tip=pivot+self.length_m*np.array([math.cos(angle_rad),math.sin(angle_rad),0.])
        return pivot,tip


@dataclass(frozen=True)
class RotatingArm:
    window: StaticWindow
    shape: ArmShape
    initial_angle_rad: float
    angular_velocity_rad_s: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.initial_angle_rad,self.angular_velocity_rad_s)):
            raise ValueError("arm motion must be finite")
        model=self.window.model
        pivot,tip=self.shape.endpoints_local(0.)
        if (abs(pivot[0])+self.shape.arm_radius_m>=model.width_m/2 or
                abs(pivot[1])+self.shape.arm_radius_m>=model.height_m/2 or
                self.shape.length_m<=.3):
            raise ValueError("pivot must be inside the opening and arm must be substantial")

    def angle_at(self,time_s: float) -> float:
        if not math.isfinite(time_s) or time_s<0:
            raise ValueError("invalid simulation time")
        return self.initial_angle_rad+self.angular_velocity_rad_s*time_s

    def endpoints_world(self,time_s: float) -> tuple[np.ndarray,np.ndarray]:
        a,b=self.shape.endpoints_local(self.angle_at(time_s))
        rotation=self.window.rotation_window_to_world
        return self.window.center_world_m+rotation@a,self.window.center_world_m+rotation@b


def arm_clearance_m(position_world_m,window: StaticWindow,shape: ArmShape,
                    angle_rad: float,envelope_m: float) -> float:
    """Sphere-to-capsule clearance; positive means separated."""
    local=window.opening_coordinates(np.asarray(position_world_m,dtype=float))
    pivot,tip=shape.endpoints_local(angle_rad)
    direction=tip-pivot
    u=float(np.clip(np.dot(local-pivot,direction)/np.dot(direction,direction),0.,1.))
    return float(np.linalg.norm(local-(pivot+u*direction))-shape.arm_radius_m-envelope_m)


def swept_arm_clearance_m(a_world_m,b_world_m,t0_s: float,t1_s: float,
                          arm: RotatingArm,envelope_m: float,
                          resolution_m: float=.004) -> float:
    """Conservative lower bound over one linearly interpolated plant step.

    Sample subinterval ends and subtract half of a Lipschitz motion bound.
    Maximum excess conservatism is ``resolution_m / 2`` when the subdivision
    count is chosen from combined drone displacement and arm-tip travel.
    The plant's continuous motion inside an integration step is approximated
    by a line; this does not bound numerical integration error itself.
    """
    if t1_s<t0_s or resolution_m<=0:
        raise ValueError("invalid swept interval")
    a=np.asarray(a_world_m,dtype=float); b=np.asarray(b_world_m,dtype=float)
    motion=float(np.linalg.norm(b-a))+abs(arm.angular_velocity_rad_s)*(t1_s-t0_s)*arm.shape.length_m
    count=max(1,int(math.ceil(motion/resolution_m)))
    bound=motion/(2*count)
    values=[arm_clearance_m(a+(b-a)*(i/count),arm.window,arm.shape,
                            arm.angle_at(t0_s+(t1_s-t0_s)*(i/count)),envelope_m)
            for i in range(count+1)]
    return min(values)-bound
