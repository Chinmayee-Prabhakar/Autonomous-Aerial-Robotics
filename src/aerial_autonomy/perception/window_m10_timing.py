"""Causal angle tracking and conservative timed-crossing selection."""

from dataclasses import dataclass
import math
import numpy as np

from aerial_autonomy.perception.window_m10_arm import ArmShape, arm_clearance_m
from aerial_autonomy.perception.window_m10_vision import ArmDetection


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class MotionEstimate:
    valid: bool
    timestamp_s: float
    angle_rad: float | None
    angular_velocity_rad_s: float | None
    angle_uncertainty_rad: float
    rate_uncertainty_rad_s: float
    observation_age_s: float
    reason: str | None
    observation_count: int

    def predict(self,time_s: float) -> tuple[float,float]:
        if not self.valid or time_s<self.timestamp_s:
            raise ValueError("invalid motion prediction")
        dt=time_s-self.timestamp_s
        return (self.angle_rad+self.angular_velocity_rad_s*dt,
                self.angle_uncertainty_rad+self.rate_uncertainty_rad_s*dt)


class AngularTracker:
    def __init__(self,max_age_s: float=.7,max_rate_rad_s: float=1.5):
        self.max_age_s=max_age_s
        self.max_rate_rad_s=max_rate_rad_s
        self.samples=[]
        self.misses=0

    def update(self,detection: ArmDetection):
        if not detection.valid:
            self.misses+=1
            return
        if self.samples and detection.timestamp_s<=self.samples[-1][0]:
            raise ValueError("arm timestamps must increase")
        angle=detection.angle_rad
        if self.samples:
            previous=self.samples[-1][1]
            angle=previous+wrap_angle(angle-wrap_angle(previous))
        self.samples.append((detection.timestamp_s,angle,detection.confidence))
        self.samples=self.samples[-7:]

    def estimate(self,now_s: float) -> MotionEstimate:
        if not self.samples:
            return MotionEstimate(False,now_s,None,None,math.inf,math.inf,math.inf,
                                  "no_arm_observations",0)
        age=now_s-self.samples[-1][0]
        if age<0:
            raise ValueError("query precedes observation")
        if len(self.samples)<3:
            return MotionEstimate(False,self.samples[-1][0],None,None,math.inf,math.inf,
                                  age,"insufficient_observations",len(self.samples))
        if age>self.max_age_s:
            return MotionEstimate(False,self.samples[-1][0],None,None,math.inf,math.inf,
                                  age,"stale_observations",len(self.samples))
        t=np.array([x[0] for x in self.samples]);a=np.array([x[1] for x in self.samples])
        weights=np.array([max(.15,x[2]) for x in self.samples])
        origin=t[-1]
        design=np.column_stack((np.ones(len(t)),t-origin))
        fit=np.linalg.lstsq(design*np.sqrt(weights[:,None]),a*np.sqrt(weights),rcond=None)[0]
        residual=float(np.sqrt(np.mean((a-design@fit)**2)))
        rate=float(fit[1])
        if abs(rate)>self.max_rate_rad_s:
            return MotionEstimate(False,origin,None,None,math.inf,math.inf,age,
                                  "implausible_rate",len(t))
        span=t[-1]-t[0]
        angle_uncertainty=max(.045,3*residual+.02)
        rate_uncertainty=max(.025,3*residual/max(span,.1)+.015)
        return MotionEstimate(True,origin,float(fit[0]),rate,angle_uncertainty,
                              rate_uncertainty,age,None,len(t))


@dataclass(frozen=True)
class CrossingChoice:
    valid: bool
    commit_time_s: float | None
    predicted_plane_time_s: float | None
    predicted_angle_rad: float | None
    predicted_min_clearance_m: float | None
    reason: str


def reference_dynamic_clearance(trajectory,commit_time_s: float,estimate: MotionEstimate,
                                window,shape: ArmShape,envelope_m: float,
                                sample_s: float=.02) -> float:
    if not estimate.valid:
        return -math.inf
    count=max(1,math.ceil(trajectory.duration_s/sample_s))
    if len(trajectory.waypoints_m)==2:
        u=np.linspace(0.,1.,count+1)
        dt=u*trajectory.duration_s
        start=np.asarray(trajectory.waypoints_m[0]);end=np.asarray(trajectory.waypoints_m[1])
        s=10*u**3-15*u**4+6*u**5
        positions=start[None,:]+s[:,None]*(end-start)[None,:]
        local=(positions-window.center_world_m)@window.rotation_window_to_world
        elapsed=commit_time_s+dt-estimate.timestamp_s
        angles=estimate.angle_rad+estimate.angular_velocity_rad_s*elapsed
        uncertainty=estimate.angle_uncertainty_rad+estimate.rate_uncertainty_rad_s*elapsed
        pivot=np.array([shape.pivot_right_m,shape.pivot_up_m,0.])
        direction=shape.length_m*np.column_stack((np.cos(angles),np.sin(angles),
                                                   np.zeros(len(angles))))
        fraction=np.clip(np.sum((local-pivot)*direction,axis=1)/shape.length_m**2,0,1)
        nearest=pivot[None,:]+fraction[:,None]*direction
        best=float(np.min(np.linalg.norm(local-nearest,axis=1)-shape.arm_radius_m-
                          envelope_m-shape.length_m*uncertainty))
    else:
        best=math.inf
        for k in range(count+1):
            dt=trajectory.duration_s*k/count
            position=trajectory.evaluate(dt).position_m
            angle,uncertainty=estimate.predict(commit_time_s+dt)
            best=min(best,arm_clearance_m(position,window,shape,angle,
                         envelope_m+shape.length_m*uncertainty))
    # Lipschitz allowance between adjacent reference samples. The M7 speed
    # bound is certified separately; angle rate is bounded by tracker policy.
    return best-(1.25+shape.length_m*1.5)*trajectory.duration_s/count/2


def choose_crossing(trajectory,now_s: float,estimate: MotionEstimate,window,
                    shape: ArmShape,envelope_m: float,horizon_s: float=12.,
                    min_clearance_m: float=.035) -> CrossingChoice:
    if not estimate.valid:
        return CrossingChoice(False,None,None,None,None,"motion_unavailable")
    # Select the earliest safe launch. Scheduling never queries the true arm.
    for delay in np.arange(0,horizon_s+.001,.1):
        launch=now_s+float(delay)
        clearance=reference_dynamic_clearance(trajectory,launch,estimate,window,
                                               shape,envelope_m)
        if clearance>min_clearance_m:
            a=np.asarray(trajectory.waypoints_m[0]);b=np.asarray(trajectory.waypoints_m[-1])
            plane_fraction=float(np.clip(np.dot(window.center_world_m-a,b-a)/
                                         np.dot(b-a,b-a),0,1))
            # Monotone quintic position: invert normalized displacement.
            lo,hi=0.,1.
            for _ in range(25):
                mid=(lo+hi)/2
                s=10*mid**3-15*mid**4+6*mid**5
                if s<plane_fraction:lo=mid
                else:hi=mid
            plane_time=launch+(lo+hi)/2*trajectory.duration_s
            return CrossingChoice(True,launch,plane_time,
                                  float(wrap_angle(estimate.predict(plane_time)[0])),
                                  float(clearance),"safe_interval")
    return CrossingChoice(False,None,None,None,None,"no_safe_interval")
