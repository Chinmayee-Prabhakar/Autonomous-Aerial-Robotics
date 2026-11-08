"""M10 truth-to-RGB rendering and pixel-only rotating-arm extraction."""

from dataclasses import dataclass
import math
import numpy as np

from aerial_autonomy.perception.window_m9_render import (WindowImageRenderer,
    RenderSettings,_fill_convex,_project_local_polygon)
from aerial_autonomy.perception.window_m10_arm import RotatingArm
from aerial_autonomy.perception.window_m9_guidance import ImagePoseFix
from aerial_autonomy.simulation.dynamics import VehicleState


class RotatingArmRenderer:
    """Simulator boundary: receives truth and returns only RGB image pixels."""

    def __init__(self,base: WindowImageRenderer,arm: RotatingArm):
        self.base=base
        self.arm=arm
        self.wall=base.world.wall_for_id(arm.window.model.window_id)

    def render(self,state: VehicleState,time_s: float,settings: RenderSettings,
               rng: np.random.Generator) -> np.ndarray:
        clean=RenderSettings(0.,settings.illumination,False,settings.near_m)
        image=self.base.render(state,clean,rng)
        p,q=self.arm.shape.endpoints_local(self.arm.angle_at(time_s))
        tangent=(q-p)/self.arm.shape.length_m
        across=np.array([-tangent[1],tangent[0],0.])*self.arm.shape.arm_radius_m
        # The physical capsule has circular ends; the RGB renderer uses a
        # rectangular shaft and a visible magenta hub as appearance cues.
        z=.025
        rect=np.array([p-across,q-across,q+across,p+across]);rect[:,2]=z
        uv=_project_local_polygon(self.wall,rect,self.base.camera,state,settings.near_m)
        if uv is not None:
            _fill_convex(image,uv,np.array([248,204,33],dtype=np.uint8))
        r=.065
        hub=np.array([[p[0]-r,p[1]-r,z+.002],[p[0]+r,p[1]-r,z+.002],
                      [p[0]+r,p[1]+r,z+.002],[p[0]-r,p[1]+r,z+.002]])
        uv=_project_local_polygon(self.wall,hub,self.base.camera,state,settings.near_m)
        if uv is not None:
            _fill_convex(image,uv,np.array([246,39,233],dtype=np.uint8))
        if settings.blur:
            h,w=image.shape[:2]
            padded=np.pad(image.astype(np.float32),((1,1),(1,1),(0,0)),mode="edge")
            image=np.clip(sum(padded[dy:dy+h,dx:dx+w] for dy in range(3)
                              for dx in range(3))/9,0,255).astype(np.uint8)
        if settings.noise_std_rgb:
            image=np.clip(image.astype(float)+rng.normal(0,settings.noise_std_rgb,image.shape),
                          0,255).astype(np.uint8)
        return image


@dataclass(frozen=True)
class ArmDetection:
    valid: bool
    timestamp_s: float
    angle_rad: float | None
    pivot_local_m: tuple[float,float] | None
    tip_local_m: tuple[float,float] | None
    confidence: float
    reason: str | None
    shaft_pixel_count: int


def _pixel_to_local(pixel_uv: np.ndarray,corners_uv: np.ndarray,width_m: float,
                    height_m: float) -> np.ndarray:
    """Invert a four-corner projective map, without simulator projections."""
    local=np.array([[-width_m/2,-height_m/2],[width_m/2,-height_m/2],
                    [width_m/2,height_m/2],[-width_m/2,height_m/2]])
    rows=[]; rhs=[]
    for (x,y),(u,v) in zip(local,corners_uv):
        rows.extend(([x,y,1,0,0,0,-u*x,-u*y],
                     [0,0,0,x,y,1,-v*x,-v*y]));rhs.extend((u,v))
    h=np.linalg.solve(np.asarray(rows),np.asarray(rhs))
    H=np.array([[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],1.]])
    inverse=np.linalg.inv(H)
    homogeneous=np.column_stack((pixel_uv,np.ones(len(pixel_uv))))@inverse.T
    return homogeneous[:,:2]/homogeneous[:,2,None]


class PixelArmDetector:
    """Known yellow shaft/magenta hub appearance; no arm truth input."""

    def detect(self,image_rgb: np.ndarray,fix: ImagePoseFix,timestamp_s: float) -> ArmDetection:
        def failed(reason,n=0):
            return ArmDetection(False,float(timestamp_s),None,None,None,0.,reason,n)
        if not fix.valid or fix.detection.corners_uv is None:
            return failed("window_pose_unavailable")
        values=np.asarray(image_rgb,dtype=np.int16)
        if values.ndim!=3 or values.shape[2]!=3:
            return failed("invalid_image")
        shaft=(values[:,:,0]>160)&(values[:,:,1]>125)&(values[:,:,2]<115)
        hub=(values[:,:,0]>155)&(values[:,:,2]>140)&(values[:,:,1]<115)
        yy,xx=np.nonzero(shaft); hy,hx=np.nonzero(hub)
        if len(xx)<9 or len(hx)<3:
            return failed("arm_or_hub_missing",len(xx))
        # Use only pixels within the image-detected opening quadrilateral.
        width=fix.estimated_window.model.width_m
        height=fix.estimated_window.model.height_m
        try:
            points=_pixel_to_local(np.column_stack((xx+.5,yy+.5)),
                                   fix.detection.corners_uv,width,height)
            hub_points=_pixel_to_local(np.column_stack((hx+.5,hy+.5)),
                                       fix.detection.corners_uv,width,height)
        except np.linalg.LinAlgError:
            return failed("degenerate_window_homography",len(xx))
        in_open=(abs(points[:,0])<width/2+.08)&(abs(points[:,1])<height/2+.08)
        in_hub=(abs(hub_points[:,0])<width/2+.08)&(abs(hub_points[:,1])<height/2+.08)
        points=points[in_open];hub_points=hub_points[in_hub]
        if len(points)<8 or len(hub_points)<3:
            return failed("arm_outside_opening",len(points))
        pivot=np.median(hub_points,axis=0)
        vectors=points-pivot
        lengths=np.linalg.norm(vectors,axis=1)
        selected=vectors[lengths>=np.quantile(lengths,.75)]
        tip=np.median(selected,axis=0)
        length=float(np.linalg.norm(tip))
        if not .32<length<1.5:
            return failed("implausible_extent",len(points))
        angle=float(math.atan2(tip[1],tip[0]))
        confidence=float(min(1.,len(points)/55)*min(1.,len(hub_points)/8))
        return ArmDetection(True,float(timestamp_s),angle,tuple(map(float,pivot)),
                            tuple(map(float,pivot+tip)),confidence,None,len(points))
