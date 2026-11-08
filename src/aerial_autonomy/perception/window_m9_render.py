"""Deterministic RGB wall renderer; simulator-only truth-to-image boundary."""

from dataclasses import dataclass

import numpy as np

from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_m9_scene import WindowCourseWorld, WindowWall
from aerial_autonomy.simulation.dynamics import VehicleState


@dataclass(frozen=True)
class RenderSettings:
    noise_std_rgb: float = 0.
    illumination: float = 1.
    blur: bool = False
    near_m: float = .05

    def __post_init__(self):
        if (not np.isfinite(self.noise_std_rgb) or self.noise_std_rgb < 0
                or not np.isfinite(self.illumination) or not .4 <= self.illumination <= 1.5
                or self.near_m <= 0):
            raise ValueError("invalid rendering settings")


def _clip_near(points: np.ndarray,near_m: float) -> np.ndarray:
    """Clip a convex camera-space polygon against the optical near plane."""
    result=[]
    for previous,current in zip(points,np.roll(points,-1,axis=0)):
        inside_a=previous[2]>=near_m
        inside_b=current[2]>=near_m
        if inside_a:
            result.append(previous)
        if inside_a != inside_b:
            fraction=(near_m-previous[2])/(current[2]-previous[2])
            result.append(previous+fraction*(current-previous))
    return np.asarray(result,dtype=float).reshape((-1,3))


def _fill_convex(image: np.ndarray,vertices_uv: np.ndarray,color: np.ndarray):
    if len(vertices_uv)<3 or not np.all(np.isfinite(vertices_uv)):
        return
    h,w=image.shape[:2]
    x0=max(0,int(np.floor(np.min(vertices_uv[:,0]))))
    x1=min(w,int(np.ceil(np.max(vertices_uv[:,0])))+1)
    y0=max(0,int(np.floor(np.min(vertices_uv[:,1]))))
    y1=min(h,int(np.ceil(np.max(vertices_uv[:,1])))+1)
    if x0>=x1 or y0>=y1:
        return
    xx,yy=np.meshgrid(np.arange(x0,x1)+.5,np.arange(y0,y1)+.5)
    area=np.sum(vertices_uv[:,0]*np.roll(vertices_uv[:,1],-1)-
                vertices_uv[:,1]*np.roll(vertices_uv[:,0],-1))
    sign=1 if area>=0 else -1
    mask=np.ones(xx.shape,dtype=bool)
    for a,b in zip(vertices_uv,np.roll(vertices_uv,-1,axis=0)):
        cross=(b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0])
        mask &= cross*sign>=-1e-7
    image[y0:y1,x0:x1][mask]=color


def _project_local_polygon(wall: WindowWall,local: np.ndarray,camera: CameraCalibration,
                           state: VehicleState,near_m: float):
    world=wall.window.center_world_m+local@wall.window.rotation_window_to_world.T
    camera_points=camera.world_to_camera(state,world)
    camera_points=_clip_near(camera_points,near_m)
    if len(camera_points)<3:
        return None
    return np.column_stack((camera.fx_px*camera_points[:,0]/camera_points[:,2]+camera.cx_px,
                            camera.fy_px*camera_points[:,1]/camera_points[:,2]+camera.cy_px))


class WindowImageRenderer:
    """Owns true walls; callers receive only RGB pixels."""

    def __init__(self,camera: CameraCalibration,world: WindowCourseWorld,
                 colors_by_id: dict[str,tuple[int,int,int]]):
        self.camera=camera
        self.world=world
        self.colors_by_id=dict(colors_by_id)
        if set(self.colors_by_id)!={w.window.model.window_id for w in world.walls}:
            raise ValueError("one distinctive color is required per true wall")

    def render(self,state: VehicleState,settings: RenderSettings,
               rng: np.random.Generator) -> np.ndarray:
        h,w=self.camera.height_px,self.camera.width_px
        yy=np.arange(h,dtype=float)[:,None]
        image=np.empty((h,w,3),dtype=np.uint8)
        # Low-contrast procedural background, independent of feature geometry.
        image[:,:,0]=np.clip(31+yy*.08,0,255).astype(np.uint8)
        image[:,:,1]=np.clip(43+yy*.06,0,255).astype(np.uint8)
        image[:,:,2]=np.clip(58+yy*.04,0,255).astype(np.uint8)
        centers=np.array([wall.window.center_world_m for wall in self.world.walls])
        optical_z=self.camera.world_to_camera(state,centers)[:,2]
        for index in np.argsort(optical_z)[::-1]:
            wall=self.world.walls[int(index)]
            center_local=wall.world_to_local(self.camera.camera_to_world(state)[1])
            face=wall.window.frame_thickness_m/2*(1 if center_local[2]>=0 else -1)
            base=np.asarray(self.colors_by_id[wall.window.model.window_id],dtype=float)
            # Recessed opening reveals darker thickness faces behind the visible plane.
            iw,ih=wall.window.model.width_m/2,wall.window.model.height_m/2
            d=wall.window.frame_thickness_m/2
            recess=np.clip(base*.38,0,255).astype(np.uint8)
            inner_faces=(np.array([[-iw,-ih,-d],[-iw,ih,-d],[-iw,ih,d],[-iw,-ih,d]]),
                         np.array([[iw,-ih,-d],[iw,ih,-d],[iw,ih,d],[iw,-ih,d]]),
                         np.array([[-iw,-ih,-d],[iw,-ih,-d],[iw,-ih,d],[-iw,-ih,d]]),
                         np.array([[-iw,ih,-d],[iw,ih,-d],[iw,ih,d],[-iw,ih,d]]))
            for polygon in inner_faces:
                uv=_project_local_polygon(wall,polygon,self.camera,state,settings.near_m)
                if uv is not None:
                    _fill_convex(image,uv,recess)
            wall_color=np.clip(np.array([105.,113.,122.])*settings.illumination,0,255).astype(np.uint8)
            for box in wall.local_boxes():
                lo,hi=np.asarray(box.minimum),np.asarray(box.maximum)
                polygon=np.array([[lo[0],lo[1],face],[hi[0],lo[1],face],
                                  [hi[0],hi[1],face],[lo[0],hi[1],face]])
                uv=_project_local_polygon(wall,polygon,self.camera,state,settings.near_m)
                if uv is not None:
                    _fill_convex(image,uv,wall_color)
            rim=.22
            bands=(((-iw-rim,-ih-rim),(-iw,ih+rim)),
                   ((iw,-ih-rim),(iw+rim,ih+rim)),
                   ((-iw,-ih-rim),(iw,-ih)),
                   ((-iw,ih),(iw,ih+rim)))
            color=np.clip(base*settings.illumination,0,255).astype(np.uint8)
            for (x0,y0),(x1,y1) in bands:
                polygon=np.array([[x0,y0,face],[x1,y0,face],[x1,y1,face],[x0,y1,face]])
                uv=_project_local_polygon(wall,polygon,self.camera,state,settings.near_m)
                if uv is not None:
                    _fill_convex(image,uv,color)
        if settings.blur:
            padded=np.pad(image.astype(np.float32),((1,1),(1,1),(0,0)),mode="edge")
            smoothed=sum(padded[dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3))/9
            image=np.clip(smoothed,0,255).astype(np.uint8)
        if settings.noise_std_rgb:
            image=np.clip(image.astype(np.float32)+
                          rng.normal(0,settings.noise_std_rgb,image.shape),0,255).astype(np.uint8)
        return image
