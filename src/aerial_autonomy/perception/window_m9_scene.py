"""M9 physical wall openings and independent whole-segment collision checks.

The scene owns simulator truth. Guidance constructs a separate planning scene
from image-derived poses; it never receives this scene or its windows.
"""

from dataclasses import dataclass
import math

import numpy as np

from aerial_autonomy.perception.window_geometry import StaticWindow
from aerial_autonomy.planning.geometry import AABB, COLLISION_TOLERANCE_M, Workspace, segment_aabb_distance, vector3


@dataclass(frozen=True)
class WindowWall:
    window: StaticWindow
    outer_width_m: float
    outer_height_m: float

    def __post_init__(self):
        if (not math.isfinite(self.outer_width_m) or not math.isfinite(self.outer_height_m)
                or self.outer_width_m <= self.window.model.width_m
                or self.outer_height_m <= self.window.model.height_m
                or self.window.frame_thickness_m <= 0):
            raise ValueError("wall needs larger finite outer bounds and positive thickness")

    def local_boxes(self) -> tuple[AABB,...]:
        iw,ih = self.window.model.width_m/2,self.window.model.height_m/2
        ow,oh = self.outer_width_m/2,self.outer_height_m/2
        d = self.window.frame_thickness_m/2
        return (AABB((-ow,-oh,-d),(-iw,oh,d)),
                AABB((iw,-oh,-d),(ow,oh,d)),
                AABB((-iw,-oh,-d),(iw,-ih,d)),
                AABB((-iw,ih,-d),(iw,oh,d)))

    def world_to_local(self, points_world_m):
        points=np.asarray(points_world_m,dtype=float)
        return (points-self.window.center_world_m)@self.window.rotation_window_to_world

    def segment_clearance_m(self,a,b,envelope_m: float) -> float:
        endpoints=self.world_to_local(np.asarray([a,b],dtype=float))
        return min(segment_aabb_distance(endpoints[0],endpoints[1],box)-envelope_m
                   for box in self.local_boxes())

    def plane_coordinate_m(self,point_world_m) -> float:
        return float(self.world_to_local(np.asarray(point_world_m))[2])


@dataclass(frozen=True)
class WindowCourseWorld:
    workspace: Workspace
    walls: tuple[WindowWall,...]
    vehicle_radius_m: float = .18
    clearance_m: float = .07

    def __post_init__(self):
        object.__setattr__(self,"walls",tuple(self.walls))
        if (not self.walls or self.vehicle_radius_m <= 0 or self.clearance_m < 0
                or not all(isinstance(x,WindowWall) for x in self.walls)):
            raise ValueError("invalid window course")

    @property
    def envelope_m(self):
        return self.vehicle_radius_m+self.clearance_m

    def segment_clearance_m(self,a,b):
        p,q=np.asarray(vector3(a,"segment start")),np.asarray(vector3(b,"segment end"))
        e=self.envelope_m
        bounds=[]
        for point in (p,q):
            bounds.extend(point-np.asarray(self.workspace.minimum)-e)
            bounds.extend(np.asarray(self.workspace.maximum)-point-e)
        return float(min([*bounds,*(wall.segment_clearance_m(p,q,e) for wall in self.walls)]))

    def segment_free(self,a,b):
        return self.segment_clearance_m(a,b)>COLLISION_TOLERANCE_M

    def point_free(self,p):
        return self.segment_free(p,p)

    def wall_for_id(self,window_id: str) -> WindowWall:
        matches=[wall for wall in self.walls if wall.window.model.window_id==window_id]
        if len(matches)!=1:
            raise ValueError("unknown or duplicate window identifier")
        return matches[0]
