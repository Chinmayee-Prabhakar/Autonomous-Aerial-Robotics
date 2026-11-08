"""Whole-segment collision geometry for a spherical vehicle envelope.

Obstacle contact within 1e-9 m is collision. Workspace contact is rejected
with the same tolerance. The spherical envelope is a geometric safety model,
not a full drone-body or rotor model. Coordinates are ENU metres.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


COLLISION_TOLERANCE_M = 1e-9


def vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be three finite coordinates")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class Workspace:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum", vector3(self.minimum, "workspace minimum"))
        object.__setattr__(self, "maximum", vector3(self.maximum, "workspace maximum"))
        if any(lo >= hi for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("workspace maximum must exceed minimum on every axis")


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", vector3(self.center, "sphere center"))
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("sphere radius must be finite and positive")


@dataclass(frozen=True)
class AABB:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum", vector3(self.minimum, "box minimum"))
        object.__setattr__(self, "maximum", vector3(self.maximum, "box maximum"))
        if any(lo >= hi for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("box maximum must exceed minimum on every axis")


Obstacle = Sphere | AABB


def segment_sphere_distance(a: Sequence[float], b: Sequence[float], center: Sequence[float]) -> float:
    """Minimum Euclidean distance from a closed segment to a point."""
    p, q, c = (np.asarray(vector3(x, "segment/center")) for x in (a, b, center))
    d = q - p
    squared = float(d @ d)
    t = 0.0 if squared == 0 else float(np.clip(((c - p) @ d) / squared, 0.0, 1.0))
    return float(np.linalg.norm(p + t * d - c))


def segment_aabb_distance(a: Sequence[float], b: Sequence[float], box: AABB) -> float:
    """Exact segment-to-AABB distance via piecewise quadratic intervals.

    Each axis changes nearest face only where the segment crosses a box plane.
    Between sorted crossings, squared distance is quadratic. Its minimum is
    at an interval endpoint or its stationary point.
    """
    p = np.asarray(vector3(a, "segment start"))
    d = np.asarray(vector3(b, "segment end")) - p
    lo, hi = np.asarray(box.minimum), np.asarray(box.maximum)
    breaks = [0.0, 1.0]
    for axis in range(3):
        if d[axis] != 0.0:
            for face in (lo[axis], hi[axis]):
                t = float((face - p[axis]) / d[axis])
                if 0.0 < t < 1.0:
                    breaks.append(t)
    breaks = sorted(set(breaks))

    def squared_distance(t: float) -> float:
        point = p + t * d
        gap = np.maximum(lo - point, 0.0) + np.maximum(point - hi, 0.0)
        return float(gap @ gap)

    best = math.inf
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = (left + right) / 2.0
        point = p + middle * d
        active = (point < lo) | (point > hi)
        face = np.where(point < lo, lo, hi)
        slope = np.where(active, d, 0.0)
        offset = np.where(active, p - face, 0.0)
        denom = float(slope @ slope)
        candidates = [left, right]
        if denom > 0:
            candidates.append(float(np.clip(-(slope @ offset) / denom, left, right)))
        best = min(best, *(squared_distance(t) for t in candidates))
    return math.sqrt(max(best, 0.0))


@dataclass(frozen=True)
class World:
    workspace: Workspace
    obstacles: tuple[Obstacle, ...]
    vehicle_radius_m: float
    clearance_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            raise ValueError("workspace must be a Workspace")
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        if any(not isinstance(item, (Sphere, AABB)) for item in self.obstacles):
            raise ValueError("obstacles must be spheres or AABBs")
        if not math.isfinite(self.vehicle_radius_m) or self.vehicle_radius_m <= 0:
            raise ValueError("vehicle radius must be finite and positive")
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0:
            raise ValueError("clearance must be finite and nonnegative")
        if any(hi - lo <= 2 * (self.envelope_m + COLLISION_TOLERANCE_M)
               for lo, hi in zip(self.workspace.minimum, self.workspace.maximum)):
            raise ValueError("workspace has no vehicle-center interior")

    @property
    def envelope_m(self) -> float:
        return self.vehicle_radius_m + self.clearance_m

    def segment_clearance_m(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Exact minimum signed clearance to obstacles and workspace walls."""
        p, q = vector3(a, "segment start"), vector3(b, "segment end")
        envelope = self.envelope_m
        values = [coord - lo - envelope for point in (p, q)
                  for coord, lo in zip(point, self.workspace.minimum)]
        values += [hi - coord - envelope for point in (p, q)
                   for coord, hi in zip(point, self.workspace.maximum)]
        for obstacle in self.obstacles:
            if isinstance(obstacle, Sphere):
                values.append(segment_sphere_distance(p, q, obstacle.center)
                              - obstacle.radius_m - envelope)
            else:
                values.append(segment_aabb_distance(p, q, obstacle) - envelope)
        return min(values)

    def segment_free(self, a: Sequence[float], b: Sequence[float]) -> bool:
        return self.segment_clearance_m(a, b) > COLLISION_TOLERANCE_M

    def point_free(self, point: Sequence[float]) -> bool:
        return self.segment_free(point, point)
