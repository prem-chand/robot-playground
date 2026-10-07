"""A scenario is plain data: what to track, what to push, how to scale the world,
and what a seed may perturb. Every model edit lives in `robot_pg.robots`, which
imports this package, never the reverse.

References evaluate on integer ticks, so the reference a run publishes and the
hold boundaries the Panda fail rule reads come from the same arithmetic.
"""

import importlib
import itertools
import math
import pkgutil
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from robot_pg.contract import Array, SignalSpec

Ticks = NDArray[np.int64]
Vec3 = tuple[float, float, float]

PATH_POINTS = 256


def whole_ticks(seconds: float, dt: float) -> int:
    ticks = round(seconds / dt)
    if abs(ticks * dt - seconds) > 1e-9:
        raise ValueError(f"{seconds:g} s is not a whole number of {dt:g} s ticks")
    return ticks


@dataclass(frozen=True)
class Arc:
    """Constant curvature. 0 is a straight line, and a positive value turns left."""

    length: float
    curvature: float


@dataclass(frozen=True)
class Path:
    """A chain of arcs from the origin heading +x, followed at `speed` by arc
    length. The nominal point stops at the end, and the path does not loop.
    `ref_path` is the geometry, constant in time, as `PATH_POINTS` points equally
    spaced in arc length.
    """

    arcs: tuple[Arc, ...]
    speed: float
    kind: ClassVar[str] = "path"
    signal: ClassVar[SignalSpec] = SignalSpec("ref_path", (PATH_POINTS, 2), "m")

    @classmethod
    def figure8(cls, radius: float, speed: float) -> "Path":
        """Two full circles that touch at the origin, both tangent to +x there.
        The first turns left around (0, r), the second right around (0, -r)."""
        lap = 2 * math.pi * radius
        return cls((Arc(lap, 1 / radius), Arc(lap, -1 / radius)), speed)

    @property
    def length(self) -> float:
        return sum(a.length for a in self.arcs)

    def _starts(self) -> list[tuple[float, float, float, float]]:
        """(s, x, y, heading) at the start of each arc."""
        s, x, y, th = 0.0, 0.0, 0.0, 0.0
        starts = []
        for arc in self.arcs:
            starts.append((s, x, y, th))
            end = _arc_pose(x, y, th, arc.curvature, np.array(arc.length))
            x, y, th = (float(v) for v in end)
            s += arc.length
        return starts

    def pose(self, s: Array) -> Array:
        """(..., 3) x, y, heading at arc length s, clamped to the path."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        out = np.empty((*s.shape, 3))
        for arc, (s0, x0, y0, th0) in zip(self.arcs, self._starts(), strict=True):
            on = s >= s0
            out[on] = np.stack(_arc_pose(x0, y0, th0, arc.curvature, s[on] - s0), -1)
        return out

    def polyline(self) -> Array:
        return self.pose(np.linspace(0.0, self.length, PATH_POINTS))[:, :2]

    def nearest(self, xy: Array) -> tuple[Array, Array]:
        """For each point and each arc, the nearest arc length on that arc and the
        distance to it, both (T, len(arcs)). The figure-8 crosses itself at the
        origin, so a progress measure picks per row the candidate nearest its
        previous arc length.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        s_out = np.empty((len(xy), len(self.arcs)))
        d_out = np.empty_like(s_out)
        for i, (arc, (s0, x0, y0, th0)) in enumerate(
            zip(self.arcs, self._starts(), strict=True)
        ):
            p0, k = np.array([x0, y0]), arc.curvature
            if k == 0:
                t0 = np.array([math.cos(th0), math.sin(th0)])
                u = np.clip((xy - p0) @ t0, 0.0, arc.length)
                d = np.linalg.norm(xy - (p0 + u[:, None] * t0), axis=1)
            else:
                rho = 1 / abs(k)
                c = p0 + np.array([-math.sin(th0), math.cos(th0)]) / k
                r = xy - c
                th = np.arctan2(r[:, 1], r[:, 0]) + math.copysign(math.pi / 2, k)
                u = np.mod((th - th0) * math.copysign(1.0, k), 2 * math.pi) * rho
                d = np.abs(np.linalg.norm(r, axis=1) - rho)
                beyond = u > arc.length
                if beyond.any():
                    ends = np.stack(
                        [
                            p0,
                            np.array(
                                _arc_pose(x0, y0, th0, k, np.array(arc.length))[:2]
                            ),
                        ]
                    )
                    to_ends = np.linalg.norm(xy[beyond, None, :] - ends, axis=2)
                    nearer = to_ends.argmin(axis=1)
                    u[beyond] = nearer * arc.length
                    d[beyond] = to_ends.min(axis=1)
            s_out[:, i], d_out[:, i] = s0 + u, d
        return s_out, d_out

    def cross_track(self, xy: Array) -> Array:
        return self.nearest(xy)[1].min(axis=1)

    def values(self, ticks: Ticks, dt: float) -> Array:
        return np.broadcast_to(self.polyline(), (len(ticks), PATH_POINTS, 2))


def _arc_pose(
    x0: float, y0: float, th0: float, k: float, u: Array
) -> tuple[Array, Array, Array]:
    if k == 0:
        return x0 + u * math.cos(th0), y0 + u * math.sin(th0), th0 + 0 * u
    th = th0 + k * u
    return (
        x0 + (np.sin(th) - math.sin(th0)) / k,
        y0 - (np.cos(th) - math.cos(th0)) / k,
        th,
    )


@dataclass(frozen=True)
class Velocity:
    """Forward base speed, piecewise linear through the `(t, v)` knots. Two knots
    at one time make a step, and the later one holds from that time on. Before
    the first knot and after the last the end value holds.
    """

    knots: tuple[tuple[float, float], ...]
    kind: ClassVar[str] = "velocity"
    signal: ClassVar[SignalSpec] = SignalSpec("ref_base_vel", (1,), "m/s")

    def __post_init__(self) -> None:
        times = [t for t, _ in self.knots]
        if not times or times != sorted(times) or not np.isfinite(self.knots).all():
            raise ValueError("velocity knots need finite, non-decreasing times")
        if any(a == b == c for a, b, c in zip(times, times[1:], times[2:])):
            raise ValueError("at most two velocity knots may share a time")

    @classmethod
    def constant(cls, v: float) -> "Velocity":
        return cls(((0.0, v),))

    def values(self, ticks: Ticks, dt: float) -> Array:
        k = np.asarray(self.knots, dtype=np.float64)
        t = np.asarray(ticks) * dt
        i = np.clip(np.searchsorted(k[:, 0], t, side="right") - 1, 0, len(k) - 1)
        j = np.minimum(i + 1, len(k) - 1)
        span = k[j, 0] - k[i, 0]
        w = np.where(span > 0, (t - k[i, 0]) / np.where(span > 0, span, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)
        return (k[i, 1] + w * (k[j, 1] - k[i, 1]))[:, None]


@dataclass(frozen=True)
class Targets:
    """End-effector targets in the base frame, each held `hold_s` in order. After
    the last hold the last target stays."""

    points: tuple[Vec3, ...]
    hold_s: float = 2.0
    kind: ClassVar[str] = "targets"
    signal: ClassVar[SignalSpec] = SignalSpec("ref_ee_target", (3,), "m")

    @classmethod
    def grid(
        cls,
        xs: tuple[float, ...],
        ys: tuple[float, ...],
        zs: tuple[float, ...],
        hold_s: float = 2.0,
    ) -> "Targets":
        """Product order, x slowest and z fastest."""
        return cls(tuple(itertools.product(xs, ys, zs)), hold_s)

    def permuted(self, order: tuple[int, ...]) -> "Targets":
        return Targets(tuple(self.points[i] for i in order), self.hold_s)

    def hold_of(self, ticks: Ticks, dt: float) -> NDArray[np.int64]:
        return np.minimum(
            np.asarray(ticks) // whole_ticks(self.hold_s, dt), len(self.points) - 1
        )

    def hold_ends(self, dt: float, n_ticks: int) -> NDArray[np.int64]:
        """The last tick of each hold that ends inside a run of `n_ticks` ticks."""
        h = whole_ticks(self.hold_s, dt)
        ends = np.arange(1, len(self.points) + 1) * h - 1
        return ends[ends < n_ticks]

    def values(self, ticks: Ticks, dt: float) -> Array:
        return np.asarray(self.points, dtype=np.float64)[self.hold_of(ticks, dt)]


Reference = Path | Velocity | Targets
_KINDS: Mapping[str, type[Path] | type[Velocity] | type[Targets]] = {
    c.kind: c for c in (Path, Velocity, Targets)
}


@dataclass(frozen=True)
class Push:
    """A world-frame force on the robot's base body at its center of mass, for
    the ticks from `at_s` to `at_s + duration_s`."""

    at_s: float
    duration_s: float
    force_n: Vec3

    def ticks(self, dt: float) -> range:
        start = whole_ticks(self.at_s, dt)
        return range(start, start + whole_ticks(self.duration_s, dt))


@dataclass(frozen=True)
class SeedSpace:
    """What a nonzero seed may draw. The base moves only in x, y, and yaw, so no
    seed starts a foot or a wheel inside the floor. `base_m` bounds the planar
    offset, with dx and dy each within `base_m / sqrt(2)`.
    """

    base_m: float = 0.05
    base_yaw_rad: float = math.radians(5.0)
    joint_rad: float = 0.05
    friction: tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class Scenario:
    """`name` is `<robot>.<rest>`, defined in `scenarios/<robot>_<rest>.py`.

    `friction_scale` multiplies every geom friction, and the seed's friction
    factor multiplies it again. `payload_kg` adds to the mass of the robot's
    payload body and leaves its inertia unchanged.
    """

    name: str
    duration: float
    reference: Reference
    pushes: tuple[Push, ...] = ()
    friction_scale: float = 1.0
    payload_kg: float = 0.0
    seed_space: SeedSpace = field(default_factory=SeedSpace)

    def __post_init__(self) -> None:
        robot, _, rest = self.name.partition(".")
        if not robot or not rest:
            raise ValueError(f"scenario name {self.name!r} is not <robot>.<name>")
        if not self.duration > 0:
            raise ValueError(f"scenario {self.name}: duration must be above 0")
        if not (math.isfinite(self.friction_scale) and self.friction_scale > 0):
            raise ValueError(f"scenario {self.name}: friction_scale must be above 0")
        if not (math.isfinite(self.payload_kg) and self.payload_kg >= 0):
            raise ValueError(f"scenario {self.name}: payload_kg must be at least 0")

    @property
    def robot(self) -> str:
        return self.name.partition(".")[0]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, with the reference kind as its tag."""
        d = asdict(self)
        d["reference"] = {"kind": self.reference.kind, **asdict(self.reference)}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scenario":
        """The inverse of `to_dict`. An override of any field is
        `from_dict({**s.to_dict(), field: value})`."""
        names = {f for f in cls.__dataclass_fields__}
        if unknown := sorted(set(d) - names):
            raise ValueError(
                f"unknown scenario fields {unknown}, valid: {sorted(names)}"
            )
        ref = {k: _tuple(v) for k, v in d["reference"].items()}
        kind = _KINDS[ref.pop("kind")]
        if kind is Path:
            ref["arcs"] = tuple(Arc(**arc) for arc in ref["arcs"])
        return cls(
            name=d["name"],
            duration=float(d["duration"]),
            reference=kind(**ref),
            pushes=tuple(
                Push(**{k: _tuple(v) for k, v in p.items()})
                for p in d.get("pushes", ())
            ),
            friction_scale=float(d.get("friction_scale", 1.0)),
            payload_kg=float(d.get("payload_kg", 0.0)),
            seed_space=SeedSpace(
                **{k: _tuple(v) for k, v in d.get("seed_space", {}).items()}
            ),
        )


def _tuple(v: Any) -> Any:
    """JSON lists back to the tuples the frozen dataclasses hold."""
    return tuple(_tuple(x) for x in v) if isinstance(v, list | tuple) else v


class _Registry(Mapping[str, Scenario]):
    """Imports every module of this package on first use. Module
    `diffdrive_figure8` defines `SCENARIO` named `diffdrive.figure8`."""

    def __init__(self) -> None:
        self._loaded: dict[str, Scenario] | None = None

    def _all(self) -> dict[str, Scenario]:
        if self._loaded is None:
            found = {}
            for info in pkgutil.iter_modules(__path__):
                scenario = importlib.import_module(f"{__name__}.{info.name}").SCENARIO
                if scenario.name != info.name.replace("_", ".", 1):
                    raise ValueError(
                        f"scenarios/{info.name}.py defines {scenario.name!r}, "
                        f"expected {info.name.replace('_', '.', 1)!r}"
                    )
                found[scenario.name] = scenario
            self._loaded = dict(sorted(found.items()))
        return self._loaded

    def __getitem__(self, name: str) -> Scenario:
        try:
            return self._all()[name]
        except KeyError:
            raise KeyError(
                f"unknown scenario {name!r}, valid: {list(self._all())}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


SCENARIOS: Mapping[str, Scenario] = _Registry()
