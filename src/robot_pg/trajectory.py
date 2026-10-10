"""One row per physics tick, saved as one NPZ file.

Row k is tick k at time `k * dt`. It holds the plant signals sensed at the start
of the tick, the block outputs and command of the tick, and the `effort` the
command produced over the step.
"""

import json
import os
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from robot_pg.contract import Array
from robot_pg.plant import Actuator, CtrlMap
from robot_pg.schedule import SMALL

FORMAT = 1
Column = tuple[slice, tuple[int, ...]]


class TrajectoryError(ValueError): ...


@dataclass(frozen=True, eq=False)
class Trajectory:
    dt: float
    signals: Mapping[str, Array]
    timing: NDArray[np.int64]
    blocks: tuple[str, ...]
    command_names: tuple[str, ...]
    actuators: tuple[Actuator, ...]
    warnings: Mapping[str, int]
    content: str = ""

    def __len__(self) -> int:
        return self.timing.shape[0]

    @property
    def time(self) -> Array:
        return np.arange(len(self)) * self.dt

    @property
    def command(self) -> Array:
        cols = [self.signals[n].reshape(len(self), -1) for n in self.command_names]
        return np.concatenate(cols, axis=1)

    @property
    def ctrl(self) -> Array:
        return CtrlMap.of(self.actuators).apply(self.command)

    @property
    def clipped(self) -> NDArray[np.bool_]:
        """(T, nu) True where the command left the actuator's ctrlrange."""
        return CtrlMap.of(self.actuators).clipped(self.command)

    def save(self, path: str | Path) -> Path:
        """Write `<path>.npz` through a temporary file, so a crash leaves no half file."""
        dest = Path(path).with_suffix(".npz")
        meta = {
            "format": FORMAT,
            "dt": self.dt,
            "signals": list(self.signals),
            "blocks": list(self.blocks),
            "command_names": list(self.command_names),
            "actuators": [asdict(a) for a in self.actuators],
            "warnings": dict(self.warnings),
            "content": self.content,
        }
        arrays = {f"signal/{n}": a for n, a in self.signals.items()}
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), timing=self.timing, **arrays)
        os.replace(tmp, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as f:
            try:
                meta = json.loads(str(f["meta"]))
                signals = {n: f[f"signal/{n}"] for n in meta["signals"]}
                timing = f["timing"]
            except KeyError as e:
                raise TrajectoryError(f"{path}: missing {e.args[0]}") from None
        if meta["format"] != FORMAT:
            raise TrajectoryError(f"{path}: format {meta['format']}, not {FORMAT}")
        actuators = tuple(
            Actuator(
                **{
                    **a,
                    "ctrl_range": tuple(a["ctrl_range"]),
                    "command_range": tuple(a["command_range"]),
                }
            )
            for a in meta["actuators"]
        )
        return cls(
            meta["dt"],
            signals,
            timing,
            tuple(meta["blocks"]),
            tuple(meta["command_names"]),
            actuators,
            meta["warnings"],
            meta.get("content", ""),
        )


class Recorder:
    def __init__(
        self,
        columns: Mapping[str, Column],
        n_blocks: int,
        record: Collection[str] = (),
        capacity: int = 1024,
    ) -> None:
        if unknown := sorted(set(record) - set(columns)):
            raise TrajectoryError(
                f"record names {unknown}, not signals of {list(columns)}"
            )
        small = [n for n, (_, s) in columns.items() if prod(s) <= SMALL]
        self._prefix = max((columns[n][0].stop for n in small), default=0)
        extra = [n for n in columns if n in record and n not in small]
        self._layout: dict[str, Column] = {n: columns[n] for n in small}
        self._extra: list[tuple[slice, slice]] = []
        top = self._prefix
        for n in extra:
            sl, shape = columns[n]
            self._layout[n] = (slice(top, top + prod(shape)), shape)
            self._extra.append((slice(top, top + prod(shape)), sl))
            top += prod(shape)
        self._rows = np.empty((capacity, top))
        self._timing = np.empty((capacity, n_blocks), dtype=np.int64)
        self.n = 0

    def append(self, frame: Array, timing_ns: NDArray[np.int64]) -> None:
        if self.n == len(self._rows):
            self._rows = np.concatenate([self._rows, np.empty_like(self._rows)])
            self._timing = np.concatenate([self._timing, np.empty_like(self._timing)])
        row = self._rows[self.n]
        row[: self._prefix] = frame[: self._prefix]
        for to, sl in self._extra:
            row[to] = frame[sl]
        self._timing[self.n] = timing_ns
        self.n += 1

    def freeze(
        self,
        dt: float,
        blocks: tuple[str, ...],
        command_names: tuple[str, ...],
        actuators: tuple[Actuator, ...],
        warnings: Mapping[str, int],
        content: str = "",
    ) -> Trajectory:
        n, rows = self.n, self._rows[: self.n]
        signals = {
            name: rows[:, sl].reshape(n, *shape)
            for name, (sl, shape) in self._layout.items()
        }
        timing = self._timing[:n]
        for a in (*signals.values(), timing):
            a.flags.writeable = False
        return Trajectory(
            dt, signals, timing, blocks, command_names, actuators, dict(warnings), content
        )
