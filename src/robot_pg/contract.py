from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SignalSpec:
    """A float64 array. `initial` is a scalar fill or `size` values in C order."""

    name: str
    shape: tuple[int, ...]
    unit: str = ""
    initial: float | tuple[float, ...] = 0.0

    def __post_init__(self) -> None:
        if np.size(self.initial) not in (1, self.size):
            raise ValueError(f"signal {self.name}: initial needs {self.size} values")

    @property
    def size(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class BlockSpec:
    """Each `feedback` output is read back as `<name>_prev`."""

    rate_hz: float
    latency_s: float = 0.0
    inputs: tuple[SignalSpec, ...] = ()
    outputs: tuple[SignalSpec, ...] = ()
    budget_s: float | None = None
    feedback: tuple[str, ...] = ()


class Block(Protocol):
    """`inputs` are read-only views, valid only during the call. `digest` names
    the block's own config, so two blocks with the same spec hash apart.
    """

    @property
    def spec(self) -> BlockSpec: ...
    @property
    def digest(self) -> str: ...
    def reset(self, seed: int) -> None: ...
    def step(self, inputs: Mapping[str, Array]) -> Mapping[str, ArrayLike]: ...


@dataclass(frozen=True)
class External:
    """`Schedule.set_slot` fills this slot's outputs."""

    spec: BlockSpec

    @property
    def digest(self) -> str:
        return "external"


@dataclass(frozen=True)
class Graph:
    """Blocks in declaration order, command in actuator order, "block.port" wires."""

    blocks: Mapping[str, Block | External]
    command: tuple[str, ...]
    wires: Mapping[str, str] = field(default_factory=dict)

    def signal_of(self, block: str, port: str) -> str:
        return self.wires.get(f"{block}.{port}", port)
