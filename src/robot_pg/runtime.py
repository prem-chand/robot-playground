from collections.abc import Collection, Mapping

import mujoco
from numpy.typing import ArrayLike

from robot_pg.contract import Array, Graph
from robot_pg.plant import MujocoPlant, RunUnstable, check_command, parse_model
from robot_pg.schedule import GraphError, compile_graph
from robot_pg.trajectory import Recorder, Trajectory

__all__ = ["RunUnstable", "Session", "run"]


class Session:
    """After `RunUnstable` the session refuses to advance until `reset`."""

    def __init__(
        self,
        model: mujoco.MjModel,
        graph: Graph,
        seed: int,
        *,
        record: Collection[str] = (),
        capacity: int = 1024,
    ) -> None:
        spec = parse_model(model)
        self.plant = MujocoPlant(model, spec)
        self._schedule = compile_graph(
            graph, spec.signals, model.opt.timestep, model.nu
        )
        check_command(graph, spec.actuators)
        self._graph, self._record, self._capacity = graph, tuple(record), capacity
        self.dt: float = model.opt.timestep
        self.reset(seed)

    def reset(self, seed: int) -> None:
        """Start a new episode. A `Trajectory` returned earlier keeps its rows."""
        self.plant.reset()
        self._schedule.reset(seed)
        self._recorder = Recorder(
            self._schedule.columns,
            len(self._graph.blocks),
            self._record,
            self._capacity,
        )
        self._fault: RunUnstable | None = None

    def advance(self, end: int) -> dict[str, Array] | None:
        """Run ticks until tick `end`. Return the slot inputs when the slot is due,
        then call `set_slot` and `advance` again. Return None at `end`.
        """
        if self._fault is not None:
            raise self._fault
        s, plant = self._schedule, self.plant
        while s.k < end:
            try:
                if (inputs := s.advance(plant, s.k + 1)) is not None:
                    return inputs
            except RunUnstable as bad:
                self._recorder.append(s.frame, s.timing_ns)
                bad.trajectory, self._fault = self.trajectory(), bad
                raise
            self._recorder.append(s.frame, s.timing_ns)
        return None

    def set_slot(self, outputs: Mapping[str, ArrayLike], compute_ns: int = -1) -> None:
        self._schedule.set_slot(outputs, compute_ns)

    def signal(self, name: str) -> Array:
        """Read-only view of a signal at the last tick. Valid until `advance`."""
        sl, shape = self._schedule.columns[name]
        return self._schedule.frame[sl].reshape(shape)

    def trajectory(self) -> Trajectory:
        """Rows recorded since `reset`, as read-only views."""
        return self._recorder.freeze(
            self.dt,
            tuple(self._graph.blocks),
            self._graph.command,
            self.plant.spec.actuators,
            self.plant.warnings(),
        )


def run(
    model: mujoco.MjModel,
    graph: Graph,
    duration: float,
    seed: int,
    record: Collection[str] = (),
) -> Trajectory:
    """Run `duration` seconds, a whole number of ticks, and return the trajectory."""
    ticks = round(duration / model.opt.timestep)
    if ticks < 1 or abs(ticks * model.opt.timestep - duration) > 1e-9:
        raise ValueError(f"duration {duration:g} s is not a whole number of ticks")
    session = Session(model, graph, seed, record=record, capacity=ticks)
    if session.advance(ticks) is not None:
        raise GraphError("graph has an external slot; drive it with Session")
    return session.trajectory()
