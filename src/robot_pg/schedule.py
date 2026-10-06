import math
from collections.abc import Callable, Mapping
from time import perf_counter_ns
from types import MappingProxyType
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike

from robot_pg.contract import Array, External, Graph, SignalSpec

SMALL = 64


class GraphError(ValueError): ...


class Plant(Protocol):
    def sense(self, signals: Mapping[str, Array]) -> None: ...
    def act(self, command: Array) -> None: ...


class _Out(NamedTuple):
    port: str
    shape: tuple[int, ...]
    signal: Array
    ring: Array
    prev: Array | None


class _Run(NamedTuple):
    col: int
    name: str
    step: Callable[[Mapping[str, Array]], Mapping[str, ArrayLike]] | None
    inputs: Mapping[str, Array]
    outs: tuple[_Out, ...]
    period: int
    latency: int


def compile_graph(
    graph: Graph,
    plant_signals: tuple[SignalSpec, ...],
    physics_dt: float,
    n_actuators: int,
) -> "Schedule":
    specs = {n: b.spec for n, b in graph.blocks.items()}
    slots = [n for n, b in graph.blocks.items() if isinstance(b, External)]
    if len(slots) > 1:
        raise GraphError(f"graph has two external slots: {slots[0]}, {slots[1]}")
    hz, ticks = 1 / physics_dt, {}
    for name, s in specs.items():
        p, n = hz / s.rate_hz if s.rate_hz > 0 else 0.5, s.latency_s / physics_dt
        if p < 1 or abs(p - round(p)) > 1e-9 * p:
            div = f"{s.rate_hz:g} Hz does not divide {hz:g} Hz"
            raise GraphError(f"block {name}: {div}")
        if n < 0 or abs(n - round(n)) > 1e-9 * max(n, 1):
            why = f"is {n:g} ticks, not a whole number of ticks"
            raise GraphError(f"block {name}: latency {s.latency_s:g} s {why}")
        ticks[name] = (round(p), round(n))
    ports = {f"{b}.{p.name}" for b, s in specs.items() for p in s.inputs + s.outputs}
    if bad := [w for w in graph.wires if w not in ports]:
        raise GraphError(f"wire {bad[0]} names no port of a block")
    table = {s.name: ("plant", s) for s in plant_signals}
    for name, s in specs.items():
        for o in s.outputs:
            if (sig := graph.signal_of(name, o.name)) in table:
                who = f"{table[sig][0]}, {name}"
                raise GraphError(f"signal {sig} has two producers: {who}")
            table[sig] = (name, o)
    same_tick: dict[str, list[tuple[str, str]]] = {n: [] for n in specs}
    for name, s in specs.items():
        for i in s.inputs:
            sig, head = graph.signal_of(name, i.name), f"block {name}: input {i.name}"
            if sig not in table:
                raise GraphError(f"{head} reads signal {sig}, which has no producer")
            src, o = table[sig]
            if o.shape != i.shape:
                tail = f"signal {sig} from {src} has shape {o.shape}"
                raise GraphError(f"{head} has shape {i.shape} but {tail}")
            if src in ticks and ticks[src][1] == 0:
                same_tick[name].append((src, sig))
    if (width := sum(table[c][1].size for c in graph.command)) != n_actuators:
        parts = ", ".join(f"{c} {table[c][1].size}" for c in graph.command)
        tail = f"but the plant has {n_actuators} actuators"
        raise GraphError(f"command has {width} values ({parts}) {tail}")
    return Schedule(graph, _run_order(same_tick), table, ticks, hz)


def _run_order(edges: Mapping[str, list[tuple[str, str]]]) -> list[str]:
    order: list[str] = []
    while len(order) < len(edges):
        left = [b for b in edges if b not in order]
        if not (ready := [b for b in left if all(p in order for p, _ in edges[b])]):
            hops, b = [], left[0]
            while b not in [h[0] for h in hops]:
                src, sig = next(e for e in edges[b] if e[0] in left)
                hops, b = [*hops, (b, sig, src)], src
            hops = hops[[h[0] for h in hops].index(b) :]
            loop = ", ".join(f"{a} reads {s} from {c}" for a, s, c in hops)
            raise GraphError(f"cycle inside one tick: {loop}")
        order.append(ready[0])
    return order


class Schedule:
    def __init__(
        self, graph: Graph, order: list[str], table: Mapping, ticks: Mapping, hz: float
    ) -> None:
        names, specs = list(table), {n: b.spec for n, b in graph.blocks.items()}
        rest = [s for s in names if s not in graph.command]
        rest.sort(key=lambda s: (table[s][1].size > SMALL, names.index(s)))
        cells: list[tuple[object, SignalSpec, int]]
        cells = [(s, table[s][1], 1) for s in (*graph.command, *rest)]
        frame = sum(o.size for _, o, _ in cells)
        for n, (per, lat) in ticks.items():
            for o in specs[n].outputs if lat else ():
                cells.append(((n, o.name), o, -(-lat // per)))
                cells += [((n, o.name, "prev"), o, 1)] * (o.name in specs[n].feedback)
        at, top, init = {}, 0, [np.zeros(0)]
        for key, o, rows in cells:
            at[key], top = slice(top, top + rows * o.size), top + rows * o.size
            init.append(np.resize(np.asarray(o.initial, float), rows * o.size))
        self._init = np.concatenate(init)
        self._buf = buf = self._init.copy()
        ro = buf.view()
        ro.flags.writeable = False

        def view(a: Array, key: object, shape: tuple[int, ...]) -> Array:
            return a[at[key]].reshape(shape)

        runs = {}
        for idx, (name, block) in enumerate(graph.blocks.items()):
            s, (per, lat), outs, ins = block.spec, ticks[name], [], {}
            for o in s.outputs:
                sig, pk = graph.signal_of(name, o.name), (name, o.name, "prev")
                ring = view(buf, (name, o.name) if lat else sig, (-1, *o.shape))
                pv = view(buf, pk, o.shape) if pk in at else None
                outs.append(_Out(o.name, o.shape, view(buf, sig, o.shape), ring, pv))
                if o.name in s.feedback:
                    ins[f"{o.name}_prev"] = view(ro, sig if pv is None else pk, o.shape)
            for i in s.inputs:
                ins[i.name] = view(ro, graph.signal_of(name, i.name), i.shape)
            step = None if isinstance(block, External) else block.step
            ins = MappingProxyType(ins)
            runs[name] = _Run(idx, name, step, ins, tuple(outs), per, lat)
        self._order = seq = tuple(runs[n] for n in order)
        plan = []
        for h in range(math.lcm(*(r.period for r in seq))):
            due = tuple(r for r in seq if h % r.period == 0)
            rel = tuple(r for r in seq if r.latency and (h - r.latency) % r.period == 0)
            plan.append((rel, due))
        self._plan, self._hz = tuple(plan), hz
        self._blocks = [b for b in graph.blocks.values() if not isinstance(b, External)]
        plant = {
            s: view(buf, s, o.shape) for s, (p, o) in table.items() if p == "plant"
        }
        self._plant = MappingProxyType(plant)
        self._command = ro[: sum(table[c][1].size for c in graph.command)]
        self.frame = ro[:frame]
        self.columns = MappingProxyType({s: (at[s], table[s][1].shape) for s in names})
        self.timing_ns = np.full(len(graph.blocks), -1, dtype=np.int64)
        self.k, self._cursor, self._waiting = 0, None, None

    def reset(self, seed: int) -> None:
        np.copyto(self._buf, self._init)
        self.k, self._cursor, self._waiting = 0, None, None
        for b in self._blocks:
            b.reset(seed)

    def advance(self, plant: Plant, end: int) -> dict[str, Array] | None:
        """Run to `end`, or pause the tick at a due slot and return its inputs."""
        if self._waiting is not None:
            raise RuntimeError(f"slot {self._waiting.name} is waiting for set_slot")
        while True:
            k = self.k
            releases, runs = self._plan[k % len(self._plan)]
            if self._cursor is None:
                if k >= end:
                    return None
                plant.sense(self._plant)
                self.timing_ns.fill(-1)
                for r in releases:
                    j = (k - r.latency) // r.period
                    for o in r.outs:
                        o.signal[...] = o.ring[j % len(o.ring)]
                self._cursor = 0
            for i in range(self._cursor, len(runs)):
                if (r := runs[i]).step is None:
                    self._cursor, self._waiting = i + 1, r
                    return {n: v.copy() for n, v in r.inputs.items()}
                t0 = perf_counter_ns()
                out = r.step(r.inputs)
                self.timing_ns[r.col] = perf_counter_ns() - t0
                self._emit(r, out, k)
            self.k, self._cursor = k + 1, None
            plant.act(self._command)

    def tick(self, k: int, plant: Plant) -> Array:
        if k != self.k or self.advance(plant, k + 1) is not None:
            raise RuntimeError(f"tick {k}: next tick is {self.k}, or a slot is due")
        return self._command

    def set_slot(self, outputs: Mapping[str, ArrayLike], compute_ns: int = -1) -> None:
        if (r := self._waiting) is None:
            raise RuntimeError("no slot is waiting for set_slot")
        self._emit(r, outputs, self.k)
        self._waiting, self.timing_ns[r.col] = None, compute_ns

    def _emit(self, r: _Run, out: Mapping[str, ArrayLike], k: int) -> None:
        for o in r.outs:
            v = out[o.port]
            if (shape := v.shape if type(v) is np.ndarray else np.shape(v)) != o.shape:
                raise ValueError(f"block {r.name}: {o.port} is {shape}, not {o.shape}")
            o.ring[(k // r.period) % len(o.ring)] = v
            if o.prev is not None:
                o.prev[...] = v

    def __repr__(self) -> str:
        rows = [f"Schedule {self._hz:g} Hz, hyperperiod {len(self._plan)} ticks"]
        for i, r in enumerate(self._order, 1):
            slot = ", slot" if r.step is None else ""
            rows.append(f"  {i}. {r.name} every {r.period}, latency {r.latency}{slot}")
        return "\n".join(rows)
