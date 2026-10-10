"""The content hash of a run setup, per ROADMAP Section 5.3.

Two runs whose `run()` arguments give the same content hash replay bitwise on
the same machine. The hash covers the MuJoCo version, the full compiled model
including every scenario patch, the graph with each block's class, spec, and
`digest`, and the seed. It trusts each block's `digest`: a controller that
changes its code without a new `digest` keeps the hash. `record` names change
only what a run saves, so they are outside the hash.
"""

import hashlib
import json

import mujoco
import numpy as np

from robot_pg.contract import BlockSpec, Graph, SignalSpec

__all__ = ["content_hash"]


def content_hash(model: mujoco.MjModel, graph: Graph, seed: int) -> str:
    """A short digest of everything that decides the bits of a run."""
    h = hashlib.sha256()
    h.update(mujoco.__version__.encode())
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    h.update(bytes(buf))
    h.update(json.dumps(_graph_specs(graph), sort_keys=True).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def _graph_specs(graph: Graph) -> dict[str, object]:
    blocks: dict[str, object] = {}
    for name, block in graph.blocks.items():
        blocks[name] = _block_spec(
            type(block).__qualname__, block.spec, block.digest
        )
    return {"blocks": blocks, "wires": dict(graph.wires), "command": graph.command}


def _block_spec(kind: str, spec: BlockSpec, digest: str) -> dict[str, object]:
    return {
        "kind": kind,
        "digest": digest,
        "rate_hz": spec.rate_hz,
        "latency_s": spec.latency_s,
        "inputs": [_signal(s) for s in spec.inputs],
        "outputs": [_signal(s) for s in spec.outputs],
        "budget_s": spec.budget_s,
        "feedback": spec.feedback,
    }


def _signal(s: SignalSpec) -> dict[str, object]:
    return {
        "name": s.name,
        "shape": s.shape,
        "unit": s.unit,
        "initial": s.initial,
    }
