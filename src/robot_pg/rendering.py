import math
import os
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_pg.contract import Array

Pixels = NDArray[np.uint8]


def frames(model: mujoco.MjModel, qpos: Array) -> Iterator[Pixels]:
    """One frame per row of `qpos`, at the model's offscreen buffer size."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model, model.vis.global_.offheight, model.vis.global_.offwidth
    )
    try:
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        cam = _fit(model, data)
        offset = cam.lookat - _free_body_pos(model, data)
        for row in qpos:
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            cam.lookat[:] = offset + _free_body_pos(model, data)
            renderer.update_scene(data, camera=cam)
            yield renderer.render().copy()
    finally:
        renderer.close()


def write_png(path: str | Path, pixels: Pixels) -> Path:
    """An 8-bit RGB PNG, written through a temporary file."""
    dest = Path(path)
    h, w, _ = pixels.shape
    raw = b"".join(b"\0" + pixels[r].tobytes() for r in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(png)
    os.replace(tmp, dest)
    return dest


def _fit(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    robot = model.geom_bodyid != 0
    r = model.geom_rbound[robot][:, None]
    lo = (data.geom_xpos[robot] - r).min(axis=0)
    hi = (data.geom_xpos[robot] + r).max(axis=0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = (
        np.linalg.norm(hi - lo) / 2 / math.sin(math.radians(model.vis.global_.fovy) / 2)
    )
    cam.azimuth, cam.elevation = model.vis.global_.azimuth, model.vis.global_.elevation
    return cam


def _free_body_pos(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if not free.size:
        return np.zeros(3)
    return data.xpos[model.jnt_bodyid[free[0]]].copy()
