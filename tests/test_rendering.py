import struct
import zlib
from pathlib import Path

import numpy as np

from robot_pg.rendering import frames, write_png
from robot_pg.robots import ROBOTS


def test_png_holds_the_pixels(tmp_path: Path):
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    data = write_png(tmp_path / "f.png", pixels).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (6, 4)
    length = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    raw = zlib.decompress(data[41 : 41 + length])
    rows = [raw[r * 19 : (r + 1) * 19] for r in range(4)]
    assert all(row[0] == 0 for row in rows)
    assert np.array_equal(
        np.frombuffer(b"".join(row[1:] for row in rows), np.uint8).reshape(4, 6, 3),
        pixels,
    )
