import mujoco  # noqa: F401
import numpy as np

from robot_pg.contract import Graph
from robot_pg.provenance import content_hash
from robot_pg.runtime import run
from robot_pg.trajectory import Trajectory
from tests.test_runtime import Pd, pendulum


def graph(target: float):
    return Graph({"pd": Pd(target)}, ("command",))


def test_same_setup_gives_the_same_hash_and_the_same_bits():
    model = pendulum()
    a = run(model, graph(0.5), 2.0, 3)
    b = run(pendulum(), graph(0.5), 2.0, 3)
    assert a.content == b.content != ""
    assert a.content == content_hash(model, graph(0.5), 3)
    assert a.timing.shape == b.timing.shape
    for name, signal in a.signals.items():
        np.testing.assert_array_equal(signal, b.signals[name])


def test_the_seed_and_the_graph_change_the_hash():
    model = pendulum()
    baseline = content_hash(model, graph(0.5), 0)
    small = content_hash(model, graph(0.4), 0)
    other = content_hash(model, graph(0.5), 1)
    assert len({baseline, small, other}) == 3


def test_a_model_patch_changes_the_hash():
    model = pendulum()
    baseline = content_hash(model, graph(0.5), 0)
    model.opt.timestep *= 0.5
    assert content_hash(model, graph(0.5), 0) != baseline


def test_save_and_load_keep_the_content(tmp_path):
    path = run(pendulum(), graph(0.5), 2.0, 3).save(tmp_path / "run")
    loaded = Trajectory.load(path)
    assert loaded.content == content_hash(pendulum(), graph(0.5), 3)
