import numpy as np
import pytest
from numpy.testing import assert_allclose

from robot_pg.contract import SignalSpec
from robot_pg.controllers import Driver, JointPd, PathPursuit, Reach, Setpoint, drive
from robot_pg.robots import ROBOTS
from robot_pg.scenarios import SCENARIOS, Path, Targets

DT = 0.002


def test_joint_pd_outputs_the_exact_law():
    out = SignalSpec("tau", (3,), "N m")
    pd = JointPd(kp=100.0, kd=7.0, out=out)
    inputs = {
        "target": np.array([0.1, -0.2, 0.3]),
        "qpos_joints": np.array([0.0, 0.1, 0.4]),
        "qvel_joints": np.array([1.0, 2.0, 3.0]),
    }
    assert_allclose(pd.step(inputs)["tau"], [3.0, -44.0, -31.0], rtol=0, atol=1e-12)


def test_joint_pd_takes_a_force_unit_only():
    with pytest.raises(ValueError, match="JointPd outputs force"):
        JointPd(kp=1, kd=1, out=SignalSpec("target", (3,), "rad"))


def test_setpoint_shapes_its_value_like_the_port():
    out = SignalSpec("target", (3,))
    hold = Setpoint(out, [1.0, 2.0, 3.0])
    assert_allclose(hold.step({})["target"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not \\(3,\\)"):
        Setpoint(out, [1.0, 2.0])


def test_path_pursuit_tracks_the_figure8_on_the_diffdrive():
    episode = ROBOTS["diffdrive"].episode(SCENARIOS["diffdrive.figure8"], 0)
    model = episode.model()
    path = episode.reference
    assert isinstance(path, Path)
    graph = {"follow": PathPursuit(
        track_m=0.24,
        wheel_radius_m=0.05,
        lookahead_m=0.3,
        length_m=path.length,
        speed_m_s=path.speed,
    )}
    traj = drive(model, episode, graph, ("w",), 30.0)
    assert traj.warnings == {}
    assert ROBOTS["diffdrive"].fail_rule(traj, episode) is None
    cross_track = path.cross_track(traj.signals["base_pos"][::10, :2])
    assert cross_track.max() < 0.05
    assert path.cross_track(traj.signals["base_pos"][-1:, :2])[0] < 0.05


def test_reach_land_shorts_only_the_target_grid_on_the_panda():
    episode = ROBOTS["panda"].episode(SCENARIOS["panda.reach_grid"], 0)
    graph = {"reach": Reach(arm=7, hand=0.04)}
    traj = drive(episode.model(), episode, graph, ("arm", "hand"), 54.0)
    assert traj.warnings == {}
    assert ROBOTS["panda"].fail_rule(traj, episode) is None
    ref = episode.reference
    assert isinstance(ref, Targets)
    ends = ref.hold_ends(DT, len(traj))
    errors = np.linalg.norm(
        traj.signals["ee_pos"][ends] - ref.values(ends, DT), axis=1
    )
    assert errors.max() < 0.05


def test_driver_serves_reference_rows_in_order():
    ref = ROBOTS["panda"].episode(SCENARIOS["panda.reach_grid"], 0).reference
    driver = Driver(ref, DT)
    rows = np.stack(
        [driver.step()["ref_ee_target"], driver.step()["ref_ee_target"]]
    )
    grid = SCENARIOS["panda.reach_grid"].reference
    assert_allclose(grid.values(np.array([0, 1]), DT), rows)


def test_pursuit_seed_offset_leaves_the_figure8_within_the_fail_rule():
    episode = ROBOTS["diffdrive"].episode(SCENARIOS["diffdrive.figure8"], 4)
    path = episode.reference
    assert isinstance(path, Path)
    graph = {"follow": PathPursuit(0.24, 0.05, 0.3, path.length, path.speed)}
    traj = drive(episode.model(), episode, graph, ("w",), 30.0)
    assert traj.warnings == {}
    assert ROBOTS["diffdrive"].fail_rule(traj, episode) is None
