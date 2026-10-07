import json
import math
from dataclasses import replace

import numpy as np
import pytest

from robot_pg.robots import ROBOTS
from robot_pg.scenarios import SCENARIOS, Path, Push, Scenario, Targets, Velocity

DT = 0.002
AT_0_5_10 = np.array([0, 2500, 5000])


def test_registry_discovers_one_scenario_per_file():
    assert list(SCENARIOS) == ["diffdrive.figure8", "panda.reach_grid"]
    with pytest.raises(KeyError, match="valid: \\['diffdrive.figure8'"):
        SCENARIOS["panda.nope"]


def test_figure8_reference_at_0_5_and_10_s():
    path = SCENARIOS["diffdrive.figure8"].reference
    assert isinstance(path, Path)
    assert path.speed == 0.5
    assert path.length == pytest.approx(4 * math.pi)
    s = np.minimum(path.speed * AT_0_5_10 * DT, path.length)
    assert path.pose(s).round(4).tolist() == [
        [0.0, 0.0, 0.0],
        [0.5985, 1.8011, 2.5],
        [-0.9589, 0.7163, 5.0],
    ]
    assert path.values(AT_0_5_10, DT).shape == (3, 256, 2)


def test_figure8_cross_track_is_exact():
    path = SCENARIOS["diffdrive.figure8"].reference
    assert isinstance(path, Path)
    on = path.pose(np.linspace(0.0, path.length, 1001))[:, :2]
    assert path.cross_track(on).max() < 1e-12
    assert path.cross_track(
        np.array([[0.0, 0.3], [0.0, 2.6], [3.0, 0.0]])
    ).tolist() == pytest.approx([0.3, 0.6, math.sqrt(10) - 1])
    s, d = path.nearest(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert s[0].tolist() == pytest.approx([0.0, 2 * math.pi], abs=1e-12)
    assert s[1, 0] == pytest.approx(math.pi / 2, abs=1e-12)
    assert d.ravel().tolist() == pytest.approx(
        [0.0, 0.0, 0.0, math.sqrt(5) - 1], abs=1e-12
    )


def test_velocity_knots_make_steps_and_ramps():
    step = Velocity(((0.0, 0.0), (3.1, 0.0), (3.1, 0.4)))
    assert step.values(np.array([1549, 1550, 5000]), DT).ravel().tolist() == [
        0.0,
        0.4,
        0.4,
    ]
    ramp = Velocity(((0.0, 0.2), (10.0, 0.5)))
    assert ramp.values(np.array([0, 2500, 6000]), DT).ravel().tolist() == pytest.approx(
        [0.2, 0.35, 0.5]
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        Velocity(((1.0, 0.0), (0.0, 0.0)))


def test_reach_grid_is_27_targets_in_product_order_held_2_s():
    ref = SCENARIOS["panda.reach_grid"].reference
    assert isinstance(ref, Targets)
    assert len(ref.points) == 27
    assert ref.points[:4] == (
        (0.4, -0.2, 0.3),
        (0.4, -0.2, 0.45),
        (0.4, -0.2, 0.6),
        (0.4, 0.0, 0.3),
    )
    assert ref.points[-1] == (0.6, 0.2, 0.6)
    assert ref.values(AT_0_5_10, DT).tolist() == [
        [0.4, -0.2, 0.3],
        [0.4, -0.2, 0.6],
        [0.4, 0.0, 0.6],
    ]
    assert ref.hold_ends(DT, 27000).tolist()[:3] == [999, 1999, 2999]
    assert len(ref.hold_ends(DT, 27000)) == 27
    assert SCENARIOS["panda.reach_grid"].duration == 54.0


def test_seed_0_draws_the_nominal_sample():
    for robot, scenario in (
        ("diffdrive", "diffdrive.figure8"),
        ("panda", "panda.reach_grid"),
    ):
        s = ROBOTS[robot].episode(SCENARIOS[scenario], 0).sample
        assert s.friction_factor == 1.0
        assert s.base_offset == (0.0, 0.0, 0.0)
        assert set(s.joint_offsets) <= {0.0}
    ep = ROBOTS["panda"].episode(SCENARIOS["panda.reach_grid"], 0)
    assert ep.sample.target_order == tuple(range(27))
    assert ep.reference == SCENARIOS["panda.reach_grid"].reference


def test_joint_offsets_stay_inside_the_joint_range():
    m = ROBOTS["panda"].model()
    for seed in range(1, 40):
        q = (
            ROBOTS["panda"]
            .episode(SCENARIOS["panda.reach_grid"], seed)
            .model()
            .key_qpos[0]
        )
        lo, hi = m.jnt_range[:7].T
        assert np.all((q[:7] >= lo) & (q[:7] <= hi))


def test_push_covers_whole_ticks():
    assert Push(5.0, 0.1, (0.0, 20.0, 0.0)).ticks(DT) == range(2500, 2550)
    with pytest.raises(ValueError, match="not a whole number"):
        Push(1.0, 0.0001, (50.0, 0.0, 0.0)).ticks(DT)


def test_scenario_fields_are_checked():
    with pytest.raises(ValueError, match="payload_kg must be at least 0"):
        replace(SCENARIOS["panda.reach_grid"], payload_kg=-1.0)
    with pytest.raises(ValueError, match="is not <robot>.<name>"):
        Scenario("figure8", 1.0, Velocity.constant(0.0))
