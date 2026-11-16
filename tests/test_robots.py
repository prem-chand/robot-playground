import math
from collections.abc import Mapping
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from robot_pg.cli import main
from robot_pg.contract import Array, BlockSpec, Graph, SignalSpec
from robot_pg.plant import parse_model
from robot_pg.robots import ASSETS, ROBOTS, Episode
from robot_pg.runtime import Session, run
from robot_pg.scenarios import SCENARIOS
from robot_pg.trajectory import Trajectory

DT = 0.002


class Hold:
    def __init__(self, *outputs: tuple[SignalSpec, list[float]]) -> None:
        self.spec = BlockSpec(500.0, outputs=tuple(s for s, _ in outputs))
        self.out = {s.name: np.array(v) for s, v in outputs}
        self.digest = f"hold {[v for _, v in outputs]}"

    def reset(self, seed: int) -> None:
        pass

    def step(self, inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        return self.out


def episode(robot: str, scenario: str, seed: int = 0, **fields: float) -> Episode:
    return ROBOTS[robot].episode(replace(SCENARIOS[scenario], **fields), seed)


def synthetic(
    signals: Mapping[str, Array], warnings: Mapping[str, int] | None = None
) -> Trajectory:
    n = len(next(iter(signals.values())))
    return Trajectory(
        DT, signals, np.zeros((n, 0), np.int64), (), (), (), warnings or {}
    )


def test_each_robot_has_its_actuator_kinds():
    kinds = {
        name: [(a.kind, a.unit) for a in parse_model(robot.model()).actuators]
        for name, robot in ROBOTS.items()
    }
    assert kinds == {
        "diffdrive": [("velocity", "rad/s")] * 2,
        "panda": [("position", "rad")] * 7 + [("position", "m")],
    }


def test_panda_gripper_takes_a_0_to_4_cm_command_and_the_ee_site_is_added_at_load():
    m = ROBOTS["panda"].model()
    assert parse_model(m).actuators[7].command_range == pytest.approx((0.0, 0.04))
    assert m.site("ee").bodyid == m.body("hand").id
    assert m.site_pos[m.site("ee").id].tolist() == [0.0, 0.0, 0.1034]
    vendored = mujoco.MjModel.from_xml_path(
        str(ASSETS / "franka_emika_panda/scene.xml")
    )
    assert vendored.nsite == 0


def test_vendored_g1_loads_standing_on_the_floor_with_its_license():
    folder = ASSETS / "unitree_g1"
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert (m.nq, m.nu, m.nkey) == (36, 29, 1)
    assert [(a.kind, a.unit) for a in parse_model(m).actuators] == [
        ("position", "rad")
    ] * 29
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    assert d.body("pelvis").xpos[2] == pytest.approx(0.79)
    assert (
        (folder / "LICENSE")
        .read_text()
        .startswith("Copyright (c) 2016-2023 HangZhou YuShu")
    )
    assert (folder / "VENDORED.txt").read_text().splitlines()[1:] == [
        "4d038b3feae26ec82b46a4d586379114012a8ac7",
        "unitree_g1/",
    ]


def test_inspect_lists_kinds_and_base_signals(capsys: pytest.CaptureFixture[str]):
    assert main(["inspect", "diffdrive"]) == 0
    out = capsys.readouterr().out
    assert "actuators 2: 2 velocity" in out
    for signal in ("base_pos", "base_quat", "base_linvel_body", "base_angvel_body"):
        assert f"  {signal} " in out
    main(["inspect", "panda"])
    out = capsys.readouterr().out
    assert "actuators 8: 8 position" in out
    assert "tendon split  0 to 0.04" in out
    assert "  ee_pos " in out


def test_diffdrive_holds_home_on_three_contacts():
    m = episode("diffdrive", "diffdrive.figure8").model()
    traj = run(
        m,
        Graph({"h": Hold((SignalSpec("w", (2,), "rad/s"), [0.0, 0.0]))}, ("w",)),
        1.0,
        0,
    )
    assert traj.warnings == {}
    assert abs(traj.signals["base_pos"][-1, 2] - 0.05) < 1e-3
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    touching = {m.geom(int(c.geom2)).name for c in d.contact[: d.ncon]}
    assert touching == {"caster", "left_wheel", "right_wheel"}


def test_panda_holds_home_for_1_s():
    m = ROBOTS["panda"].model()
    home = m.key_qpos[0][:7]
    hold = Hold(
        (SignalSpec("arm", (7,), "rad"), home.tolist()),
        (SignalSpec("grip", (1,), "m"), [0.04]),
    )
    traj = run(m, Graph({"h": hold}, ("arm", "grip")), 1.0, 0)
    assert traj.warnings == {}
    assert np.abs(traj.signals["qpos_joints"][:, :7] - home).max() < 1e-2
    assert np.allclose(traj.signals["ee_pos"][0], [0.5545, 0.0, 0.5211], atol=1e-4)


def test_episode_edits_a_copy_and_applying_twice_equals_once():
    before = ROBOTS["panda"].model()
    ep = episode(
        "panda", "panda.reach_grid", seed=3, payload_kg=2.0, friction_scale=0.4
    )
    m = ep.model()
    hand = m.body("hand").id
    fresh = ROBOTS["panda"].model()
    fresh.body_mass[hand] += 2.0
    assert m.body_mass[hand] == fresh.body_mass[hand]
    assert np.array_equal(m.body_inertia, before.body_inertia)
    assert np.array_equal(
        m.geom_friction, before.geom_friction * (0.4 * ep.sample.friction_factor)
    )
    owned = (m.geom_friction.copy(), m.body_mass.copy(), m.key_qpos.copy())
    ep.apply(m)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(owned, (m.geom_friction, m.body_mass, m.key_qpos))
    )
    after = ROBOTS["panda"].model()
    assert np.array_equal(after.body_mass, before.body_mass)
    assert np.array_equal(after.geom_friction, before.geom_friction)


def test_seed_offsets_reach_the_plant_at_tick_0():
    ep = episode("diffdrive", "diffdrive.figure8", seed=2)
    m = ep.model()
    session = Session(
        m, Graph({"h": Hold((SignalSpec("w", (2,), "rad/s"), [0.0, 0.0]))}, ("w",)), 2
    )
    session.advance(1)
    dx, dy, yaw = ep.sample.base_offset
    home = ROBOTS["diffdrive"].model().key_qpos[0]
    assert np.allclose(session.signal("base_pos"), home[:3] + [dx, dy, 0.0])
    assert np.allclose(
        session.signal("base_quat"), [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    )
    assert np.allclose(
        session.signal("qpos_joints"), home[7:] + ep.sample.joint_offsets
    )


def test_qpos_rebuild_from_recorded_signals_is_exact():
    ep = episode("diffdrive", "diffdrive.figure8", seed=1)
    m = ep.model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    traj = run(
        m,
        Graph({"h": Hold((SignalSpec("w", (2,), "rad/s"), [0.0, 0.0]))}, ("w",)),
        0.1,
        1,
    )
    q = ROBOTS["diffdrive"].qpos(traj.signals)
    assert q.shape == (50, 9)
    assert np.array_equal(q[0], d.qpos)


def test_diffdrive_fails_beyond_50_cm_cross_track():
    ep = episode("diffdrive", "diffdrive.figure8")
    pos = np.zeros((3, 3))
    pos[:, 1] = [0.0, -0.49, -0.51]
    failure = ROBOTS["diffdrive"].fail_rule(synthetic({"base_pos": pos}), ep)
    assert failure is not None
    assert (failure.tick, failure.reason) == (2, "cross-track 0.510 m > 0.5 m")


def test_panda_fails_only_at_the_end_of_a_hold_and_against_the_shuffled_target():
    ep = episode("panda", "panda.reach_grid", seed=5)
    targets = ep.reference.values(np.arange(3000), DT)
    ee = targets.copy()
    ee[1000:1500] += [0.5, 0.0, 0.0]
    ee[1999] += [0.0, 0.19, 0.0]
    assert ROBOTS["panda"].fail_rule(synthetic({"ee_pos": ee}), ep) is None
    ee[2999] += [0.0, 0.0, 0.21]
    failure = ROBOTS["panda"].fail_rule(synthetic({"ee_pos": ee}), ep)
    assert failure is not None
    assert (failure.tick, failure.reason) == (
        2999,
        "ee error 0.210 m > 0.2 m at the end of hold 3/27",
    )
    nominal = SCENARIOS["panda.reach_grid"].reference.values(np.arange(3000), DT)
    assert ROBOTS["panda"].fail_rule(synthetic({"ee_pos": nominal}), ep) is not None


def test_every_robot_fails_on_run_unstable_at_the_last_tick():
    ep = episode("diffdrive", "diffdrive.figure8")
    traj = synthetic({"base_pos": np.zeros((6, 3))}, {"BADQACC": 1, "CONTACTFULL": 3})
    failure = ROBOTS["diffdrive"].fail_rule(traj, ep)
    assert failure is not None
    assert (failure.tick, failure.reason) == (5, "RunUnstable: bad BADQACC")
    assert (
        ROBOTS["diffdrive"].fail_rule(
            synthetic({"base_pos": np.zeros((6, 3))}, {"CONTACTFULL": 3}), ep
        )
        is None
    )


def test_fail_rules_describe_their_thresholds():
    assert {n: r.fail_rule.describe() for n, r in ROBOTS.items()} == {
        "diffdrive": "cross-track > 0.5 m, or RunUnstable",
        "panda": "ee error > 0.2 m at the end of a hold, or RunUnstable",
    }


def test_a_push_on_the_fixed_panda_base_is_refused():
    from robot_pg.scenarios import Push

    with pytest.raises(ValueError, match="pushes the fixed base of panda"):
        ROBOTS["panda"].episode(
            replace(
                SCENARIOS["panda.reach_grid"],
                pushes=(Push(1.0, 0.1, (10.0, 0.0, 0.0)),),
            ),
            0,
        )
