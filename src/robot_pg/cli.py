import argparse
import platform
import subprocess
import sys
from collections.abc import Mapping


def _cpu() -> str:
    if sys.platform == "darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    return platform.processor() or platform.machine()


def doctor() -> int:
    import mujoco
    import numpy

    print(f"python {platform.python_version()}")
    print(f"mujoco {mujoco.__version__}")
    print(f"numpy {numpy.__version__}")
    print(f"cpu {_cpu()}")
    print("ok")
    return 0


def inspect(args: argparse.Namespace) -> int:
    if args.scenario is not None:
        return _inspect_scenario(args.scenario)
    return _inspect_robot(args.subject)


def _inspect_robot(name: str) -> int:
    import mujoco

    from robot_pg.plant import parse_model
    from robot_pg.robots import ROBOTS

    robot = _lookup(ROBOTS, name, "robot")
    m = robot.model()
    spec = parse_model(m)
    integrator = mujoco.mjtIntegrator(m.opt.integrator).name.removeprefix("mjINT_")
    print(
        f"robot {robot.name}  mjcf {robot.mjcf}  dt {m.opt.timestep:g} s  integrator {integrator}"
    )
    ee = robot.ee_site.body if robot.ee_site else "-"
    print(f"base {robot.base_body}  payload {robot.payload_body}  ee site on {ee}")
    print(f"fail rule: {robot.fail_rule.describe()}")
    kinds = [a.kind for a in spec.actuators]
    counts = ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds))
    print(f"actuators {len(kinds)}: {counts}")
    rows = [("name", "kind", "unit", "drives", "command range", "ctrl range")]
    for i, a in enumerate(spec.actuators):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            drives = f"tendon {m.tendon(m.actuator_trnid[i, 0]).name}"
        else:
            drives = f"joint {m.joint(m.actuator_trnid[i, 0]).name}"
        rows.append(
            (
                a.name,
                a.kind,
                a.unit,
                drives,
                _range(a.command_range),
                _range(a.ctrl_range),
            )
        )
    _table(rows)
    print("signals")
    _table([(s.name, str(s.shape), s.unit) for s in spec.signals])
    return 0


def _inspect_scenario(name: str) -> int:
    import numpy as np

    from robot_pg.robots import ROBOTS
    from robot_pg.scenarios import SCENARIOS, Path, Targets, Velocity

    scenario = _lookup(SCENARIOS, name, "scenario")
    dt = _lookup(ROBOTS, scenario.robot, "robot").model().opt.timestep
    ref = scenario.reference
    print(
        f"scenario {scenario.name}  duration {scenario.duration:g} s  reference {ref.kind}"
    )
    print(
        f"friction_scale {scenario.friction_scale:g}  payload_kg {scenario.payload_kg:g}  "
        f"pushes {len(scenario.pushes)}"
    )
    for push in scenario.pushes:
        print(f"  push {push.force_n} N at {push.at_s:g} s for {push.duration_s:g} s")
    space = scenario.seed_space
    print(
        f"seed space: base {space.base_m:g} m, yaw {np.degrees(space.base_yaw_rad):g} deg, "
        f"joints {space.joint_rad:g} rad, friction {space.friction[0]:g} to {space.friction[1]:g}"
    )
    if isinstance(ref, Path):
        print(f"path length {ref.length:.4f} m  speed {ref.speed:g} m/s")
    if isinstance(ref, Targets):
        print(f"{len(ref.points)} targets  hold {ref.hold_s:g} s")
    for t in (0.0, 5.0, 10.0):
        tick = np.array([round(t / dt)])
        match ref:
            case Path():
                s = min(ref.speed * t, ref.length)
                x, y, heading = ref.pose(np.array(s))
                print(
                    f"t={t:g} s  s={s:.4f} m  x={x:.4f} m  y={y:.4f} m  heading={heading:.4f} rad"
                )
            case Velocity():
                print(
                    f"t={t:g} s  {ref.signal.name} {ref.values(tick, dt)[0, 0]:.4f} m/s"
                )
            case Targets():
                hold = int(ref.hold_of(tick, dt)[0])
                x, y, z = ref.values(tick, dt)[0]
                print(
                    f"t={t:g} s  target {hold + 1}/{len(ref.points)}  "
                    f"{ref.signal.name} ({x:.4f}, {y:.4f}, {z:.4f}) m"
                )
    return 0


def render(args: argparse.Namespace) -> int:
    import mujoco

    from robot_pg.rendering import frames, write_png
    from robot_pg.robots import ROBOTS

    m = _lookup(ROBOTS, args.robot, "robot").model()
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key < 0:
        valid = [m.key(k).name for k in range(m.nkey)]
        raise SystemExit(f"unknown keyframe {args.keyframe!r}, valid: {valid}")
    out = write_png(args.out, next(frames(m, m.key_qpos[key : key + 1])))
    print(out)
    return 0


def _lookup[T](registry: Mapping[str, T], name: str, what: str) -> T:
    if name not in registry:
        raise SystemExit(f"unknown {what} {name!r}, valid: {list(registry)}")
    return registry[name]


def _range(r: tuple[float, float]) -> str:
    return f"{r[0]:g} to {r[1]:g}"


def _table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print(
            "  "
            + "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-pg")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="print the Python, MuJoCo, NumPy, and CPU in use, then ok"
    ).set_defaults(run=lambda _: doctor())
    p = commands.add_parser("inspect", help="print a robot or a scenario")
    subject = p.add_mutually_exclusive_group(required=True)
    subject.add_argument("subject", nargs="?", help="a robot name")
    subject.add_argument("--scenario", help="a scenario name")
    p.set_defaults(run=inspect)
    p = commands.add_parser("render", help="render one keyframe of a robot to a PNG")
    p.add_argument("--robot", required=True)
    p.add_argument("--keyframe", default="home")
    p.add_argument("--out", required=True)
    p.set_defaults(run=render)
    args = parser.parse_args(argv)
    return args.run(args)
