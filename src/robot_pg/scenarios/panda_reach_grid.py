from robot_pg.scenarios import Scenario, Targets

SCENARIO = Scenario(
    "panda.reach_grid",
    54.0,
    Targets.grid((0.4, 0.5, 0.6), (-0.2, 0.0, 0.2), (0.3, 0.45, 0.6), hold_s=2.0),
)
