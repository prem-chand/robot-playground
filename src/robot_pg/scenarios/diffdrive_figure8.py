from robot_pg.scenarios import Path, Scenario

SCENARIO = Scenario("diffdrive.figure8", 30.0, Path.figure8(radius=1.0, speed=0.5))
