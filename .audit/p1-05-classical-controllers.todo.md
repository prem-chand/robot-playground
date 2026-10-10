# P1-05: classical controllers through the contract

Predicate: `uv run pytest` passes with closed-loop gates on the real MuJoCo plant.
diffdrive figure8 cross-track bounded. Panda reach grid final error bounded.
PD block asserts exact kp*e - kd*v.

1. [x] Trail skeleton
2. [x] Read signals, schedule tests, robot registry to name the data shapes.
3. [x] Closed-loop harness in conftest (graph build + scenario drive + metrics).
4. [x] PD block with an exact-value unit gate.
5. [x] diffdrive path follower block, gate on real model.
6. [x] Panda DLS IK block, gate on the reach grid.
7. [x] Full suite, deslop. Commit on request.
