# Robot Playground: Project Concept

**Document type:** Descriptive and procedural\
**Writing standard:** ASD-STE100 (Simplified Technical English)\
**Default physics engine:** MuJoCo

## 1. Purpose

This document gives the concept for the Robot Playground project.

The Robot Playground is a software platform for robot experiments in simulation. It supports wheeled robots, legged robots, and manipulators. It supports classical controllers, RL controllers, and hybrid controllers.

## 2. Abbreviations

| Abbreviation | Meaning |
|---|---|
| API | Application Programming Interface |
| CBF | Control Barrier Function |
| CI/CD | Continuous Integration and Continuous Deployment |
| CMA-ES | Covariance Matrix Adaptation Evolution Strategy |
| CPU | Central Processing Unit |
| DC | Direct Current |
| DoF | Degree of Freedom |
| GPU | Graphics Processing Unit |
| LLM | Large Language Model |
| LQR | Linear Quadratic Regulator |
| MJCF | MuJoCo XML model format |
| MJX | MuJoCo XLA (JAX implementation of MuJoCo) |
| MPC | Model Predictive Control |
| MPM | Material Point Method |
| PD | Proportional-Derivative |
| PPO | Proximal Policy Optimization |
| QP | Quadratic Program |
| RL | Reinforcement Learning |
| SHAC | Short-Horizon Actor-Critic |
| Sim-to-real | Transfer from simulation to real hardware |
| Sysid | System identification |
| URDF | Unified Robot Description Format |
| WASM | WebAssembly |
| WBC | Whole-Body Control |

## 3. Thesis

The Robot Playground has one main function. It makes the claims about each controller measurable, comparable, and falsifiable. This function applies in simulation and on real hardware.

All the capabilities in this document support this function.

## 4. Positioning

A different project with the name "MuJoCo Playground" exists. Google DeepMind released it in 2025. It supplies GPU-parallel RL environments on MJX.

Do these steps:

1. Select a different name for this project.
2. Make this project different from MuJoCo Playground in these three areas:
    - Equal support for classical controllers and learned controllers.
    - Realistic actuator models.
    - Accurate measurement of the sim-to-real gap.

## 5. Architecture

### 5.1 Controller Contract

The controller contract is the most important design decision.

The contract is a standard interface for all controllers. It does not change with the type of controller. Each controller is a block. Each block declares these properties:

- Control rate
- Latency
- Observation specification
- Action specification
- Compute budget

A PD loop, a convex MPC, a QP-based WBC, and a PPO policy all connect through the same interface.

You can connect blocks together. Examples:

- An RL policy sends references to an MPC.
- A residual RL policy adds corrections to an LQR.
- An MPC is the teacher for the distillation of a policy.

Hybrid controllers frequently give the most interesting results. Many frameworks make hybrid controllers difficult to build. This contract makes them easy to build.

### 5.2 Layers

Four layers are around the controller contract. You can replace each layer independently.

| Layer | Contents |
|---|---|
| Robot description | MJCF or URDF, with metadata |
| Actuator models | Refer to Section 6.1 |
| Tasks and scenarios | Terrain, disturbances, objectives |
| Physics backend | MuJoCo on CPU for high fidelity and debug. MJX or MuJoCo Warp for parallel training on many environments. |

### 5.3 Reproducibility

Give each run a content hash. The hash includes these items:

- Robot model
- Controller
- Random seed
- Engine version

With this hash, you can replay each result exactly.

## 6. Core Capabilities

### 6.1 Actuator Models

Most sim-to-real failures occur in the actuators. Most simulators model an actuator as an ideal torque source. This model is not accurate.

Make actuators first-class objects in the platform. Supply a library of actuator models. The fidelity of the models increases in this sequence:

1. Ideal torque source
2. DC motor with a torque-speed curve and current limits
3. Gearbox friction, backlash, and reflected inertia
4. Thermal derating
5. Series elasticity
6. Learned actuator networks (refer to Hwangbo et al., 2019)

Add a sysid pipeline to the library. The pipeline does these steps:

1. It reads real data from a test bench or from the fleet.
2. It fits the parameters of an actuator model to the data.
3. It supplies the fitted model to all robots in the platform.

This capability is also useful to other laboratories.

### 6.2 Reality-Gap Meter

The reality-gap meter measures the difference between simulation and real hardware.

The meter does these steps:

1. It replays real robot data in simulation, open-loop.
2. It measures the divergence over a time horizon.

The meter measures the divergence in these quantities:

- Joint tracking error
- Contact timing
- Energy

With this meter, the sim-to-real gap becomes a number. You can monitor this number across simulator versions, actuator models, and randomization settings.

### 6.3 Controller Zoo

The controller zoo is a set of controllers with standardized metrics. All controllers use the same robot and the same scenario distribution.

| Metric | Description |
|---|---|
| Tracking error | Difference between the commanded motion and the actual motion |
| Cost of transport | Energy for each unit of weight and distance |
| Push-recovery envelope | Largest push that the robot can resist without a fall |
| Failure rate | Rate of failure under a fixed perturbation distribution |
| Compute per step | Compute time for each control step |
| Latency sensitivity | Decrease in performance when latency increases |

The field does not have good data that compares MPC, RL, and hybrid controllers in equal conditions. The controller zoo supplies this data.

### 6.4 Falsification

Do not report only the average reward. Find the conditions that cause each controller to fail.

Use adversarial search for this task. Applicable methods:

- CMA-ES
- Bayesian optimization
- Learned adversary

The search changes these parameters:

- Pushes
- Terrain
- Payload
- Friction
- Latency

Report the failure frontier. Do not report only the mean.

This method gives confidence in a policy before you deploy it to a fleet.

### 6.5 Classical Analysis of Learned Policies

This capability connects controls engineering and RL. Few platforms supply it.

Do these steps:

1. Linearize a trained policy around a nominal gait or posture.
2. Calculate the effective gains from the linear model.
3. Calculate the stability margins.
4. Make Bode plots.
5. Compare the results with an LQR for the same plant.
6. (Optional) Try to certify regions of attraction. Use learned Lyapunov functions or CBFs.

This capability answers a research question: "In controls terms, what did the PPO policy learn?" The answer to this question is applicable for publication.

### 6.6 Engine Randomization

Run the same scenario in different physics engines:

- MuJoCo
- Isaac (PhysX)
- Drake
- Genesis
- Newton

The engines do not always agree. The disagreement is an estimate of the simulation uncertainty.

**Hypothesis:** A policy that is robust across engines transfers to hardware better than a policy from parameter randomization in one engine. You can test this hypothesis.

### 6.7 Differentiable Methods

Use MJX gradients for these tasks:

- Trajectory optimization
- Gradient-based sysid
- Analytic policy gradients (SHAC method)

Compare the gradient-based methods with PPO for each robot class. Measure these two values:

- Sample efficiency
- Final performance

## 7. Long-Term Capabilities

These capabilities are not possible at this time, or they are at the limit of current technology.

### 7.1 One Policy for All Robot Bodies

When the platform has many robots, it becomes a data source. Use this data to train a cross-embodiment controller. The controller is a transformer or a graph network. A morphology description is an input to the policy.

Add procedural robot generation. Example parameters:

- Number of legs
- Link lengths
- Actuator positions

Then you can optimize the morphology and the controller together.

### 7.2 Autonomous Controls Researcher

Connect an LLM agent to the experiment API. The agent does these tasks:

- It proposes reward terms (Eureka method).
- It proposes controller variants and ablations.
- It starts parameter sweeps.
- It reads the falsification results and makes changes.
- It writes reports of the results.

The controller contract and the metrics layer are necessary for this capability. The agent must have a clear objective.

### 7.3 Natural Language to Robot

The user writes a description in natural language. Example: "A hexapod with 3-DoF legs and a 2 kg payload bay, on gravel."

The platform then makes these items:

1. An MJCF model
2. A selection of actuators from the library
3. A task
4. A trained policy
5. A falsification report

### 7.4 Digital-Twin Loop with the Fleet

The loop does these steps:

1. Real deployment data goes back to the platform automatically.
2. The sysid pipeline fits the twin of each robot again.
3. The platform tests each candidate policy against the calibrated twins before rollout.

This loop is CI/CD for robot behavior.

### 7.5 Open-Ended Curricula

The terrain generator and the policy change together (POET method). The generator learns to make terrain that is slightly more difficult than the current limit of the controller.

### 7.6 Hardware-in-the-Loop

Do these steps:

1. Put a real actuator on a dynamometer bench.
2. Connect the actuator to the simulation loop in real time.
3. Use the real actuator in place of its simulated equivalent.

This is an intermediate step between pure simulation and tests on the full robot.

### 7.7 Physics Outside the MuJoCo Capability

MuJoCo does not simulate some physics well. Add these capabilities:

- Granular terrain (sand, mud, snow) with MPM coupling
- Deformable bodies
- Residual neural physics corrections, trained on real data

### 7.8 Browser Access

MuJoCo compiles to WASM. Thus a browser interface is possible.

In the browser, a user can do these tasks:

- Push a robot.
- Change the gains live.
- Compare MPC and RL on the same push.

Applications: teaching and hiring screens.

## 8. Sequence of Work

Keep the scope narrow and deep at the start. Do the phases in this sequence.

#### Phase 1: Foundations

1. Select three robots: a differential-drive base, a humanoid, and a 7-DoF arm.
2. For each robot, make three controllers: one classical, one RL, and one hybrid.
3. Connect all controllers through the controller contract (Section 5.1).
4. Make sure that each run is deterministic and that you can replay it.

#### Phase 2: Realism

1. Add the actuator library (Section 6.1).
2. Add the reality-gap meter (Section 6.2).

**NOTE:** Existing team experience in actuator sysid gives an advantage in this phase.

#### Phase 3: Results

1. Add the falsification engine (Section 6.4).
2. Add the policy-linearization analysis (Section 6.5).

**NOTE:** These two capabilities make the results that other researchers will cite.

#### Phase 4: Long-Term Capabilities

1. Start the capabilities in Section 7. Start with cross-embodiment, the autonomous researcher, and the fleet digital-twin loop.

**NOTE:** These capabilities are easy after Phases 1 to 3. They are almost impossible without Phases 1 to 3.

## 9. Open Decision

Decide the primary use of the platform:

- **Option A:** A personal research and learning platform.
- **Option B:** A platform for an engineering team or for an open-source community.

This decision sets the necessary investment in the contract layer and in API stability at the start.
