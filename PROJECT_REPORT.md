# Autonomous Aerial Robotics — Project Report

## Objective

The objective was to build and evaluate a simulated quadrotor that can use camera observations to navigate an indoor obstacle course, cross a window containing a rotating arm at a safe time, return through the previous windows, and land at its starting point. The final system should generate its own trajectory references and control inputs, rather than moving a visual drone model along a prescribed animation path.

The final evaluation uses Blender to produce RGB camera images from each new simulated vehicle pose. Python handles perception, planning, feedback control, vehicle dynamics, and collision checks. A separately declared map route is used only to go around the final wall on the return journey.

## Introduction

A window-crossing flight combines several robotics problems that are easy to demonstrate separately but harder to connect. The drone needs an estimate of where an opening is, a way to choose an approach from the correct side, a trajectory that respects its motion limits, and a controller that can follow that trajectory. The rotating arm adds a timing problem: a geometrically clear path can become unsafe while the drone is moving through it.

This project was built in stages, beginning with the quadrotor plant and control loop, then estimation, path planning, camera geometry, window perception, and dynamic-obstacle tracking. The final simulated course has three ordinary windows (alpha, beta, gamma) and a fourth window (delta) containing the rotating arm. After delta, the drone travels around the wall's outside edge before passing back through gamma, beta, and alpha. The room, openings, and drone are also represented in a Blender indoor facility.

The work was inspired by the project sequence in WPI's **RBE595-F02-ST: Hands-On Autonomous Aerial Robotics**, taught by **Prof. Nitin J. Sanket**, including the final drone-race assignment. This implementation is a standalone, simulation-based project with its own controller, geometry, test suite, and experimental evaluation.

## Methodology

### Quadrotor model and control

The vehicle is a six-degree-of-freedom rigid body. Its state includes world position and velocity, an attitude quaternion, and body angular rates. The translational model includes gravity, thrust along the vehicle's body axis, and linear drag. The rotational model uses rigid-body inertia, applied body torques, and gyroscopic coupling. A fixed-step numerical plant advances the state.

A cascaded controller converts position and velocity errors into a desired acceleration and thrust direction. Attitude and body-rate feedback then generate total thrust and three body torques, subject to configured limits. These commands are the actual numerical inputs to the plant. The rendered quadrotor position and attitude come from the saved integrated state, rather than from an artistic spline. The model does not include four separate motors, rotor aerodynamics, wind, or detailed landing-gear contact.

IMU simulation, a Madgwick attitude filter, and a quaternion unscented Kalman filter were developed and evaluated separately. The final integrated mission retains ideal simulated vehicle-state feedback. This is important: the project demonstrates camera-based **target localization**, not complete visual-inertial localization of the vehicle.

### Planning and trajectories

A collision-aware 3D RRT* implementation was tested separately in open, obstacle-detour, altitude-change, and blocked environments. It performs collision checking along candidate edges and includes parent selection, rewiring, and cost updates. The complete window mission does **not** claim to run a new global RRT* search at every stage.

During the mission, perception provides the estimated window pose. The guidance logic selects approach and exit targets relative to that estimate and generates time-parameterized quintic references. The controller follows each trajectory, with another observation and plan at appropriate transitions. Segments can include safe stopping/settling points. This is segment-level image-informed planning, not continuous visual servoing or a continuously optimized global path.

The return route around delta is **mapped**: waypoints describing a known exterior corridor are supplied in advance. The quadrotor still flies that corridor under the same dynamics and controller. It then reacquires gamma, beta, and alpha from camera images on their reverse sides.

### Image perception and timed crossing

The camera uses calibrated pinhole geometry and explicit world, body, and optical frames. Rendered RGB pixels are processed to identify the target window, order its corners, and estimate the relative pose from known window dimensions. The operational detector uses an existing color-frame fit first and tries an adapted appearance-based fit when the first fails. Observations are checked for validity and age before they can authorize a narrow crossing. Simulator object coordinates are reserved for rendering and evaluation, not supplied as true navigation targets.

Inside delta, a one-ended capsule-shaped arm rotates about the center of the stationary opening. The accepted configuration uses a **0.88 m arm length** and **0.10 m radius**. Multiple timestamped RGB observations provide arm-angle and angular-rate estimates. The scheduler predicts where the arm will be during the intended traversal, including the time needed for the vehicle's full modeled envelope to clear the opening. If no safe interval is predicted, the drone waits or aborts. It does not freeze the arm or use the true arm phase to select a crossing time.

Blender supplies a connected indoor environment, camera imagery, and presentation renders. The physical motion, controller inputs, collision geometry, and scored outcomes remain in Python. For the final trials, each image is rendered from the **newly computed** vehicle state: the resulting decisions affect the next flight segment. These are new closed-loop simulations, not replays of the earlier Python-image mission.

### Evaluation and safety checks

The evaluator checks the complete opening-crossing sequence, one and only one delta opening crossing, the distinct exterior bypass, and return-to-home landing. The drone is represented by a **0.25 m spherical collision envelope**. Static wall/frame clearance, the moving arm, and nearby furniture are checked against the actual flight; the moving obstacle is evaluated over the simulated motion between samples using the declared approximation.

For reproducibility, the system stores raw rendered RGB images, capture timestamps, detection and pose results, reference trajectories, applied thrust/body torques, physical states, failure reasons, and independent scoring. The audit recomputes flight motion and collision evidence from these records and can replay the image-to-guidance decisions. Deliberate visual loss and blocked-arm conditions are also tested so that an unsafe situation is not simply classified as a failed successful flight.

## Results

### Integrated Blender-camera mission

The final predefined evaluation comprised **three full missions** using fresh Blender camera images. **All three completed and landed**. Each used **208 new RGB captures**, traversed alpha → beta → gamma → delta → gamma → beta → alpha, crossed delta's opening exactly once, and completed the distinct exterior bypass. Each flight took approximately **189 s of simulated mission time**; rendering took considerably longer in wall-clock time.

| Measure | Result |
| --- | ---: |
| Complete final missions | **3 / 3** |
| Required opening crossings per complete mission | **7 / 7, in order** |
| Delta opening crossings | **1 per mission** |
| Exterior delta bypasses | **1 per mission** |
| Minimum extra clearance from walls/frames | **0.440 m** |
| Minimum extra clearance from the rotating arm | **0.304 m** |
| Minimum extra clearance from furniture | **1.162 m** |
| Safe negative-case outcomes | **2 / 2** |
| Full regression suite | **214 / 214 passed** |

The clearance figures are the smallest results across the three final missions and represent extra separation beyond the modeled vehicle envelope. They are simulation results, not guarantees about real hardware. Tracking, control limits, and all crossing events were independently checked against saved physical traces.

The three successful final trials did not need to wait at delta: a suitable interval was available at their arrival phase. A separately labeled development trial demonstrated a genuine **2.8 s wait**, followed by a safe delta crossing, exterior bypass, reverse flight, and landing. The two negative cases showed safe responses to persistent loss of usable images and to a rotating arm with no acceptable crossing interval.

### Perception findings

The realistic-image trials also exposed errors that would be hidden by a success-only report. Each final flight retained **three accepted window-pose observations outside the strict 12-pixel corner-error criterion**. These arrived after the corresponding static traversal trajectory had already been committed, so they did not change that segment's planned crossing. Independent clearance remained positive, but this does not mean the inaccurate detections were correct or that similar errors would always be harmless.

The final method assumes recognizable colored frame features, calibrated cameras, known target identities and dimensions, and synchronous image capture. The Blender environment is visually more realistic than the earlier native renderer but is still synthetic. The small set of successful missions supports feasibility under these test conditions, not a general reliability estimate.

## Main design choices and trade-offs

**Classical vision instead of a learned detector.** Colored trim and geometry checks make the system relatively simple, transparent, and easy to test. The trade-off is limited appearance robustness and occasional false or inaccurate detections in more varied lighting and viewpoints.

**Quintic segments instead of aggressive continuous replanning.** Generating feasible segments from successive camera estimates keeps trajectory/controller integration manageable and allows explicit clearance validation. It also means the drone can commit to a segment and continue briefly after losing the opening from view, rather than replanning every instant.

**Mapped exterior bypass instead of unknown-space exploration.** Providing the corridor isolates window perception and moving-obstacle timing as the main autonomy challenge. It does not demonstrate map building, SLAM, or independent exploration of an unfamiliar building.

**Direct thrust and torque rather than rotor-level simulation.** This retains physically integrated six-degree-of-freedom motion while avoiding motor-model complexity. It cannot establish rotor-command tracking, wind rejection, or hardware-level flight safety.

**Blender images in a synchronous Python loop.** Rendering newly updated camera poses establishes a causal image-to-control pipeline without requiring real-time Blender integration. Rendering pauses simulated time, so processing delay and real-time performance have not been validated.

## Challenges, corrections, and limitations

A key challenge was maintaining compatible world/body/camera transformations across the Python physics and Blender. Independent projection tests checked the mapping of window corners under front- and reverse-side views. Detector failure on realistic beta images also motivated a fixed, source-declared fallback rule, rather than silently giving the planner true window positions.

At delta, a close hold position clipped the large window in the camera image; the hold distance was adjusted so all frame sides could be observed. A physical yaw correction, based on a prior image-derived target bearing, helped reacquire the opening. Fresh window and arm observations were still required before timed crossing commitment.

Moving the rotating arm to the opening center and enlarging it also required moving the physical crossing corridor; otherwise the vehicle envelope would overlap the pivot region. The revised scene, arm sweep, route, and full mission were revalidated rather than merely redrawn. Evaluation code was corrected where earlier comparisons had confused saved detector outputs with geometric ground truth; the final saved-data audit keeps those categories separate.

Remaining limitations include the ideal vehicle-state feedback, known window geometry and identity, direct thrust/torque actuation, approximate spherical collision checks, mapped bypass, simplified touchdown model, small trial set, and no hardware or real-time camera validation. The independent attitude-filter and RRT* studies should not be described as active components in every step of the final mission.

## Conclusion

The project demonstrates an end-to-end simulated perception-to-control pipeline: new Blender RGB images inform window localization and rotating-arm crossing decisions, the trajectory generator provides feasible references, the cascaded controller produces thrust and torques, and the numerical plant advances the quadrotor through a complete outbound-and-return indoor course. Three predefined full missions landed with positive modeled clearance; the evaluation also records detection mistakes and safe aborts. The result is a reproducible simulation demonstration, with clear boundaries between what is modeled and what has not been tested in the real world.
