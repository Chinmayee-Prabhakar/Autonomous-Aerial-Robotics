# References and ownership boundary

The project draws topic inspiration from WPI's [RBE595-F02-ST: Hands-On Autonomous Aerial Robotics](https://pear.wpi.edu/teaching.html), taught by Prof. Nitin J. Sanket. Its P5 final race follows earlier work in estimation, planning, and perception. RBE/CS549: Computer Vision is a separate course.

| Reference section | Concept learned from it | Independent project response |
| --- | --- | --- |
| [RBE595 aerial robotics course](https://pear.wpi.edu/teaching.html): estimation projects | Gyro/accelerometer filtering and quaternion UKF are relevant background. | M2–M4 independently implement the synthetic IMU, project-specific quaternion conventions, Madgwick and UKF estimators, and error metrics. |
| [RBE595 aerial robotics course](https://pear.wpi.edu/teaching.html): planning projects | Geometric planning and quadrotor flight are relevant background. | M6 independently implements a geometric map/clearance model, deterministic RRT*, waypoint interface, and native-Python visualization. |
| [RBE595 aerial robotics course](https://pear.wpi.edu/teaching.html): perception and final race | Camera-based window navigation and a complete obstacle course motivate the mission sequence. | M8–M13 independently implement calibrated projection, pixel-based pose and arm estimation, simulated flight, Blender rendering, and evaluation. |

The tracked project tree contains its own dynamics, control, package layout, TOML parameters, tests, and plotting code. The repository audit found no imported third-party source files or datasets. Standard rigid-body mechanics and quaternion algebra are mathematical background.

## Madgwick algorithm source for M3

- S. O. H. Madgwick, A. J. L. Harrison, and R. Vaidyanathan, “[Estimation of IMU and MARG orientation using a gradient descent algorithm](https://doi.org/10.1109/ICORR.2011.5975346),” *IEEE International Conference on Rehabilitation Robotics*, DOI 10.1109/ICORR.2011.5975346.
- S. O. H. Madgwick, “[An efficient orientation filter for inertial and inertial/magnetic sensor arrays](https://x-io.co.uk/downloads/madgwick_internal_report.pdf),” especially the IMU objective, Jacobian, and normalized gradient correction.

M3 implements the mathematical six-axis method independently. Its gravity objective and Jacobian are derived and numerically checked for this project's ENU world, FLU body, scalar-first body-to-world convention. No source code from either publication or the course materials was copied. The optional accelerometer-magnitude gate is a project-specific safeguard; the default evaluation uses the ungated baseline.

## Quaternion filtering background for M4

- J. Solà, [*Quaternion kinematics for the error-state Kalman filter*](https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf): quaternion conventions, SO(3) exponential/logarithm maps, local errors, and Jacobians.
- E. Kraft, [*A Quaternion-based Unscented Kalman Filter for Orientation Tracking*](https://natanaso.github.io/ece276a2017/ref/2_Kraft_UKF.pdf): quaternion-aware sigma-point attitude filtering.
- E. A. Wan and R. van der Merwe, [*The Unscented Kalman Filter for Nonlinear Estimation*](https://ieeexplore.ieee.org/abstract/document/882463/authors): scaled unscented-transform background.

M4's Python code, right-local covariance transport, test cases, parameter choices, and evaluation pipeline were written independently for this project. The filter uses only IMU readings; recorded truth is confined to evaluation. These references explain mathematical concepts, not a source-code derivation or copied implementation.

## Geometric planning background for M6

- S. Karaman and E. Frazzoli, [*Sampling-based Algorithms for Optimal Motion Planning*](https://arxiv.org/abs/1105.1186), *The International Journal of Robotics Research* (2011): RRT* parent selection, rewiring, and the theoretical setting for asymptotic optimality.

M6 independently implements a finite-budget geometric planner with a fixed neighbor radius. Its returned paths are collision-checked under the declared spherical vehicle envelope, but no asymptotic-optimality or dynamic-feasibility guarantee is claimed.

