# Autonomous Aerial Robotics

This project is a Python simulation of a quadrotor that finds and flies through several window openings inside a Blender-rendered indoor facility. The final opening contains a rotating arm. The drone crosses it once, flies around the outside of that wall, returns through the earlier windows in reverse order, and lands near where it started.

The important part is that the full mission uses **new Blender camera images from the drone's current simulated position** to find the openings and time the rotating-arm crossing. The flight is not a prerecorded path played back in Blender. Python generates the trajectory references and controller commands, integrates the vehicle dynamics, and checks for modeled collisions.

## What is in the project?

- **Flight and control:** six-degree-of-freedom quadrotor model, cascaded position/attitude/rate control, thrust and body-torque commands.
- **Estimation and planning:** separate IMU/Madgwick/UKF experiments, 3D RRT* planning experiments, and quintic flight trajectories.
- **Vision and navigation:** RGB window detection, camera-based relative pose estimation, rotating-arm tracking, and a mission state machine.
- **Rendering:** a connected indoor Blender scene with four openings, a rotating arm, and a quadrotor that follows its simulated pose.
- **Evaluation:** saved images, physical flight traces, crossing events, control signals, failure cases, and reproducibility checks.

## Set up

The commands below use **Windows PowerShell**, Python 3.11, and Blender 3.5. Run them from the repository's root directory. Change the Blender executable path if your installation is elsewhere.

```powershell
py -3.11 -m pip install "numpy>=1.24,<3" "matplotlib>=3.7,<4" "Pillow>=10,<12"
$env:PYTHONPATH = 'src'
$blender = 'C:\Program Files\Blender Foundation\Blender 3.5\blender.exe'

# Check the code first. Generated outputs are not required.
py -3.11 -m unittest discover -s tests -q
```

## Run the complete mission

There are two useful ways to run it. **The Python-only mission** is the quickest way to reproduce the complete flight and physical results. **The Blender-camera mission** uses new rendered RGB images during the flight and takes much longer to run.

### 1. Python-only complete mission

```powershell
py -3.11 -m aerial_autonomy.evaluation.mission_m11 --config configs/mission_m11.toml --output-dir outputs/mission_m11
py -3.11 -m aerial_autonomy.visualization.mission_m11_plot --config configs/mission_m11.toml --output-dir outputs/mission_m11
```

Open `outputs/mission_m11/trial_101/figures/mission_3d.gif` to see the full outbound-and-return flight. Its `physical_trace.npz` stores the simulated state and applied control inputs. This run uses the lightweight Python camera renderer, not Blender images for guidance.

### 2. Build the indoor Blender scene

Run this after both Python-only commands: the scene-plan builder reads the saved M11 flight **and its visualization timeline**. You do **not** need a prebuilt `.blend` file from the repository.

```powershell
py -3.11 -m aerial_autonomy.realism.bridge --config configs/realism_m12.toml --output-dir outputs/realism_m12
& $blender -b --factory-startup --python tools/m12/blender_scene.py -- --plan outputs/realism_m12/scene_plan.json --output-blend outputs/realism_m12/indoor_facility.blend --preview-dir outputs/realism_m12/previews
```

Look at `outputs/realism_m12/previews/environment_isometric.png` to see the rooms and windows. The `.blend` file is generated inside `outputs/`, not tracked as source.

### 3. Run a new Blender-image-guided flight

Use a new output directory for each run. The runner writes image-capture logs as it goes, so **do not rerun into an existing trial folder**.

```powershell
py -3.11 -m aerial_autonomy.realism.mission_m13 --config configs/mission_m13.toml --blender $blender --trial-kind final --seed 101 --phase-rad 0.9 --output-dir outputs/mission_m13/final_trial_101

# Independently inspect the saved physical flight and image decisions.
py -3.11 -m aerial_autonomy.realism.mission_m13_audit outputs/mission_m13/final_trial_101
py -3.11 -m aerial_autonomy.realism.mission_m13_evaluate outputs/mission_m13/final_trial_101
```

For a finished video/GIF set, first run the visual-loss example and then the presenter:

```powershell
py -3.11 -m aerial_autonomy.realism.mission_m13 --config configs/mission_m13.toml --blender $blender --trial-kind negative --seed 201 --forced-visual-loss --output-dir outputs/mission_m13/negative_visual_loss
py -3.11 -m aerial_autonomy.realism.mission_m13_present outputs/mission_m13/final_trial_101 outputs/mission_m13/visuals --blender $blender --negative-dir outputs/mission_m13/negative_visual_loss
```

Open `outputs/mission_m13/visuals/full_indoor_m13.gif` for the complete mission, `onboard_rgb_used.gif` for actual images observed during flight, `delta_tracking_crossing.gif` for the rotating arm, and `controls_tracking_clearance.png` for the generated thrust, torques, tracking errors, and collision margins.

If you want all predefined final conditions, their seeds and settings are in `configs/mission_m13.toml`. The audit and evaluation modules can run again on saved results **without rendering them again**. Rendering is synchronous and slower than real time; this is an offline simulation.

## Where to look in the code

| Part | Main location |
| --- | --- |
| Quadrotor dynamics and cascaded controller | `src/aerial_autonomy/simulation/`, `src/aerial_autonomy/control/` |
| RRT* and quintic trajectory experiments | `src/aerial_autonomy/planning/`, `src/aerial_autonomy/trajectory/` |
| Window and rotating-arm perception | `src/aerial_autonomy/perception/` |
| Complete Python-only mission | `src/aerial_autonomy/evaluation/mission_m11.py` |
| Blender scene and camera renderer | `tools/m12/`, `src/aerial_autonomy/realism/bridge.py` |
| New Blender-guided complete mission | `src/aerial_autonomy/realism/mission_m13.py` |
| Saved-flight checks | `src/aerial_autonomy/realism/mission_m13_audit.py` |

## Results and limits

The predefined Blender-camera evaluation completed **3/3 full missions** with all seven opening traversals, one rotating-arm crossing, an exterior bypass, and return-to-start landing. Two separate deliberately unsafe cases aborted without a modeled collision. The full source-only test suite passed **214/214 tests**. Details, including detection errors, are in [PROJECT_REPORT.md](PROJECT_REPORT.md).

This is **not** a hardware test or a real-time flight system. The windows have known identities and dimensions; the low-level controller uses ideal simulated vehicle-state feedback; the bypass is a known map route; and the collision and landing models are simplified. The separately tested RRT* and attitude filters are not claimed to be active inside every final-mission step.

## Course inspiration

The project was inspired by the project sequence in **RBE595-F02-ST: Hands-On Autonomous Aerial Robotics**, taught by **Prof. Nitin J. Sanket** at **Worcester Polytechnic Institute**, especially **P5: The Final Race!** It is an independently implemented, simulation-based extension, not a reproduction of a course submission. See the [official WPI course listing](https://pear.wpi.edu/teaching.html).
