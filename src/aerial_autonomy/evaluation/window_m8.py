"""Seeded M8 camera/pose demonstrations with strictly separate truth scoring."""

import argparse
from collections import Counter
import json
from pathlib import Path
import tomllib

import numpy as np

from aerial_autonomy.perception.camera import CameraCalibration
from aerial_autonomy.perception.window_geometry import StaticWindow, WindowModel
from aerial_autonomy.perception.window_observation import ObservationConfig, observe_window
from aerial_autonomy.perception.window_pose import estimate_window_pose
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import euler_to_quaternion
from aerial_autonomy.visualization.window_m8_plot import save_camera_view, save_metric_plots


def load_experiment(path: str | Path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    c = raw["camera"]
    camera = CameraCalibration(c["fx_px"],c["fy_px"],c["cx_px"],c["cy_px"],
                               c["width_px"],c["height_px"],
                               np.array(c["rotation_camera_to_body"]),
                               np.array(c["translation_camera_in_body_m"]))
    windows = []
    for key in ("window_primary","window_secondary"):
        w = raw[key]
        windows.append(StaticWindow(WindowModel(w["window_id"],w["width_m"],w["height_m"]),
                                    np.array(w["center_world_m"]),np.array(w["right_world"]),
                                    np.array(w["up_world"]),w["frame_thickness_m"]))
    if windows[0].model.window_id == windows[1].model.window_id:
        raise ValueError("window identifiers must be unique")
    experiment = raw["experiment"]
    if (not experiment["seeds"] or len(set(experiment["seeds"])) != len(experiment["seeds"])
            or any(type(seed) is not int or seed < 0 for seed in experiment["seeds"])
            or any(x < 0 for x in experiment["noise_std_px"])
            or any(not 0 <= x <= 1 for x in experiment["dropout_probabilities"])):
        raise ValueError("invalid experiment grid")
    return raw,camera,windows


def state_at(position, yaw_rad=0., pitch_rad=0.) -> VehicleState:
    return VehicleState(np.asarray(position,dtype=float),np.zeros(3),
                        euler_to_quaternion(0.,pitch_rad,yaw_rad),np.zeros(3))


def truth_relative_pose(camera, state, window):
    """Evaluation-only truth; never called from the estimator."""
    rotation_camera_to_world,center_camera_world = camera.camera_to_world(state)
    return (rotation_camera_to_world.T@window.rotation_window_to_world,
            rotation_camera_to_world.T@(window.center_world_m-center_camera_world))


def _safe_float(value):
    return None if value is None or not np.isfinite(value) else float(value)


def evaluate_one(group,case,seed,window,camera,state,observation_config,
                 output_dir: Path,save_view=False):
    observation = observe_window(window,camera,state,observation_config,
                                 np.random.default_rng(seed))
    # The estimator sees only this object, the physical dimensions and calibration.
    estimate = estimate_window_pose(observation,window.model,camera)
    rotation_truth,translation_truth = truth_relative_pose(camera,state,window)
    translation_error = (float(np.linalg.norm(estimate.translation_window_in_camera_m-translation_truth))
                         if estimate.valid else None)
    orientation_error = (float(np.degrees(np.arccos(np.clip(
        (np.trace(estimate.rotation_window_to_camera@rotation_truth.T)-1)/2,-1,1))))
                         if estimate.valid else None)
    row = {"group":group,"case":case,"window_id":window.model.window_id,"seed":seed,
           "noise_std_px":observation_config.pixel_noise_std,
           "corner_dropout_probability":observation_config.corner_dropout_probability,
           "frame_dropout_probability":observation_config.frame_dropout_probability,
           "visible_count":observation.visible_count,"usable_observation":observation.usable,
           "success":estimate.valid,"failure_reason":estimate.reason,
           "translation_error_m":_safe_float(translation_error),
           "orientation_error_deg":_safe_float(orientation_error),
           "reprojection_rmse_px":_safe_float(estimate.reprojection_rmse_px)}
    name = f"{group}_{case}_seed_{seed}"
    measurement = {"window_id":observation.window_id,"timestamp_s":observation.timestamp_s,
                   "visible":observation.visible.tolist(),"visible_count":observation.visible_count,
                   "reason":observation.reason,
                   "pixels_uv":np.where(np.isfinite(observation.pixels_uv),
                                        observation.pixels_uv,np.nan).tolist()}
    # JSON cannot carry NaN: absent image points become null.
    measurement["pixels_uv"] = [[_safe_float(x) for x in pair] for pair in observation.pixels_uv]
    estimate_record = {"window_id":estimate.window_id,"valid":estimate.valid,
                       "reason":estimate.reason,"visible_count":estimate.visible_count,
                       "frame":"window-local to optical camera RDF",
                       "rotation_window_to_camera":(estimate.rotation_window_to_camera.tolist()
                                                    if estimate.valid else None),
                       "translation_window_in_camera_m":(estimate.translation_window_in_camera_m.tolist()
                                                         if estimate.valid else None),
                       "normal_camera":estimate.normal_camera.tolist() if estimate.valid else None,
                       "facing_side":estimate.facing_side,
                       "reprojection_rmse_px":estimate.reprojection_rmse_px}
    truth = {"window_id":window.model.window_id,"rotation_window_to_camera":rotation_truth.tolist(),
             "translation_window_in_camera_m":translation_truth.tolist(),
             "window_center_world_m":window.center_world_m.tolist(),
             "camera_body_position_world_m":state.position_m.tolist(),
             "camera_body_quaternion_bw":state.quaternion_bw.tolist()}
    for folder,record in (("observations",measurement),("estimates",estimate_record),
                          ("evaluation_truth",truth)):
        target = output_dir/folder
        target.mkdir(parents=True,exist_ok=True)
        (target/f"{name}.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",
                                            encoding="utf-8")
    if save_view:
        ideal,_,_ = camera.project(camera.world_to_camera(state,window.world_corners_m()))
        save_camera_view(output_dir/"figures"/f"{name}.png",camera,observation,ideal,
                         estimate,window.model,f"M8 {case}: {window.model.window_id}")
    return row


def _sequence_from_m7(output_dir: Path,camera,window):
    source = Path("outputs/trajectory_m7/open/seed_11/flight.npz")
    if not source.exists():
        return {"available":False,"reason":"saved M7 open flight not present"}
    flight = np.load(source)
    positions = flight["actual_position_m"]
    angles = flight["actual_euler_rad"]
    times = flight["time_s"]
    sequence_window = StaticWindow(WindowModel("m7_route_demo",1.4,1.1),
                                   np.array([6.5,3.,1.]),np.array([0.,-1.,0.]),
                                   np.array([0.,0.,1.]),window.frame_thickness_m)
    frames = []
    for frame_index,target_x in enumerate(np.linspace(1,4.5,9)):
        idx = int(np.argmin(abs(positions[:,0]-target_x)))
        state = VehicleState(positions[idx],np.zeros(3),
                             euler_to_quaternion(*angles[idx]),np.zeros(3))
        observation = observe_window(sequence_window,camera,state,ObservationConfig(),
                                     np.random.default_rng(100+frame_index),float(times[idx]))
        estimate = estimate_window_pose(observation,sequence_window.model,camera)
        ideal,_,_ = camera.project(camera.world_to_camera(state,sequence_window.world_corners_m()))
        name = f"frame_{frame_index:02d}.png"
        save_camera_view(output_dir/"sequence"/name,camera,observation,ideal,estimate,
                         sequence_window.model,f"M7 saved flight, t={times[idx]:.2f} s, x={positions[idx,0]:.2f} m")
        frames.append({"image":f"sequence/{name}","m7_index":idx,
                       "time_s":float(times[idx]),"body_x_m":float(positions[idx,0]),
                       "valid":estimate.valid,"visible_count":observation.visible_count,
                       "projected_width_px":float(np.max(ideal[:,0])-np.min(ideal[:,0]))})
    try:
        from PIL import Image
        images = [Image.open(output_dir/frame["image"]).convert("RGB") for frame in frames]
        images[0].save(output_dir/"window_approach.gif",save_all=True,
                       append_images=images[1:],duration=380,loop=0)
        for image in images:
            image.close()
    except ImportError:
        pass
    return {"available":True,"source":str(source),"frames":frames,
            "gif_available":(output_dir/"window_approach.gif").exists()}


def run(config_path: str | Path,output_dir: str | Path):
    raw,camera,(primary,secondary) = load_experiment(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"config_snapshot.json").write_text(json.dumps(raw,indent=2)+"\n",encoding="utf-8")
    rows = []
    def add(group,case,seed,window,state,noise=0.,corner_dropout=0.,frame_dropout=0.,view=False):
        rows.append(evaluate_one(group,case,seed,window,camera,state,
                                 ObservationConfig(noise,corner_dropout,frame_dropout),
                                 output_dir,view))
    frontal = state_at([0.,0.,1.5])
    oblique = state_at([0.,-.8,1.5],.3,.08)
    reverse = state_at([8.,0.,1.5],np.pi)
    add("geometry","frontal",11,primary,frontal,view=True)
    add("geometry","oblique",11,primary,oblique,view=True)
    add("geometry","reverse",11,primary,reverse,view=True)
    add("multiple","alpha",11,primary,frontal,view=True)
    add("multiple","beta",11,secondary,frontal,view=True)
    for noise in raw["experiment"]["noise_std_px"]:
        for seed in raw["experiment"]["seeds"]:
            add("noise",f"sd_{noise:g}",seed,primary,oblique,noise,
                view=(seed==11 and noise>0))
    for probability in raw["experiment"]["dropout_probabilities"]:
        for seed in raw["experiment"]["seeds"]:
            add("dropout",f"p_{probability:g}",seed,primary,frontal,
                corner_dropout=probability,
                view=(seed==11 and probability in (.5,1.)))
    add("visibility","behind",11,primary,state_at([0,0,1.5],np.pi),view=True)
    add("visibility","outside",11,primary,state_at([0,8,1.5]),view=True)
    add("visibility","frame_dropout",11,primary,frontal,frame_dropout=1.,view=True)
    # Deliberately use a shifted camera, so at least one corner clips at the edge.
    add("visibility","partial",11,primary,state_at([0,-2.5,1.5]),view=True)
    save_metric_plots(output_dir/"figures",rows)
    sequence = _sequence_from_m7(output_dir,camera,primary)
    def aggregate(subset):
        successful = [r for r in subset if r["success"]]
        return {"trials":len(subset),"usable_observations":sum(r["usable_observation"] for r in subset),
                "successes":len(successful),"observation_rate":sum(r["usable_observation"] for r in subset)/len(subset),
                "success_rate":len(successful)/len(subset),
                "mean_translation_error_m":float(np.mean([r["translation_error_m"] for r in successful])) if successful else None,
                "max_translation_error_m":max((r["translation_error_m"] for r in successful),default=None),
                "mean_orientation_error_deg":float(np.mean([r["orientation_error_deg"] for r in successful])) if successful else None,
                "max_orientation_error_deg":max((r["orientation_error_deg"] for r in successful),default=None),
                "mean_reprojection_rmse_px":float(np.mean([r["reprojection_rmse_px"] for r in successful])) if successful else None,
                "failure_reasons":dict(Counter(r["failure_reason"] for r in subset if not r["success"]))}
    metrics = {"all":aggregate(rows),"by_group":{},"noise_levels":{},"dropout_levels":{},
               "accuracy_gates":raw["experiment"],"m7_sequence":sequence}
    for group in sorted({r["group"] for r in rows}):
        metrics["by_group"][group] = aggregate([r for r in rows if r["group"]==group])
    for level in raw["experiment"]["noise_std_px"]:
        metrics["noise_levels"][str(level)] = aggregate([r for r in rows if r["group"]=="noise" and r["noise_std_px"]==level])
    for level in raw["experiment"]["dropout_probabilities"]:
        metrics["dropout_levels"][str(level)] = aggregate([r for r in rows if r["group"]=="dropout" and r["corner_dropout_probability"]==level])
    target = [r for r in rows if r["group"]=="noise" and r["noise_std_px"]==1.5]
    metrics["predeclared_gate_pass"] = bool(target and all(r["success"] and
        r["translation_error_m"] < raw["experiment"]["max_translation_error_m_at_1_5_px"] and
        r["orientation_error_deg"] < raw["experiment"]["max_orientation_error_deg_at_1_5_px"] and
        r["reprojection_rmse_px"] < raw["experiment"]["max_reprojection_rmse_px_at_1_5_px"]
        for r in target))
    (output_dir/"trial_metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output_dir/"aggregate_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/window_m8.toml")
    parser.add_argument("--output-dir",default="outputs/window_m8")
    args = parser.parse_args()
    metrics = run(args.config,args.output_dir)
    print(json.dumps({"all":metrics["all"],"predeclared_gate_pass":metrics["predeclared_gate_pass"]},indent=2))


if __name__ == "__main__":
    main()
