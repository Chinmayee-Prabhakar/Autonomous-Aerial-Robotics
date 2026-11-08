"""Preset rendered-image detector study with separate M8 oracle diagnostics."""

import json
from pathlib import Path

import numpy as np

from aerial_autonomy.evaluation.window_m8 import state_at
from aerial_autonomy.perception.window_m9_detect import ColorWindowDetector
from aerial_autonomy.perception.window_m9_guidance import ImageWindowGuidance, TargetModel
from aerial_autonomy.perception.window_m9_render import RenderSettings, WindowImageRenderer
from aerial_autonomy.perception.window_observation import ObservationConfig, observe_window
from aerial_autonomy.perception.window_pose import estimate_window_pose, reproject_estimate
from aerial_autonomy.visualization.window_m9_plot import save_camera_examples


def run_detector_study(raw,vehicle,limits,camera,world,targets,colors,output_dir):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    detector=ColorWindowDetector(colors)
    renderer=WindowImageRenderer(camera,world,colors)
    cases=[
        ("frontal_far","alpha",[0,0,1.6],0,1.,3.,False,"positive_normal",None),
        ("frontal_near","alpha",[2.7,0,1.6],0,1.,3.,False,"positive_normal",None),
        ("oblique","beta",[6.5,0,1.7],.10,1.,3.,False,"positive_normal",None),
        ("altitude","gamma",[11.5,-.25,1.65],-.08,1.,3.,False,"positive_normal",None),
        ("reverse","alpha",[8,0,1.6],np.pi,1.,3.,False,"negative_normal",None),
        ("lowlight","alpha",[0,0,1.6],0,.7,8.,False,"positive_normal",None),
        ("bright_blurred","alpha",[0,0,1.6],0,1.2,5.,True,"positive_normal",None),
        ("noisy","alpha",[0,0,1.6],0,.9,12.,False,"positive_normal",None),
        ("partial_occlusion","alpha",[0,0,1.6],0,1.,3.,False,"positive_normal","occlude"),
        ("out_of_frame","alpha",[0,0,1.6],.9,1.,3.,False,"positive_normal",None),
        ("solid_color_false_positive","alpha",[0,0,1.6],0,1.,0.,False,"positive_normal","solid"),
        ("blank_false_positive","alpha",[0,0,1.6],0,1.,0.,False,"positive_normal","blank"),
    ]
    rows=[];examples={}
    for index,(name,target_id,position,yaw,light,noise,blur,side,alter) in enumerate(cases):
        state=state_at(position,yaw)
        image=renderer.render(state,RenderSettings(noise,light,blur),np.random.default_rng(100+index))
        if alter=="occlude":
            image[:,0:150]=np.array([34,43,55],dtype=np.uint8)
        elif alter=="solid":
            image[:]=np.array([34,43,55],dtype=np.uint8)
            image[70:170,100:220]=colors[target_id]
        elif alter=="blank":
            image[:]=np.array([34,43,55],dtype=np.uint8)
        case_targets=dict(targets)
        if side=="negative_normal":
            old=case_targets[target_id]
            case_targets[target_id]=TargetModel(old.model,old.frame_thickness_m,
                old.outer_width_m,old.outer_height_m,side)
        guidance=ImageWindowGuidance(camera,detector,case_targets,world.workspace,
            vehicle.vehicle,limits,world.vehicle_radius_m,
            raw["workspace"]["planning_clearance_m"])
        fix=guidance.perceive(image,state,target_id,0.)
        wall=world.wall_for_id(target_id)
        true_pixels,_,inside=camera.project(camera.world_to_camera(
            state,wall.window.world_corners_m()))
        corner_error=(float(np.sqrt(np.mean(np.sum((fix.detection.corners_uv-true_pixels)**2,axis=1))))
                      if fix.detection.valid and np.all(inside) else None)
        translation_error=(float(np.linalg.norm(fix.estimated_window.center_world_m-
            wall.window.center_world_m)) if fix.valid else None)
        rotation_error=(float(np.degrees(np.arccos(np.clip((np.trace(
            fix.estimated_window.rotation_window_to_world@
            wall.window.rotation_window_to_world.T)-1)/2,-1,1)))) if fix.valid else None)
        # Accepted M8 oracle is an evaluation-only comparator and never enters guidance.
        oracle_observation=observe_window(wall.window,camera,state,ObservationConfig(),
                                          np.random.default_rng(100+index))
        oracle=estimate_window_pose(oracle_observation,wall.window.model,camera)
        true_camera_rotation=camera.camera_to_world(state)[0]
        true_rotation_camera=true_camera_rotation.T@wall.window.rotation_window_to_world
        oracle_translation_error=(float(np.linalg.norm(
            oracle.translation_window_in_camera_m-
            camera.world_to_camera(state,wall.window.center_world_m[None,:])[0]))
            if oracle.valid else None)
        oracle_orientation_error=(float(np.degrees(np.arccos(np.clip((np.trace(
            oracle.rotation_window_to_camera@true_rotation_camera.T)-1)/2,-1,1))))
            if oracle.valid else None)
        row={"case":name,"target_id":target_id,"seed":100+index,
             "illumination":light,"rgb_noise_std":noise,"blur":blur,
             "alteration":alter,"detected":fix.detection.valid,"detection_reason":fix.detection.reason,
             "pose_valid":fix.valid,"pose_reason":fix.reason,
             "corner_rmse_px":corner_error,"translation_error_m":translation_error,
             "orientation_error_deg":rotation_error,
             "reprojection_rmse_px":fix.estimate.reprojection_rmse_px if fix.valid else None,
             "oracle_corner_pose_valid":oracle.valid,
             "oracle_reprojection_rmse_px":oracle.reprojection_rmse_px if oracle.valid else None,
             "oracle_translation_error_m":oracle_translation_error,
             "oracle_orientation_error_deg":oracle_orientation_error,
             "false_positive":bool(alter in ("solid","blank") and fix.detection.valid)}
        rows.append(row)
        reprojection=(reproject_estimate(fix.estimate,wall.window.model,camera)
                      if fix.valid else None)
        examples[name]={"image":image,"detection":fix.detection,
                        "estimated_reproject_uv":reprojection,"target_id":target_id}
    save_camera_examples(examples,output_dir/"figures")
    (output_dir/"detector_cases.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    return rows,examples
