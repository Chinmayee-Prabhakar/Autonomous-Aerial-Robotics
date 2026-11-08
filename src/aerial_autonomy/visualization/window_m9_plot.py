"""M9 rendered-camera overlays, animated flight views, and safety figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def camera_overlay(image_rgb,detected,estimated_uv,title,status):
    image=Image.fromarray(image_rgb).resize((image_rgb.shape[1]*2,image_rgb.shape[0]*2))
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,image.width,39),fill=(5,12,23))
    draw.text((9,6),title,fill="white")
    draw.text((9,22),status,fill="#ffe680" if not detected.valid else "#a8f0b6")
    if detected.valid:
        corners=detected.corners_uv*2
        loop=[tuple(x) for x in corners]+[tuple(corners[0])]
        draw.line(loop,fill="#ffe066",width=3)
        for label,point in zip(("BL","BR","TR","TL"),corners):
            u,v=map(float,point)
            draw.ellipse((u-5,v-5,u+5,v+5),outline="#ffe066",width=2)
            draw.text((u+5,v-17),label,fill="#ffe066")
    if estimated_uv is not None:
        corners=estimated_uv*2
        loop=[tuple(x) for x in corners]+[tuple(corners[0])]
        draw.line(loop,fill="#ff70c2",width=2)
        for u,v in corners:
            draw.line((u-5,v,u+5,v),fill="#ff70c2",width=2)
            draw.line((u,v-5,u,v+5),fill="#ff70c2",width=2)
    return image


def save_camera_examples(frames,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():
        det=frame["detection"]
        status=(f"detected conf={det.confidence:.2f}; yellow=detected, pink=reprojected"
                if det.valid else f"rejected: {det.reason}")
        camera_overlay(frame["image"],det,frame.get("estimated_reproject_uv"),
            f"M9 {name}: {frame['target_id']}",status).save(output_dir/f"camera_{name}.png")


def save_camera_gif(frames,path):
    images=[]
    for frame in frames:
        det=frame["detection"]
        title=f"t={frame['time_s']:.1f} s | {frame['state']} | target {frame['target_id']}"
        status=("detected; yellow corners, pink pose reprojection" if det.valid
                else f"no valid detection: {det.reason}")
        images.append(camera_overlay(frame["image"],det,frame["estimated_reproject_uv"],
                                     title,status))
    images[0].save(path,save_all=True,append_images=images[1:],duration=170,loop=0,
                   optimize=True)
    for image in images:
        image.close()


def _wall_art(ax,world,colors):
    for wall in world.walls:
        w=wall.window
        transform=w.rotation_window_to_world
        center=w.center_world_m
        panels=[]
        face=w.frame_thickness_m/2
        for box in wall.local_boxes():
            lo,hi=np.asarray(box.minimum),np.asarray(box.maximum)
            local=np.array([[lo[0],lo[1],face],[hi[0],lo[1],face],
                            [hi[0],hi[1],face],[lo[0],hi[1],face]])
            panels.append(center+local@transform.T)
        ax.add_collection3d(Poly3DCollection(panels,facecolors="#718395",
                           edgecolors="#6b8093",alpha=.10,linewidths=.35))
        corners=w.world_corners_m()
        loop=np.vstack((corners,corners[0]))
        color=np.asarray(colors[w.model.window_id])/255
        ax.plot(loop[:,0],loop[:,1],loop[:,2],color=color,linewidth=3.5)
        ax.text(center[0],center[1],center[2]+w.model.height_m/2+.18,
                w.model.window_id,color=color,fontsize=9)


def _base_3d(ax,world,colors):
    _wall_art(ax,world,colors)
    ax.set_xlim(-.5,17.5);ax.set_ylim(-2.7,2.7);ax.set_zlim(0,3.4)
    ax.set_xlabel("east x (m)");ax.set_ylabel("north y (m)")
    ax.set_zlabel("up z (m)",labelpad=-2)
    ax.set_box_aspect((18,5.4,3.4))
    ax.view_init(elev=20,azim=-65)


def save_trajectory_3d(actual,world,colors,crossings,path):
    positions=np.array([row["position_m"] for row in actual])
    reference=np.array([row["reference_position_m"] for row in actual])
    fig=plt.figure(figsize=(13,6),dpi=135)
    ax=fig.add_subplot(projection="3d")
    _base_3d(ax,world,colors)
    ax.plot(reference[:,0],reference[:,1],reference[:,2],linestyle="--",
            color="#b552be",linewidth=1.3,label="M7 reference")
    ax.plot(positions[:,0],positions[:,1],positions[:,2],color="#0a729a",
            linewidth=2.3,label="actual M1 flight")
    for event in crossings:
        wall=world.wall_for_id(event["window_id"])
        p=wall.window.center_world_m
        ax.scatter(*p,color="#1aab56",s=55)
    ax.legend(loc="upper left")
    ax.set_title("M9 continuous image-guided outbound flight: alpha → beta → gamma")
    fig.tight_layout();fig.savefig(path,bbox_inches="tight",pad_inches=.4);plt.close(fig)


def save_3d_gif(actual,world,colors,crossings,path):
    times=np.array([row["time_s"] for row in actual])
    points=np.array([row["position_m"] for row in actual])
    reference=np.array([row["reference_position_m"] for row in actual])
    frame_times=np.arange(0,times[-1]+.01,1.25)
    indices=np.unique(np.searchsorted(times,frame_times).clip(0,len(times)-1))
    images=[]
    fig=plt.figure(figsize=(11,5.7),dpi=92)
    for index in indices:
        fig.clear()
        ax=fig.add_subplot(projection="3d")
        _base_3d(ax,world,colors)
        ax.plot(reference[:index+1,0],reference[:index+1,1],reference[:index+1,2],
                color="#bb5ac7",linestyle="--",linewidth=1.25,label="M7 reference")
        ax.plot(points[:index+1,0],points[:index+1,1],points[:index+1,2],
                color="#008fc5",linewidth=2.2,label="actual trail")
        ax.scatter(*points[index],color="#ffb800",s=85,label="vehicle")
        row=actual[index]
        ax.set_title(f"M9 t={times[index]:.1f} s | {row['phase']} | target {row['target_id']}")
        ax.legend(loc="upper left",fontsize=8)
        fig.canvas.draw()
        rgba=np.asarray(fig.canvas.buffer_rgba())
        height,width=rgba.shape[:2]
        # The 3D course is long and narrow; trim unused side margins so the
        # openings, vehicle, and trail remain readable in the animated view.
        images.append(Image.fromarray(rgba[:,:,:3].copy()).crop(
            (int(width*.22),0,int(width*.88),height)))
    plt.close(fig)
    images[0].save(path,save_all=True,append_images=images[1:],duration=180,loop=0,
                   optimize=True)
    for image in images:
        image.close()
    return [float(times[i]) for i in indices]


def save_perception_plot(scoring,path):
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,dpi=135)
    specs=(("corner_rmse_px","Detected-corner RMSE (px)"),
           ("translation_error_m","Window-center error (m)"),
           ("orientation_error_deg","Window orientation error (deg)"))
    colors={"alpha":"#e65a50","beta":"#3ba864","gamma":"#5575dc"}
    for axis,(key,label) in zip(axes,specs):
        for target in colors:
            rows=[r for r in scoring if r["target_id"]==target and r[key] is not None]
            axis.scatter([r["time_s"] for r in rows],[r[key] for r in rows],
                         s=13,label=target,color=colors[target])
        axis.set_ylabel(label);axis.grid(alpha=.3)
    axes[0].legend()
    axes[-1].set_xlabel("simulated time (s)")
    fig.suptitle("M9 image-detected geometry; truth used only for evaluation")
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_safety_plot(actual,crossings,path):
    t=np.array([r["time_s"] for r in actual])
    speed=np.array([np.linalg.norm(r["velocity_m_s"]) for r in actual])
    accel=np.array([np.linalg.norm(r["acceleration_m_s2"]) for r in actual])
    series=(("tracking_error_m","Tracking error (m)"),
            ("clearance_m","Actual clearance beyond 0.25 m (m)"),
            (None,"Speed (m/s)"),(None,"Acceleration (m/s²)"),
            ("thrust_n","Thrust (N)"),("tilt_deg","Tilt (deg)"))
    fig,axes=plt.subplots(6,1,figsize=(12,12),sharex=True,dpi=125)
    for i,(key,label) in enumerate(series):
        values=speed if i==2 else accel if i==3 else [r[key] for r in actual]
        axes[i].plot(t,values,color="#226c9e",linewidth=.9)
        axes[i].set_ylabel(label)
        axes[i].grid(alpha=.25)
        for event in crossings:
            axes[i].axvline(event["plane_crossing_time_s"],color="#d85558",alpha=.65)
    axes[-1].set_xlabel("simulated time (s)")
    fig.suptitle("M9 sampled M1 flight and whole-segment wall clearance; red lines = crossings")
    fig.tight_layout();fig.savefig(path);plt.close(fig)
