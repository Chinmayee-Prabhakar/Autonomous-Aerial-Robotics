"""M10 review figures and actual-attitude quadrotor animations."""

from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image,ImageDraw

from aerial_autonomy.simulation.rotation import quaternion_to_matrix


def quadrotor_geometry(position_m,quaternion_bw):
    """FLU body model; rotor disks are display only, not collision geometry."""
    center=np.asarray(position_m,dtype=float)
    rotation=quaternion_to_matrix(np.asarray(quaternion_bw,dtype=float))
    def world(local):return center+np.asarray(local)@rotation.T
    hubs=np.array([[x,y,0.] for x in (.145,-.145) for y in (.145,-.145)])
    angles=np.linspace(0,2*math.pi,17)
    rings=[world(h+np.column_stack((.075*np.cos(angles),.075*np.sin(angles),
                                     np.zeros(len(angles))))) for h in hubs]
    return {"center":center,"front":world([.12,0,.025]),
            "arms":[world(np.array([[0,0,0],h])) for h in hubs],
            "rings":rings,"body":world(np.array([[-.075,-.06,-.025],[.075,-.06,-.025],
                [.075,.06,-.025],[-.075,.06,-.025]]))}


def _quad(ax,row):
    mesh=quadrotor_geometry(row["position_m"],row["quaternion_bw"])
    ax.add_collection3d(Poly3DCollection([mesh["body"]],facecolor="#244c67",
                       edgecolor="#0c2434",alpha=.95))
    for arm in mesh["arms"]:
        ax.plot(*arm.T,color="#143040",linewidth=3)
    for ring in mesh["rings"]:
        ax.plot(*ring.T,color="#1ab4d2",linewidth=1.3)
    ax.scatter(*mesh["front"],color="#ff5144",s=28)


def _scene(ax,mission,time_s):
    wall=mission.world.walls[0];window=wall.window
    face=window.frame_thickness_m/2
    panels=[]
    for box in wall.local_boxes():
        lo=np.array(box.minimum);hi=np.array(box.maximum)
        local=np.array([[lo[0],lo[1],face],[hi[0],lo[1],face],
                        [hi[0],hi[1],face],[lo[0],hi[1],face]])
        panels.append(window.center_world_m+local@window.rotation_window_to_world.T)
    ax.add_collection3d(Poly3DCollection(panels,facecolor="#8b9aa9",
                       edgecolor="#9daeb9",alpha=.13))
    corners=window.world_corners_m();loop=np.vstack((corners,corners[0]))
    ax.plot(*loop.T,color="#e06b6b",linewidth=3)
    a,b=mission.arm.endpoints_world(time_s)
    ax.plot(*np.array([a,b]).T,color="#e4a61e",linewidth=8,solid_capstyle="round")
    ax.scatter(*a,color="#f83ad9",s=34)
    ax.set_xlim(17.1,22.4);ax.set_ylim(-1.6,1.6);ax.set_zlim(.55,2.65)
    ax.set_box_aspect((5.3,3.2,2.1));ax.set_xlabel("east x (m)")
    ax.set_ylabel("north y (m)");ax.set_zlabel("up z (m)")
    ax.view_init(elev=22,azim=-66)


def save_camera_gif(mission,path):
    images=[]
    for row in mission.frames:
        rgb=row["image"];image=Image.fromarray(rgb).resize((640,480))
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,639,47),fill=(8,17,28))
        d=row["detection"];e=row["estimate"]
        draw.text((8,5),f"M10 t={row['time_s']:.2f}s  {row['phase']}",fill="white")
        detail=(f"angle {math.degrees(e.angle_rad):+.1f} deg | rate {e.angular_velocity_rad_s:+.3f} rad/s"
                if e.valid else f"arm estimate unavailable: {e.reason}")
        draw.text((8,25),detail,fill="#fce682")
        if d.valid:
            # The overlay is derived again from RGB regions, never truth pose.
            yy,xx=np.nonzero((rgb[:,:,0]>160)&(rgb[:,:,1]>125)&(rgb[:,:,2]<115))
            hy,hx=np.nonzero((rgb[:,:,0]>155)&(rgb[:,:,2]>140)&(rgb[:,:,1]<115))
            if len(xx) and len(hx):
                pivot=np.array([np.median(hx),np.median(hy)])
                points=np.column_stack((xx,yy))
                distances=np.linalg.norm(points-pivot,axis=1)
                tip=np.median(points[distances>=np.quantile(distances,.8)],axis=0)
                draw.line((*(pivot*2),*(tip*2)),fill="#43ff8f",width=3)
                draw.ellipse((pivot[0]*2-4,pivot[1]*2-4,pivot[0]*2+4,pivot[1]*2+4),
                             outline="#43ff8f",width=2)
            draw.text((8,457),f"image arm detection valid | confidence {d.confidence:.2f}",fill="#b8ffbd")
        if row["selected"] is not None:
            draw.text((315,457),f"selected t={row['selected'].commit_time_s:.2f}s",fill="#ffd968")
        images.append(image)
    images[0].save(path,save_all=True,append_images=images[1:],duration=165,
                   loop=0,optimize=True)
    for image in images:image.close()


def save_flight_gif(mission,path,closeup=False):
    rows=mission.states
    if closeup:
        start=next(i for i,row in enumerate(rows) if row["time_s"]>=mission.commit_time_s+2.)
        indices=np.unique(np.linspace(start,len(rows)-1,min(30,len(rows)-start),dtype=int))
    else:
        indices=np.unique(np.linspace(0,len(rows)-1,min(44,len(rows)),dtype=int))
    frames=[];fig=plt.figure(figsize=(8.8,6.4),dpi=85)
    positions=np.array([r["position_m"] for r in rows])
    for index in indices:
        fig.clear();ax=fig.add_subplot(projection="3d")
        row=rows[index]
        _scene(ax,mission,row["time_s"])
        if closeup:
            ax.set_xlim(19.8,22.25);ax.set_ylim(-1.15,1.15);ax.set_zlim(.95,2.25)
            ax.set_box_aspect((2.45,2.3,1.3))
        ax.plot(*positions[:index+1].T,color="#128aaf",linewidth=2,label="actual flight trail")
        _quad(ax,row)
        ax.set_title(f"M10 {'crossing closeup' if closeup else 'actual 6-DOF flight'} | t={row['time_s']:.1f}s | {row['phase']}")
        ax.legend(loc="upper left",fontsize=7)
        fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    plt.close(fig)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=170,loop=0,optimize=True)
    for frame in frames:frame.close()
    return [float(rows[i]["time_s"]) for i in indices]


def save_overview(mission,path):
    rows=mission.states;pos=np.array([x["position_m"] for x in rows])
    fig=plt.figure(figsize=(10,6),dpi=125);ax=fig.add_subplot(projection="3d")
    _scene(ax,mission,mission.commit_time_s or 0.)
    ax.plot(*pos.T,color="#087ca3",linewidth=2,label="actual plant trajectory")
    ax.scatter(*mission.hold_position_m,color="#5fd08a",s=60,label="hold point")
    ax.scatter(*pos[0],color="#394a9b",s=40,label="start")
    _quad(ax,rows[-1]);ax.legend();ax.set_title("M10 approach, wait and single timed crossing")
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_tracking(mission,path):
    obs=mission.observations;truth=mission.truth_scoring
    t=np.array([x["time_s"] for x in obs]);fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
    ax[0].plot(t,[x["true_angle_rad"] for x in truth],label="independent truth",color="#213d5a")
    ax[0].scatter(t,[x["detected_angle_rad"] if x["arm_valid"] else np.nan for x in obs],
                  s=8,label="pixel measurement",color="#df9147")
    ax[0].plot(t,[x["estimated_angle_rad"] if x["tracker_valid"] else np.nan for x in obs],
               label="temporal estimate",color="#3a9b73")
    ax[0].set_ylabel("angle (rad)");ax[0].legend();ax[0].grid(alpha=.3)
    ax[1].plot(t,[x["true_rate_rad_s"] for x in truth],label="truth")
    ax[1].plot(t,[x["estimated_rate_rad_s"] if x["tracker_valid"] else np.nan for x in obs],
               label="estimated")
    ax[1].set_ylabel("angular rate (rad/s)");ax[1].set_xlabel("simulation time (s)")
    ax[1].legend();ax[1].grid(alpha=.3);fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_prediction(mission,path):
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for horizon in (.5,1.,2.,3.):
        rows=[x for x in mission.predictions if x["horizon_s"]==horizon]
        axes[0].plot([x["observation_time_s"] for x in rows],
                     [x["predicted_angle_rad"] for x in rows],label=f"{horizon:g} s")
        axes[1].plot([x["observation_time_s"] for x in rows],
                     [math.degrees(x["error_rad"]) for x in rows],label=f"{horizon:g} s")
    axes[0].set_ylabel("predicted unwrapped angle (rad)");axes[1].set_ylabel("error (deg)")
    axes[1].set_xlabel("observation time (s)")
    for ax in axes:ax.grid(alpha=.3);ax.legend(ncol=4)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_timing(mission,path):
    fig,ax=plt.subplots(figsize=(11,4),dpi=125)
    t=np.array([x["time_s"] for x in mission.states])
    arm=np.array([x["arm_clearance_m"] for x in mission.states])
    ax.plot(t,arm,label="actual arm clearance",color="#146b9a")
    ax.axhline(0,color="#a23c4e",linewidth=1)
    # A future safe launch means the current interval was rejected. These
    # markers show scheduler decisions, not truth-derived collision labels.
    decisions=mission.choices
    delayed=[x for x in decisions if x["valid"] and x["commit_time_s"]>x["time_s"]+.05]
    immediate=[x for x in decisions if x["valid"] and x["commit_time_s"]<=x["time_s"]+.05]
    rejected=[x for x in decisions if not x["valid"]]
    top=float(max(arm))*.94
    if delayed:ax.scatter([x["time_s"] for x in delayed],[top]*len(delayed),
                          marker="x",color="#c45351",label="predicted unsafe now")
    if immediate:ax.scatter([x["time_s"] for x in immediate],[top]*len(immediate),
                            marker="o",color="#4a9d60",label="predicted safe now")
    if rejected:ax.scatter([x["time_s"] for x in rejected],[top]*len(rejected),
                           marker="s",color="#9c4d57",label="no safe interval in horizon")
    if mission.initial_wait_s is not None:ax.axvline(mission.initial_wait_s,color="#689c4e",label="hold begins")
    if mission.commit_time_s is not None:ax.axvline(mission.commit_time_s,color="#dd8c31",label="selected commitment")
    for event in mission.crossing_events:
        ax.axvline(event["plane_crossing_time_s"],color="#783da6",label="actual plane crossing")
    if mission.envelope_entry_time_s is not None:
        ax.axvline(mission.envelope_entry_time_s,color="#b2a739",linestyle="--",
                   label="envelope enters frame")
    if mission.envelope_exit_time_s is not None:
        ax.axvline(mission.envelope_exit_time_s,color="#408a7c",linestyle="--",
                   label="envelope exits frame")
    ax.set_xlabel("simulation time (s)");ax.set_ylabel("clearance beyond 0.25 m (m)")
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_control(mission,path):
    rows=mission.states;t=[x["time_s"] for x in rows]
    specs=[("arm_clearance_m","arm clearance (m)"),("frame_clearance_m","frame clearance (m)"),
           ("tracking_error_m","tracking error (m)"),("speed","speed (m/s)"),
           ("acceleration","acceleration (m/s²)"),("thrust_n","thrust (N)"),
           ("tilt_deg","tilt (deg)")]
    fig,axes=plt.subplots(len(specs),1,figsize=(11,12),sharex=True,dpi=110)
    for ax,(key,label) in zip(axes,specs):
        values=[np.linalg.norm(x["velocity_m_s"]) if key=="speed" else
                np.linalg.norm(x["acceleration_m_s2"]) if key=="acceleration" else x[key]
                for x in rows]
        ax.plot(t,values,color="#247294",linewidth=.9);ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[-1].set_xlabel("simulation time (s)")
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_abort(mission,path):
    fig,ax=plt.subplots(figsize=(9,3.5),dpi=120)
    rows=mission.states
    ax.plot([x["time_s"] for x in rows],[x["arm_clearance_m"] for x in rows],
            label="actual arm clearance while holding")
    ax.axhline(0,color="#c44343");ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("clearance (m)");ax.set_title("Unsafe static arm phase: no crossing selected")
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(path);plt.close(fig)
