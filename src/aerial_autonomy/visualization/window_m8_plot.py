"""Synthetic M8 camera views and truth/measurement/estimate overlays."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerial_autonomy.perception.window_pose import reproject_estimate


def save_camera_view(path: str | Path, camera, observation, ideal_pixels,
                     estimate, model, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax = plt.subplots(figsize=(9,7),dpi=130)
    fig.patch.set_facecolor("#101725")
    ax.set_facecolor("#182538")
    ax.set_xlim(0,camera.width_px)
    ax.set_ylim(camera.height_px,0)
    ax.set_aspect("equal")
    ax.grid(color="#36475a",alpha=.45)
    ax.set_xlabel("image u (px)")
    ax.set_ylabel("image v (px)")
    ax.set_title(title)
    if np.all(np.isfinite(ideal_pixels)):
        loop = np.vstack((ideal_pixels,ideal_pixels[0]))
        ax.plot(loop[:,0],loop[:,1],color="#77c5ee",linewidth=12,alpha=.18,
                solid_capstyle="round",label="true frame outline")
        ax.plot(loop[:,0],loop[:,1],color="#77c5ee",linewidth=2,alpha=.8)
        ax.scatter(ideal_pixels[:,0],ideal_pixels[:,1],marker="o",s=70,
                   facecolors="none",edgecolors="#75dcff",linewidths=2,
                   label="truth projection")
    valid = observation.visible
    if np.any(valid):
        ax.scatter(observation.pixels_uv[valid,0],observation.pixels_uv[valid,1],
                   marker="x",s=110,color="#ffd166",linewidths=2.2,
                   label="measured corners")
        for i,(u,v) in enumerate(observation.pixels_uv):
            if valid[i]:
                ax.text(u+7,v-7,("BL","BR","TR","TL")[i],color="white",fontsize=9)
    estimated = reproject_estimate(estimate,model,camera)
    if estimated is not None:
        loop = np.vstack((estimated,estimated[0]))
        ax.plot(loop[:,0],loop[:,1],color="#ec6a96",linestyle="--",linewidth=1.5)
        ax.scatter(estimated[:,0],estimated[:,1],marker="+",s=100,
                   color="#f785a9",linewidths=2,label="estimated reprojection")
    status = f"{observation.visible_count}/4 corners | "
    status += (f"pose RMSE {estimate.reprojection_rmse_px:.2f} px | {estimate.facing_side}"
               if estimate.valid else f"failure: {estimate.reason}")
    ax.text(.02,.97,status,transform=ax.transAxes,va="top",color="white",
            bbox={"facecolor":"#0d1523","alpha":.85,"edgecolor":"none"})
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right",facecolor="#263449",labelcolor="white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.tight_layout()
    fig.savefig(path,facecolor=fig.get_facecolor())
    plt.close(fig)


def save_metric_plots(output_dir: str | Path, rows: list[dict]) -> None:
    output_dir = Path(output_dir)
    noise = sorted({r["noise_std_px"] for r in rows if r["group"]=="noise"})
    fig,axs = plt.subplots(1,2,figsize=(11,4),dpi=140)
    for axis,key,label in ((axs[0],"translation_error_m","Translation error (m)"),
                           (axs[1],"orientation_error_deg","Orientation error (deg)")):
        data = [[r[key] for r in rows if r["group"]=="noise" and
                 r["noise_std_px"]==level and r["success"]] for level in noise]
        axis.boxplot(data,positions=noise,widths=.23 if len(noise)>1 else .5)
        axis.set_xticks(noise)
        axis.set_xlabel("Pixel noise SD (px)")
        axis.set_ylabel(label)
        axis.grid(alpha=.3)
    fig.suptitle("M8 image-only pose accuracy across all preset seeds")
    fig.tight_layout()
    fig.savefig(output_dir/"pose_error_vs_noise.png")
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(8,4),dpi=140)
    dropout = sorted({r["corner_dropout_probability"] for r in rows if r["group"]=="dropout"})
    rates = [np.mean([r["success"] for r in rows if r["group"]=="dropout" and
                     r["corner_dropout_probability"]==level]) for level in dropout]
    ax.bar(np.arange(len(dropout)),rates,color="#467db5")
    ax.set_xticks(np.arange(len(dropout)),[str(x) for x in dropout])
    ax.set_ylim(0,1.05)
    ax.set_xlabel("Independent corner dropout probability")
    ax.set_ylabel("Estimation success fraction")
    ax.set_title("M8 observed success; every dropout trial included")
    ax.grid(axis="y",alpha=.3)
    fig.tight_layout()
    fig.savefig(output_dir/"success_vs_dropout.png")
    plt.close(fig)
