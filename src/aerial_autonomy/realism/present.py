"""Compose truthful M12 review figures and GIFs from saved M11 states and Blender RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aerial_autonomy.evaluation.mission_m11 import load_m11_config
from aerial_autonomy.realism.detector import AdaptiveColorWindowDetector
from aerial_autonomy.realism.bridge import project_path


INK = (18, 27, 34)
PAPER = (239, 244, 245)
CYAN = (51, 223, 222)
ORANGE = (245, 165, 73)
GREEN = (100, 218, 139)


def font(size):
    path = Path('C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def wrapped_lines(draw, content, text_font, width):
    lines, current = [], ''
    for word in content.split():
        candidate = (current + ' ' + word).strip()
        if current and draw.textlength(candidate, font=text_font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [])


def card(image, title, subtitle, status=None):
    measuring = ImageDraw.Draw(image)
    subtitle_font = font(12)
    lines = wrapped_lines(measuring, subtitle, subtitle_font, image.width - 30)
    footer_height = max(42, 8 + len(lines) * 17)
    canvas = Image.new('RGB', (image.width, image.height + 48 + footer_height), INK)
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 6, 47), fill=CYAN)
    title_size = 15 if image.width <= 480 else 19
    allowed = image.width - (185 if status and image.width > 480 else 30)
    while title_size > 10 and draw.textlength(title, font=font(title_size)) > allowed:
        title_size -= 1
    draw.text((18, 10), title, fill=PAPER,
              font=font(title_size))
    for index, line in enumerate(lines):
        draw.text((18, image.height + 54 + index * 17), line,
                  fill=(184, 201, 209), font=subtitle_font)
    if status and image.width > 480:
        draw.text((canvas.width - 165, 14), status, fill=GREEN, font=font(13))
    return canvas


def gif(path, frames, delay=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [x.convert('P', palette=Image.Palette.ADAPTIVE, colors=192)
              for x in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=delay, loop=0, optimize=False, disposal=2)


def _phase(phase):
    return phase.replace('_', ' ').title()


def mission_frames(plan, root):
    result = []
    for job in (j for j in plan['jobs'] if j['group'] == 'mission'):
        image = Image.open(root / 'rendered' / 'mission' /
                           (job['job_id'] + '.png')).convert('RGB')
        result.append(card(image, f"M11 FLIGHT / BLENDER REPLAY  •  {_phase(job['phase'])}",
            f"M11 state at t={job['time_s']:.2f}s  |  static {job['static_clearance_m']:.3f}m  |  "
            f"dynamic {job['dynamic_clearance_m']:.3f}m  |  timeline sampled; playback compressed",
            None))
    return result


def onboard_frames(plan, root, detector, camera):
    result = []
    for job in (j for j in plan['jobs'] if j['group'] == 'onboard'):
        image = Image.open(root / 'rendered' / 'onboard' /
                           (job['job_id'] + '.png')).convert('RGB')
        rgb = np.asarray(image)
        detection = detector.detect(rgb, camera, job['target_id'], job['expected_side'])
        if detection.valid:
            draw = ImageDraw.Draw(image)
            points = [tuple(p) for p in detection.corners_uv]
            draw.line(points + points[:1], fill=GREEN, width=2)
            status = 'PIXEL FIX'
        else:
            status = 'NO PIXEL FIX'
        enlarged = image.resize((640, 480), Image.Resampling.NEAREST)
        result.append(card(enlarged, f"M12 / ONBOARD RGB REPLAY  •  {job['target_id'].upper()}",
            f"t={job['time_s']:.2f}s  |  {_phase(job['phase'])}  |  green = M12 pixel detection  |  sampled playback",
            status))
    return result


def arm_frames(plan, root):
    result = []
    arm = plan['scene']['arm']
    for job in (j for j in plan['jobs'] if j['group'] == 'arm'):
        image = Image.open(root / 'rendered' / 'arm' /
                           (job['job_id'] + '.png')).convert('RGB')
        angle = arm['initial_angle_rad'] + arm['angular_velocity_rad_s'] * job['time_s']
        result.append(card(image, 'M12 / CENTERED ROTATING ARM  •  DELTA',
            f"M11 t={job['time_s']:.2f}s  |  arm phase={np.degrees(angle)%360:.1f}°  |  "
            f"clearance={job['dynamic_clearance_m']:.3f}m  |  sampled playback",
            None))
    return result


def _chart(draw, box, times, series, colors, limits, cursor, labels=()):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(79, 96, 106), width=1)
    start, stop = float(times[0]), float(times[-1])
    ymin, ymax = limits
    def point(t, value):
        value = min(ymax, max(ymin, float(value)))
        return (x0 + 41 + (x1 - x0 - 49) * (t - start) / (stop - start),
                y1 - 19 - (y1 - y0 - 30) * (value - ymin) / (ymax - ymin))
    for values, color in zip(series, colors):
        for i in range(1, len(times)):
            draw.line((point(times[i-1], values[i-1]), point(times[i], values[i])),
                      fill=color, width=2)
    cx = point(cursor, ymin)[0]
    draw.line(((cx, y0 + 9), (cx, y1 - 17)), fill=PAPER, width=1)
    tick_font = font(9)
    for value, y in ((ymax, y0 + 9), (ymin, y1 - 31)):
        draw.text((x0 + 3, y), f'{value:.2g}', fill=PAPER, font=tick_font)
    draw.text((x0 + 40, y1 - 16), f'{start:.0f}s', fill=PAPER, font=tick_font)
    draw.text((x1 - 28, y1 - 16), f'{stop:.0f}s', fill=PAPER, font=tick_font)
    for index, (label, color) in enumerate(zip(labels, colors)):
        draw.text((x0 + 45 + index * 94, y0 + 1), label,
                  fill=color, font=tick_font)


def engineering_frames(plan, root, trace, mission, onboard):
    sync_jobs = {j['trace_index']: j for j in plan['jobs'] if j['group'] == 'sync'}
    mission_jobs = [j for j in plan['jobs'] if j['group'] == 'mission']
    time = np.asarray(trace['time_s'])
    stride = max(1, len(time)//500)
    sample = np.arange(0, len(time), stride)
    if sample[-1] != len(time)-1:
        sample = np.append(sample, len(time)-1)
    t = time[sample]
    result = []
    for n in range(0, len(mission), 2):
        job = mission_jobs[n]
        sync = sync_jobs[job['trace_index']]
        canvas = Image.new('RGB', (960, 620), INK)
        external = mission[n].crop((0, 48, 480, 318))
        on = Image.open(root / 'rendered' / 'sync' /
                        (sync['job_id'] + '.png')).convert('RGB').resize((360, 270))
        canvas.paste(external, (0, 44))
        canvas.paste(on, (540, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 8), 'M11 RECORDED FLIGHT / BLENDER RGB REPLAY', fill=PAPER,
                  font=font(19))
        draw.text((14, 318), f"EXTERNAL · {_phase(job['phase'])}", fill=CYAN,
                  font=font(14))
        draw.text((492, 318), f"ONBOARD · same saved t={sync['time_s']:.2f}s",
                  fill=CYAN, font=font(14))
        draw.text((14, 346), 'X position: actual / reference', fill=PAPER,
                  font=font(13))
        draw.text((492, 346), 'Applied collective thrust (N)', fill=PAPER,
                  font=font(13))
        draw.text((14, 478), 'Applied body torque X/Y/Z (N m)', fill=PAPER,
                  font=font(13))
        draw.text((492, 478), 'Static / dynamic clearance (m)', fill=PAPER,
                  font=font(13))
        torque = np.asarray(trace['torque_nm'])[sample]
        _chart(draw, (14, 370, 466, 463), t,
               [trace['position_m'][sample, 0],
                trace['reference_position_m'][sample, 0]], [CYAN, ORANGE],
               (-2., 25.), job['time_s'], ('actual', 'reference'))
        _chart(draw, (492, 370, 946, 463), t, [trace['thrust_n'][sample]],
               [GREEN], (0., max(15., float(np.max(trace['thrust_n'])))),
               job['time_s'], ('thrust',))
        bound = max(.01, float(np.max(abs(torque))))
        _chart(draw, (14, 502, 466, 595), t,
               [torque[:, 0], torque[:, 1], torque[:, 2]],
               [CYAN, ORANGE, GREEN], (-bound, bound), job['time_s'],
               ('torque X', 'torque Y', 'torque Z'))
        _chart(draw, (492, 502, 946, 595), t,
               [trace['static_clearance_m'][sample],
                trace['dynamic_clearance_m'][sample]], [CYAN, ORANGE],
               (-.1, 5.), job['time_s'], ('static', 'dynamic'))
        draw.text((14, 602), f"M11 recorded state and commands  •  t={job['time_s']:.2f}s  •  "
                  'timeline sampled; playback compressed', fill=(169, 190, 200),
                  font=font(11))
        result.append(canvas)
    return result


def figures(plan, root, trace, aggregate):
    figure_dir = root / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = [r for r in aggregate['cases'] if r['domain'] == 'Blender RGB']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    splits = ['development', 'held_out']
    for algorithm, color, offset, display in [
            ('m11_operational', '#f3a65c', -.26, 'M11 exact'),
            ('m9_comparator', '#489bd9', 0, 'M9 older'),
            ('m12_adapted', '#31cdbf', .26, 'M12 adapted')]:
        detected = [sum(r['scores'][algorithm]['window']['truth_matched']
                        for r in records if r['split'] == s) for s in splits]
        usable = [sum(r['scores'][algorithm]['window']['truth_usable']
                      for r in records if r['split'] == s) for s in splits]
        false = [sum(r['scores'][algorithm]['window']['false_detection']
                     for r in records if r['split'] == s) for s in splits]
        axes[0].bar(np.arange(2)+offset, detected, .24, color=color, label=display)
        axes[1].bar(np.arange(2)+offset, false, .24, color=color, label=display)
        for i, n in enumerate(usable):
            axes[0].annotate(f'{detected[i]}/{n}',
                             (i+offset, detected[i] + .06),
                             ha='center', va='bottom', fontsize=8)
    axes[0].set_ylim(0, 7.5)
    for ax, title in zip(axes, ('Detections (denominator: corners in frame)',
                                'False detections, including unusable views')):
        ax.set_title(title)
        ax.set_xticks(range(2), splits)
        ax.set_ylabel('cases')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.savefig(figure_dir / 'perception_accuracy_and_failures.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(10, 11), constrained_layout=True)
    native = np.load(project_path(root/'scene_plan.json',
        plan['settings']['source']['trial_dir']) /
                     'camera_frames.npz')
    for ax_row, target in zip(axes, ['alpha', 'beta', 'gamma', 'delta']):
        job = next(j for j in plan['jobs'] if j['group'] == 'evaluation'
                   and j['target_id'] == target and
                   j['expected_side'] == 'positive_normal')
        ax_row[0].imshow(native['image_rgb'][job['source_camera_frame']])
        ax_row[1].imshow(Image.open(root / 'rendered' / 'evaluation' /
                                   (job['job_id'] + '.png')))
        for ax in ax_row:
            ax.set_axis_off()
        ax_row[0].set_title(f'{target}: M11 native RGB')
        ax_row[1].set_title(f'{target}: Blender RGB')
    fig.savefig(figure_dir / 'native_vs_blender_rgb.png', dpi=150)
    plt.close(fig)

    calibration = json.loads((root / 'camera_projection_checks.json').read_text(
        encoding='utf-8'))
    chosen = ['alpha_positive_normal_00', 'alpha_negative_normal_01']
    overlay = Image.new('RGB', (640, 285), INK)
    draw = ImageDraw.Draw(overlay)
    for column, name in enumerate(chosen):
        pane = Image.open(root / 'rendered' / 'evaluation' /
                          (name + '.png')).convert('RGB')
        overlay.paste(pane, (column*320, 32))
        item = next(v for v in calibration['views'] if v['job_id'] == name)
        draw.text((column*320+8, 7), item['side'].replace('_', ' '),
                  fill=PAPER, font=font(16))
        for (u, v), (bu, bv) in zip(item['python_uv'], item['blender_uv']):
            x = column*320 + u
            y = 32 + v
            draw.ellipse((x-4, y-4, x+4, y+4), outline=ORANGE, width=2)
            draw.point((column*320+bu, 32+bv), fill=CYAN)
    draw.text((8, 274), 'orange rings: Python pinhole; cyan centers: Blender',
              fill=PAPER, font=font(10))
    overlay.save(figure_dir / 'front_reverse_projection_overlay.png')

    failures = [r for r in records if r['scores']['m12_adapted']['window']['false_detection']
                or not r['scores']['m12_adapted']['window']['detection_valid']]
    failures = failures[:6]
    mosaic = Image.new('RGB', (960, len(failures)//3*275 +
                               (275 if len(failures)%3 else 0)), INK)
    labeler = ImageDraw.Draw(mosaic)
    for n, row in enumerate(failures):
        x, y = n%3*320, n//3*275
        picture = Image.open(root / 'rendered' / 'evaluation' /
                             (row['job_id'] + '.png')).convert('RGB')
        mosaic.paste(picture, (x, y+25))
        reason = row['scores']['m12_adapted']['window']['detection_reason']
        labeler.text((x+5, y+4), row['job_id'][:27], fill=PAPER, font=font(12))
        labeler.text((x+5, y+258), 'false detection' if
            row['scores']['m12_adapted']['window']['false_detection'] else
            f'miss: {reason}', fill=ORANGE, font=font(11))
    mosaic.save(figure_dir / 'failure_and_dropout_examples.png')

    delta = [r for r in records if r['target_id'] == 'delta']
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    for label, marker, color in [('m11_operational', 'x', '#f3a65c'),
                                 ('m9_comparator', '+', '#489bd9'),
                                 ('m12_adapted', 'o', '#31cdbf')]:
        for row in delta:
            arm = row['scores'][label]['arm']
            ax.scatter(row['time_s'], arm['angle_error_deg'] if arm['valid'] else -1,
                       marker=marker, color=color, label=label if row is delta[0] else None)
    ax.axhline(0, color='#666', linewidth=.8)
    ax.set_xlabel('M11 mission time (s)')
    ax.set_ylabel('arm angle error (deg); -1 = dropout')
    ax.set_title('Every delta view, including pixel-tracking failures')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figure_dir / 'rotating_arm_errors_and_dropouts.png', dpi=150)
    plt.close(fig)

    timeline = json.loads((project_path(root/'scene_plan.json',
        plan['settings']['source']['trial_dir']) /
                           'figures' / 'animation_timeline.json').read_text())
    selections = [('takeoff', 0)]
    for event in timeline['events']:
        label = (event['window_id'] + ' ' +
                 ('bypass' if event['classification'] == 'external_plane_bypass'
                  else event['direction']))
        selections.append((label, event['mission_frame']))
    selections.append(('landing', len(timeline['mission_frame_times_s'])-1))
    sheet = Image.new('RGB', (1280, 630), INK)
    labels = ImageDraw.Draw(sheet)
    for slot, (label, index) in enumerate(selections):
        x, y = (slot % 4)*320, (slot // 4)*210
        path = root / 'rendered' / 'mission' / f'mission_{index:03d}.png'
        picture = Image.open(path).convert('RGB').resize((320, 180))
        sheet.paste(picture, (x, y+25))
        time_s = timeline['mission_frame_times_s'][index]
        labels.text((x+8, y+4), f'{label} · t={time_s:.2f}s',
                    fill=PAPER, font=font(13))
    sheet.save(figure_dir / 'mission_event_contact_sheet.png')

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    sample = slice(None, None, 20)
    t = trace['time_s'][sample]
    axes[0].plot(t, trace['position_m'][sample, 0], label='actual X')
    axes[0].plot(t, trace['reference_position_m'][sample, 0], '--',
                 label='reference X')
    axes[0].set_ylabel('X position (m)'); axes[0].legend()
    axes[1].plot(t, trace['thrust_n'][sample]); axes[1].set_ylabel('thrust (N)')
    for i, name in enumerate('XYZ'):
        axes[2].plot(t, trace['torque_nm'][sample, i], label=name)
    axes[2].set_ylabel('body torque (N m)'); axes[2].legend()
    axes[3].plot(t, trace['static_clearance_m'][sample], label='static')
    axes[3].plot(t, trace['dynamic_clearance_m'][sample], label='dynamic')
    axes[3].set_ylabel('clearance (m)'); axes[3].set_xlabel('M11 simulation time (s)')
    axes[3].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(figure_dir / 'm11_recorded_controls_and_tracking.png', dpi=150)
    plt.close(fig)


def present(plan_path, output_dir):
    root = Path(output_dir)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    aggregate = json.loads((root / 'perception_aggregate_all.json').read_text(
        encoding='utf-8'))
    trial = project_path(plan_path, plan['settings']['source']['trial_dir'])
    trace = np.load(trial / 'physical_trace.npz')
    _, _, _, camera, _, _, colors, _ = load_m11_config(
        project_path(plan_path, plan['settings']['source']['mission_config']))
    detector = AdaptiveColorWindowDetector(colors)
    a = mission_frames(plan, root)
    b = onboard_frames(plan, root, detector, camera)
    c = arm_frames(plan, root)
    d = engineering_frames(plan, root, trace, a, b)
    movies = root / 'videos'
    for name, frames in [('full_indoor_mission.gif', a),
                         ('onboard_rgb_perception.gif', b),
                         ('centered_rotating_arm.gif', c),
                         ('engineering_explanation.gif', d)]:
        gif(movies / name, frames, plan['settings']['presentation']['gif_frame_delay_ms'])
        print('M12_VIDEO', movies / name, len(frames), flush=True)
    figures(plan, root, trace, aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='outputs/realism_m12/scene_plan.json')
    parser.add_argument('--output-dir', default='outputs/realism_m12')
    args = parser.parse_args()
    present(args.plan, args.output_dir)


if __name__ == '__main__':
    main()
