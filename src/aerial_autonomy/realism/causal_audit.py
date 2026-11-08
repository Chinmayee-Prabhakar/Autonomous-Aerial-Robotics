"""Independently recheck the limited M12 causal trial's saved physical path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aerial_autonomy.evaluation.mission_m11 import load_m11_config, physical_crossings
from aerial_autonomy.perception.window_m10_arm import swept_arm_clearance_m


def audit(path='outputs/realism_m12/causal_alpha'):
    root = Path(path)
    data = np.load(root/'physical_trace.npz')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    _, _, _, _, world, arm, _, _ = load_m11_config('configs/mission_m11.toml')
    t = np.asarray(data['time_s'])
    p = np.asarray(data['position_m'])
    static = np.asarray(data['static_clearance_m'])
    dynamic = np.asarray(data['dynamic_clearance_m'])
    thrust = np.asarray(data['thrust_n'])
    torque = np.asarray(data['torque_nm'])
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(p)):
        raise AssertionError('invalid causal path or time order')
    replay_static = np.array([world.segment_clearance_m(p[i-1], p[i])
                              for i in range(1, len(t))])
    replay_dynamic = np.array([swept_arm_clearance_m(
        p[i-1], p[i], float(t[i-1]), float(t[i]), arm, world.envelope_m)
        for i in range(1, len(t))])
    static_residual = float(np.max(abs(replay_static-static[1:])))
    dynamic_residual = float(np.max(abs(replay_dynamic-dynamic[1:])))
    if static_residual > 1e-8 or dynamic_residual > 1e-8:
        raise AssertionError('saved causal clearances disagree with independent replay')
    records = [{'time_s': float(t[i]), 'position_m': p[i],
                'static_clearance_m': float(static[i]),
                'dynamic_clearance_m': float(dynamic[i]),
                'phase': 'REPLAY'} for i in range(len(t))]
    crossings = physical_crossings(records, world)
    if (len(crossings) != 1 or crossings[0]['window_id'] != 'alpha' or
            crossings[0]['direction'] != 'outbound' or
            crossings[0]['classification'] != 'opening'):
        raise AssertionError('causal physical crossing count/order mismatch')
    if min(replay_static) <= 0 or min(replay_dynamic) <= 0:
        raise AssertionError('causal path collides')
    report = {'schema': 'm12_causal_independent_collision_audit_v1',
              'passed': True, 'physical_steps': len(t)-1,
              'begin_time_s': float(t[0]), 'end_time_s': float(t[-1]),
              'path_length_m': float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
              'minimum_static_clearance_m': float(min(replay_static)),
              'minimum_dynamic_clearance_m': float(min(replay_dynamic)),
              'maximum_static_replay_residual_m': static_residual,
              'maximum_dynamic_replay_residual_m': dynamic_residual,
              'crossings': crossings,
              'maximum_applied_thrust_n': float(max(thrust[1:])),
              'maximum_applied_body_torque_nm': np.max(abs(torque[1:]), axis=0).tolist(),
              'rgb_observation_count': summary['observation_count'],
              'valid_pixel_pose_count': summary['valid_window_observations']}
    (root/'independent_audit.json').write_text(json.dumps(report, indent=2),
                                               encoding='utf-8')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, p[:, 0], label='X')
    axes[0].plot(t, p[:, 1], label='Y')
    axes[0].plot(t, p[:, 2], label='Z')
    axes[0].set_ylabel('position (m)'); axes[0].legend(ncol=3)
    axes[1].plot(t, thrust, label='collective thrust')
    for i, name in enumerate('XYZ'):
        axes[1].plot(t, torque[:, i], label='torque '+name)
    axes[1].set_ylabel('applied command'); axes[1].legend(ncol=4)
    axes[2].plot(t, static, label='static extra clearance')
    axes[2].plot(t, dynamic, label='moving-arm extra clearance')
    axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].set_ylabel('clearance (m)')
    axes[2].set_xlabel('M12 causal simulation time (s)')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/'controls_and_clearance.png', dpi=150)
    plt.close(fig)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
