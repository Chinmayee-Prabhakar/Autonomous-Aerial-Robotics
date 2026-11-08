"""M4 yaw sensitivity, gravity observability, and covariance audit regressions."""

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from aerial_autonomy.config import load_imu_experiment
from aerial_autonomy.estimation.quaternion_ukf import (
    relative_rotation_vector, right_jacobian_so3, rotation_vector_to_quaternion,
)
from aerial_autonomy.evaluation.madgwick import (
    evaluate_run, load_aligned_truth, load_measurements,
)
from aerial_autonomy.evaluation.imu import export_scenario, simulate_imu_scenario
from aerial_autonomy.evaluation.ukf import load_config, run_ukf
from aerial_autonomy.evaluation.ukf_validation import (
    geometry_audit, gravity_direction, gravity_right_jacobian,
    lateral_trace, level_heading_check, run_variants, run_yaw_prior_sweep,
    static_consistency, summarize_lateral, yaw_prior_coupling,
)
from aerial_autonomy.simulation.rotation import euler_to_quaternion, quaternion_product


ROOT = Path(__file__).resolve().parents[1]


def generated_lateral(directory: Path):
    """Recreate the M2 flight and its established CSV boundary in a temporary folder."""
    experiment = load_imu_experiment(ROOT / "configs/imu.toml")
    trace = simulate_imu_scenario(experiment, "lateral_waypoint")
    export_scenario(trace, experiment, directory)
    config, _ = load_config(ROOT / "configs/ukf.toml")
    readings = load_measurements(directory / "measurements.csv")
    times = np.array([reading.timestamp_s for reading in readings])
    truth_path = directory / "truth.csv"
    truth_q = load_aligned_truth(truth_path, times)
    table = np.atleast_1d(np.genfromtxt(truth_path, delimiter=",", names=True))
    indices = np.searchsorted(table["timestamp_s"], times)
    force = np.column_stack([table[f"specific_force_body_{axis}_m_s2"][indices]
                             for axis in "xyz"])
    return config, readings, truth_q, force


class GeometryAndCovarianceReview(unittest.TestCase):
    def test_gravity_jacobian_has_two_observable_directions(self):
        q = euler_to_quaternion(.3, -.2, .4)
        H = gravity_right_jacobian(q)
        singular = np.linalg.svd(H, compute_uv=False)
        np.testing.assert_allclose(singular, [1, 1, 0], atol=1e-14)
        np.testing.assert_allclose(H @ gravity_direction(q), np.zeros(3), atol=1e-14)
        audit = geometry_audit(load_config(ROOT / "configs/ukf.toml")[0])
        self.assertLess(audit["gravity_jacobian_max_finite_difference_error"], 1e-9)
        self.assertLess(audit["sigma_covariance_reconstruction_error_rad2"], 1e-12)
        self.assertLess(audit["global_yaw_gravity_prediction_difference"], 1e-12)

    def test_covariance_reset_matches_perturbed_quaternion_distribution(self):
        correction = np.array([.3, -.2, .4])
        local_P = np.diag([2e-6, 3e-6, 4e-6])
        root = np.linalg.cholesky(local_P)
        offsets = np.vstack((root.T, -root.T))
        sigma = np.array([relative_rotation_vector(
            rotation_vector_to_quaternion(correction),
            rotation_vector_to_quaternion(correction + offset))
            for offset in offsets])
        empirical = sigma.T @ sigma / len(sigma) * 3
        jacobian = right_jacobian_so3(correction)
        predicted = jacobian @ local_P @ jacobian.T
        np.testing.assert_allclose(empirical, predicted, atol=2e-9)

    def test_exact_level_gravity_keeps_unobservable_heading_prior(self):
        config, _ = load_config(ROOT / "configs/ukf.toml")
        rows = level_heading_check(config, [5., 30., 60.], 9.81)
        for row in rows:
            self.assertAlmostEqual(row["final_heading_projection_std_deg"],
                                   row["initial_yaw_std_deg"], places=9)
            self.assertAlmostEqual(row["final_local_heading_variance_rad2"],
                                   row["initial_local_heading_variance_rad2"], places=11)
            self.assertLess(row["final_tilt_variance_x_rad2"],
                            row["initial_local_heading_variance_rad2"])

    def test_seeded_static_observable_plane_consistency_replays(self):
        config, _ = load_config(ROOT / "configs/ukf.toml")
        first = static_consistency(config, [11, 19], .2, 50., 9.81, [5., 30.])
        second = static_consistency(config, [11, 19], .2, 50., 9.81, [5., 30.])
        self.assertEqual(first["aggregate"], second["aggregate"])
        self.assertEqual(first["individual_trials"], second["individual_trials"])
        self.assertTrue(all(np.isfinite(row["final_observable_direction_nees_2d"])
                            for row in first["individual_trials"]))


class RecordedFailureReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory(prefix="m4-lateral-test-")
        cls.addClassCleanup(temporary.cleanup)
        cls.config, cls.readings, cls.truth_q, cls.ideal_force = generated_lateral(
            Path(temporary.name))

    def test_yaw_prior_sweep_reproduces_failure_without_changing_sensors(self):
        rows = run_yaw_prior_sweep(self.readings, self.truth_q, self.config,
                                   [5., 30., 60.])
        self.assertLess(rows[0]["geodesic_rmse_deg"], 2.)
        self.assertGreater(rows[1]["geodesic_rmse_deg"], 10.)
        self.assertGreater(rows[2]["geodesic_rmse_deg"], rows[1]["geodesic_rmse_deg"])
        self.assertAlmostEqual(rows[1]["final_relative_yaw_error_deg"],
                               19.5176558297, delta=.003)

    def test_larger_yaw_prior_increases_heading_gain_during_maneuver(self):
        # Only the samples through the peak mismatch are needed for this local check.
        coupling = yaw_prior_coupling(self.readings[:151], self.truth_q[:151],
                                      self.ideal_force[:151], self.config, 9.81,
                                      2.5, [5., 30.])
        self.assertGreater(coupling[1]["gain_heading_norm_at_2_88_s"],
                           10 * coupling[0]["gain_heading_norm_at_2_88_s"])

    def test_predict_only_and_offline_truth_oracle_isolate_accel_effect(self):
        metrics, traces = run_variants(self.readings, self.truth_q, self.config, 9.81)
        self.assertGreater(metrics["operational"]["geodesic_rmse_deg"], 10.)
        self.assertLess(metrics["gyro_prediction_only"]["geodesic_rmse_deg"], 1.1)
        self.assertLess(metrics["offline_truth_gravity_oracle"]["geodesic_rmse_deg"], 1.)
        self.assertEqual(metrics["gyro_prediction_only"]["accelerometer_updates_used"], 0)
        self.assertEqual(metrics["operational"]["accelerometer_updates_used"], 500)
        self.assertEqual(len(traces["operational"][0].filter_run.time_s), 501)

    def test_step_audit_and_yaw_change_follow_waypoint(self):
        rows, audit = lateral_trace(self.readings, self.truth_q, self.ideal_force,
                                    self.config, 9.81)
        summary = summarize_lateral(rows, 2.5)
        self.assertLess(audit["max_step_audit_quaternion_error_rad"], 1e-12)
        self.assertLess(audit["max_step_audit_covariance_error_rad2"], 1e-12)
        self.assertGreater(summary["first_5deg_yaw_error_s"], 2.5)
        self.assertGreater(summary["sum_accelerometer_update_yaw_change_after_switch_deg"], 10.)
        self.assertLess(abs(summary["sum_gyro_prediction_yaw_change_after_switch_deg"]), 2.)
        self.assertFalse(summary["at_2_88_s"]["norm_gate_rejects_measured"])
        self.assertGreater(summary["at_2_88_s"]["ideal_direction_mismatch_deg"], 8.)

    def test_recorded_lateral_replay_is_deterministic_and_truth_is_score_only(self):
        self.assertEqual(tuple(inspect.signature(run_ukf).parameters), ("readings", "config"))
        self.assertEqual(len(self.readings), 501)
        np.testing.assert_allclose([sample.timestamp_s for sample in self.readings],
                                   np.arange(501) / 50, rtol=0, atol=1e-12)
        self.assertEqual(self.truth_q.shape, (501, 4))
        self.assertEqual(self.ideal_force.shape, (501, 3))
        first = run_ukf(self.readings, self.config)
        second = run_ukf(self.readings, self.config)
        np.testing.assert_array_equal(first.filter_run.quaternion_bw,
                                      second.filter_run.quaternion_bw)
        np.testing.assert_array_equal(first.covariance_rad2, second.covariance_rad2)
        scored, _ = evaluate_run(first.filter_run, self.truth_q)
        self.assertAlmostEqual(scored["geodesic_rmse_deg"], 15.6617828650, delta=.003)
        altered_truth = np.array([quaternion_product(
            euler_to_quaternion(0, 0, .2), q) for q in self.truth_q])
        altered_score, _ = evaluate_run(first.filter_run, altered_truth)
        self.assertGreater(abs(altered_score["geodesic_rmse_deg"]
                               - scored["geodesic_rmse_deg"]), 1.)
        np.testing.assert_array_equal(first.filter_run.quaternion_bw,
                                      run_ukf(self.readings, self.config).filter_run.quaternion_bw)


if __name__ == "__main__":
    unittest.main()
