"""M5 causal scheduling, feedback isolation, failure, and export regressions."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aerial_autonomy.control.cascaded import CascadedHoverController
from aerial_autonomy.evaluation.closed_loop_estimation import (
    MODES, Scenario, SensorCondition, Waypoint, feedback_state, load_config,
    run_trial, save_trial, summarize_sensitivity,
)
from aerial_autonomy.evaluation.imu import simulate_imu_scenario
from aerial_autonomy.evaluation.madgwick import geodesic_error_deg
from aerial_autonomy.simulation.dynamics import VehicleState
from aerial_autonomy.simulation.rotation import euler_to_quaternion


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/closed_loop_estimation.toml")
NOMINAL = next(item for item in CONFIG.conditions if item.name == "nominal")
IDEAL = next(item for item in CONFIG.conditions if item.name == "ideal")


def short_flight(duration_s: float = .08, target=(0., 0., 1.)) -> Scenario:
    return Scenario("test_flight", duration_s,
                    (Waypoint(0., np.asarray(target), 0.),))


class FeedbackIsolationTests(unittest.TestCase):
    def test_feedback_copies_only_selected_attitude(self):
        truth = VehicleState(np.array([.4, -.2, 1.]), np.array([.3, .1, -.1]),
                             euler_to_quaternion(.1, .2, .3), np.array([.2, -.3, .4]))
        selected = euler_to_quaternion(-.2, .05, 0.)
        feedback = feedback_state(truth, selected)
        np.testing.assert_array_equal(feedback.position_m, truth.position_m)
        np.testing.assert_array_equal(feedback.velocity_m_s, truth.velocity_m_s)
        np.testing.assert_array_equal(feedback.body_rate_rad_s, truth.body_rate_rad_s)
        np.testing.assert_array_equal(feedback.quaternion_bw, selected)
        self.assertFalse(np.array_equal(feedback.quaternion_bw, truth.quaternion_bw))
        feedback.position_m[0] = 10
        self.assertEqual(truth.position_m[0], .4)

    def test_known_positive_pitch_feedback_commands_negative_pitch_torque(self):
        controller = CascadedHoverController(CONFIG.imu_experiment.base.vehicle,
                                             CONFIG.imu_experiment.base.controller)
        state = VehicleState.at_rest()
        desired = np.zeros(3)
        level = controller.command(feedback_state(state, state.quaternion_bw), desired, 0.)
        tilted = controller.command(feedback_state(
            state, euler_to_quaternion(0., .1, 0.)), desired, 0.)
        self.assertAlmostEqual(level.wrench.torque_nm[1], 0.)
        self.assertLess(tilted.wrench.torque_nm[1], 0.)
        self.assertLess(tilted.wrench.torque_nm[1] - level.wrench.torque_nm[1], -.01)

    def test_controller_receives_truth_translation_and_rates_in_ukf_arm(self):
        observed = []
        original = CascadedHoverController

        class Recorder(original):
            def command(self, state, target_position_m, target_yaw_rad):
                observed.append(state)
                return super().command(state, target_position_m, target_yaw_rad)

        with patch("aerial_autonomy.evaluation.closed_loop_estimation.CascadedHoverController", Recorder):
            trial = run_trial(CONFIG, short_flight(), "ukf", NOMINAL, 19)
        self.assertEqual(len(observed), trial.metrics["control_updates"])
        for index, state in enumerate(observed):
            np.testing.assert_array_equal(state.position_m, trial.evaluation["position_m"][index])
            np.testing.assert_array_equal(state.velocity_m_s, trial.evaluation["velocity_m_s"][index])
            np.testing.assert_array_equal(state.body_rate_rad_s, trial.evaluation["body_rate_rad_s"][index])
            np.testing.assert_array_equal(state.quaternion_bw, trial.estimator["quaternion_bw"][index])

    def test_estimator_receives_only_current_imu_measurements(self):
        readings = []

        class SpyEstimator:
            def update(self, reading):
                readings.append(reading)
                self_test.assertEqual(set(vars(reading)),
                                      {"timestamp_s", "accel_m_s2", "gyro_rad_s"})
                return SimpleNamespace(quaternion_bw=np.array([1., 0., 0., 0.]),
                                       accel_status="spy")

        self_test = self
        trial = run_trial(CONFIG, short_flight(.06), "madgwick", IDEAL, 11,
                          estimator_factory=lambda _: SpyEstimator())
        self.assertEqual([item.timestamp_s for item in readings], [0., .02, .04, .06])
        self.assertEqual(trial.metrics["imu_updates"], 4)
        self.assertEqual(trial.metrics["control_updates"], 7)
        self.assertEqual(trial.estimator["update_status"][1], "held")

    def test_synthetic_wrong_attitude_changes_future_vehicle_motion(self):
        class WrongPitch:
            def update(self, reading):
                return SimpleNamespace(quaternion_bw=euler_to_quaternion(0., .15, 0.),
                                       accel_status="synthetic_diagnostic")

        scenario = short_flight(.5)
        truth = run_trial(CONFIG, scenario, "truth", IDEAL, 11)
        perturbed = run_trial(CONFIG, scenario, "madgwick", IDEAL, 11,
                              estimator_factory=lambda _: WrongPitch())
        self.assertLess(perturbed.evaluation["applied_torque_nm"][0, 1], 0.)
        self.assertLess(perturbed.evaluation["position_m"][-1, 0], -.001)
        self.assertGreater(np.max(np.abs(perturbed.evaluation["position_m"]
                                         - truth.evaluation["position_m"])), .001)


class TimingAndReplayTests(unittest.TestCase):
    def test_truth_feedback_reproduces_m2_plant_trajectory(self):
        scenario = next(item for item in CONFIG.scenarios if item.name == "lateral_waypoint")
        trial = run_trial(CONFIG, scenario, "truth", NOMINAL, 19)
        legacy = simulate_imu_scenario(CONFIG.imu_experiment, "lateral_waypoint")
        np.testing.assert_array_equal(trial.evaluation["position_m"], legacy.position_m)
        np.testing.assert_array_equal(trial.evaluation["true_quaternion_bw"], legacy.quaternion_bw)
        self.assertEqual(trial.metrics["imu_updates"], 501)
        self.assertEqual(trial.metrics["control_updates"], 1001)

    def test_measurement_uses_previous_interval_wrench_and_correct_grid(self):
        trial = run_trial(CONFIG, short_flight(.06), "truth", IDEAL, 11)
        np.testing.assert_allclose(trial.sensor["timestamp_s"], [0., .02, .04, .06], atol=1e-14)
        self.assertEqual(trial.evaluation["source_thrust_n"][0], 0.)
        self.assertEqual(trial.sensor["accel_m_s2"][0, 2], 0.)
        np.testing.assert_allclose(trial.evaluation["source_thrust_n"][1],
                                   trial.evaluation["applied_thrust_n"][1], atol=1e-12)
        self.assertTrue(np.all(np.diff(trial.sensor["timestamp_s"]) > 0))
        self.assertEqual(trial.estimator["measurement_sampled"].tolist(),
                         [True, False, True, False, True, False, True])

    def test_same_seed_replays_physics_and_estimates_exactly(self):
        scenario = short_flight(.2, (.4, 0., 1.))
        first = run_trial(CONFIG, scenario, "ukf", NOMINAL, 19)
        second = run_trial(CONFIG, scenario, "ukf", NOMINAL, 19)
        for group in ("sensor", "estimator", "evaluation"):
            left, right = getattr(first, group), getattr(second, group)
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])

    def test_paired_noise_innovations_match_across_diverging_arms(self):
        scenario = short_flight(.4, (.5, .2, 1.))
        trials = [run_trial(CONFIG, scenario, mode, NOMINAL, 19) for mode in MODES]
        self.assertGreater(np.max(np.abs(trials[0].evaluation["position_m"]
                                         - trials[2].evaluation["position_m"])), 0.)
        for other in trials[1:]:
            np.testing.assert_allclose(other.evaluation["accel_noise_innovation_m_s2"],
                                       trials[0].evaluation["accel_noise_innovation_m_s2"], atol=1e-12)
            np.testing.assert_allclose(other.evaluation["gyro_noise_innovation_rad_s"],
                                       trials[0].evaluation["gyro_noise_innovation_rad_s"], atol=1e-12)
        alternate = run_trial(CONFIG, scenario, "truth", NOMINAL, 29)
        self.assertFalse(np.array_equal(alternate.evaluation["accel_noise_innovation_m_s2"],
                                        trials[0].evaluation["accel_noise_innovation_m_s2"]))

    def test_ideal_hover_estimator_feedback_remains_consistent(self):
        scenario = short_flight(.4)
        truth = run_trial(CONFIG, scenario, "truth", IDEAL, 11)
        for mode in ("madgwick", "ukf"):
            other = run_trial(CONFIG, scenario, mode, IDEAL, 11)
            self.assertLess(other.metrics["attitude_geodesic_rmse_deg"], 1e-5)
            np.testing.assert_allclose(other.evaluation["position_m"],
                                       truth.evaluation["position_m"], atol=1e-8)


class MetricsAndFailureTests(unittest.TestCase):
    def test_invalid_initial_estimate_is_exportable_failure(self):
        class ImmediatelyInvalid:
            def update(self, reading):
                return SimpleNamespace(quaternion_bw=np.array([float("nan")] * 4),
                                       accel_status="invalid")

        trial = run_trial(CONFIG, short_flight(), "madgwick", IDEAL, 11,
                          estimator_factory=lambda _: ImmediatelyInvalid())
        self.assertEqual(trial.metrics["failure_reason"], "invalid_estimator_quaternion")
        self.assertEqual(trial.metrics["failure_time_s"], 0.)
        self.assertEqual(trial.metrics["control_updates"], 0)
        self.assertIsNone(trial.metrics["position_rmse_m"])
        with TemporaryDirectory() as temporary:
            save_trial(trial, Path(temporary))
            self.assertTrue((Path(temporary) / "metrics.json").is_file())

    def test_bad_estimator_output_terminates_and_keeps_prior_history(self):
        class FailingEstimator:
            def __init__(self):
                self.calls = 0

            def update(self, reading):
                self.calls += 1
                q = ([1., 0., 0., 0.] if self.calls == 1 else [float("nan")] * 4)
                return SimpleNamespace(quaternion_bw=np.asarray(q), accel_status="spy")

        trial = run_trial(CONFIG, short_flight(.1), "madgwick", IDEAL, 11,
                          estimator_factory=lambda _: FailingEstimator())
        self.assertFalse(trial.metrics["success"])
        self.assertEqual(trial.metrics["failure_reason"], "invalid_estimator_quaternion")
        self.assertAlmostEqual(trial.metrics["failure_time_s"], .02)
        self.assertEqual(trial.metrics["control_updates"], 2)
        self.assertFalse(trial.metrics["completed_horizon"])

    def test_position_limit_terminates_and_reports_sample_window(self):
        limits = replace(CONFIG.limits, max_position_error_m=.1)
        trial = run_trial(replace(CONFIG, limits=limits), short_flight(), "truth", IDEAL, 11)
        self.assertEqual(trial.metrics["failure_reason"], "position_tracking_limit")
        self.assertEqual(trial.metrics["failure_time_s"], 0.)
        self.assertEqual(trial.metrics["control_updates"], 1)
        self.assertFalse(trial.metrics["completed_horizon"])

    def test_full_horizon_nonsettlement_is_reported_separately(self):
        trial = run_trial(CONFIG, short_flight(.2), "truth", IDEAL, 11)
        self.assertTrue(trial.metrics["completed_horizon"])
        self.assertFalse(trial.metrics["success"])
        self.assertEqual(trial.metrics["failure_reason"], "target_not_settled")
        self.assertAlmostEqual(trial.metrics["failure_time_s"], .2)

    def test_aggregate_retains_failed_trials_and_uses_full_horizon_only(self):
        full = run_trial(CONFIG, short_flight(.2), "truth", IDEAL, 11)
        early = run_trial(replace(CONFIG, limits=replace(CONFIG.limits,
                                max_position_error_m=.1)), short_flight(.2),
                          "truth", IDEAL, 11)
        summary = summarize_sensitivity([full.metrics, early.metrics])[0]
        self.assertEqual(summary["trials"], 2)
        self.assertEqual(summary["failures"], 2)
        self.assertEqual(summary["early_terminated"], 1)
        self.assertEqual(summary["full_horizon_trials"], 1)
        self.assertAlmostEqual(summary["position_rmse_m_full_horizon_mean"],
                               full.metrics["position_rmse_m"])

    def test_errors_have_distinct_estimation_and_true_tracking_definitions(self):
        trial = run_trial(CONFIG, short_flight(.1, (.5, 0., 1.)), "ukf", NOMINAL, 19)
        scored = geodesic_error_deg(trial.estimator["quaternion_bw"],
                                    trial.evaluation["true_quaternion_bw"])
        np.testing.assert_allclose(scored, trial.evaluation["attitude_estimation_geodesic_error_deg"], atol=1e-12)
        self.assertAlmostEqual(trial.metrics["position_rmse_m"],
                               np.sqrt(np.mean(trial.evaluation["position_error_m"] ** 2)))
        self.assertIn("true_attitude_tracking_error_deg", trial.evaluation)
        self.assertNotIn("true_attitude_tracking_error_deg", trial.estimator)

    def test_export_keeps_sensor_only_fields_separate(self):
        trial = run_trial(CONFIG, short_flight(.08), "ukf", NOMINAL, 19)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_trial(trial, directory)
            with np.load(directory / "sensor.npz", allow_pickle=False) as sensor:
                self.assertEqual(set(sensor.files), {"timestamp_s", "accel_m_s2", "gyro_rad_s"})
            with np.load(directory / "estimator.npz", allow_pickle=False) as estimate:
                self.assertIn("quaternion_bw", estimate.files)
                self.assertNotIn("geodesic_error_deg", estimate.files)
                self.assertNotIn("true_quaternion_bw", estimate.files)
            with np.load(directory / "evaluation.npz", allow_pickle=False) as evaluation:
                self.assertIn("true_quaternion_bw", evaluation.files)
                self.assertIn("desired_position_m", evaluation.files)
                self.assertIn("attitude_estimation_geodesic_error_deg", evaluation.files)
            self.assertTrue((directory / "metrics.json").is_file())
            self.assertTrue((directory / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
