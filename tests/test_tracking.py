import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune.tracking import TrackioTracker


class TrackioTrackerTests(unittest.TestCase):
    def test_local_run_logs_metrics_and_finishes(self):
        trackio = Mock()
        tracker = TrackioTracker({"project": "benchmarks"}, logging.getLogger(__name__))

        with patch("auto_tune.tracking.importlib.import_module", return_value=trackio):
            tracker.start_run(name="vllm-abcd", group="balanced", config={"model": "example/model"})
            tracker.log_metrics({"throughput": 12.5}, slo_met=True)
            tracker.finish_run()

        trackio.init.assert_called_once_with(
            project="benchmarks",
            name="vllm-abcd",
            group="balanced",
            config={"model": "example/model"},
            auto_log_cpu=False,
            auto_log_gpu=False,
        )
        trackio.log.assert_called_once_with({"throughput": 12.5, "slo_met": 1, "rate_benchmark": 0})
        trackio.finish.assert_called_once_with()

    def test_remote_destinations_and_rate_are_forwarded(self):
        for destination in (
            {"space_id": "user/benchmarks"},
            {"server_url": "http://localhost:7860"},
        ):
            with self.subTest(destination=destination):
                trackio = Mock()
                tracker = TrackioTracker({"project": "benchmarks", **destination}, logging.getLogger(__name__))
                with patch("auto_tune.tracking.importlib.import_module", return_value=trackio):
                    tracker.start_run(name="vllm-abcd", group="balanced", config={})
                    tracker.log_metrics({"throughput": 10.0}, slo_met=False, target_rate=12.0)

                self.assertEqual(trackio.init.call_args.kwargs[next(iter(destination))], next(iter(destination.values())))
                trackio.log.assert_called_once_with(
                    {"throughput": 10.0, "slo_met": 0, "rate_benchmark": 1, "target_rate": 12.0}
                )

    def test_missing_configuration_disables_tracking(self):
        tracker = TrackioTracker(None, logging.getLogger(__name__))
        with patch("auto_tune.tracking.importlib.import_module") as import_module:
            tracker.start_run(name="unused", group="unused", config={})
        import_module.assert_not_called()

    def test_hf_token_is_available_for_space_run_and_restored(self):
        trackio = Mock()
        observed_tokens = []
        trackio.init.side_effect = lambda **_: observed_tokens.append(os.environ.get("HF_TOKEN"))
        tracker = TrackioTracker(
            {"project": "benchmarks", "space_id": "user/benchmarks"},
            logging.getLogger(__name__),
            hf_token="provided-token",
        )

        with (
            patch.dict(os.environ, {"HF_TOKEN": "previous-token"}),
            patch("auto_tune.tracking.importlib.import_module", return_value=trackio),
        ):
            tracker.start_run(name="vllm-abcd", group="balanced", config={})
            self.assertEqual(os.environ["HF_TOKEN"], "provided-token")
            tracker.finish_run()
            self.assertEqual(os.environ["HF_TOKEN"], "previous-token")

        self.assertEqual(observed_tokens, ["provided-token"])

    def test_hf_token_is_restored_when_space_initialization_fails(self):
        trackio = Mock()
        trackio.init.side_effect = RuntimeError("unavailable")
        tracker = TrackioTracker(
            {"project": "benchmarks", "space_id": "user/benchmarks"},
            logging.getLogger(__name__),
            hf_token="provided-token",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("auto_tune.tracking.importlib.import_module", return_value=trackio),
        ):
            tracker.start_run(name="vllm-abcd", group="balanced", config={})
            self.assertNotIn("HF_TOKEN", os.environ)


if __name__ == "__main__":
    unittest.main()
