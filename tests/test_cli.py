import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune import cli


class AutoTuneCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_trackio_options_and_hf_token_are_forwarded(self):
        tuner = Mock()

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("scenario: {}\n")
            with patch("auto_tune.cli.AutoTuner", return_value=tuner) as auto_tuner:
                result = self.runner.invoke(
                    cli.app,
                    [
                        "--config",
                        str(config),
                        "--result-dir",
                        str(Path(directory) / "results"),
                        "--cache-dir",
                        str(Path(directory) / "cache"),
                        "--hf-token",
                        "hf-token",
                        "--trackio-project",
                        "benchmarks",
                        "--trackio-space-id",
                        "user/trackio",
                        "--trackio-group",
                        "nightly",
                        "--no-ui",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            auto_tuner.assert_called_once_with(
                config_path=str(config.resolve()),
                result_dir=str(Path(directory) / "results"),
                dataset_id=None,
                cache_dir=str(Path(directory) / "cache"),
                hf_token="hf-token",
                trackio_project="benchmarks",
                trackio_space_id="user/trackio",
                trackio_server_url=None,
                trackio_group="nightly",
                quiet=True,
            )
            tuner.run_auto_tune.assert_called_once_with()

    def test_trackio_destinations_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("scenario: {}\n")
            with patch("auto_tune.cli.AutoTuner") as auto_tuner:
                result = self.runner.invoke(
                    cli.app,
                    [
                        "--config",
                        str(config),
                        "--trackio-project",
                        "benchmarks",
                        "--trackio-space-id",
                        "user/trackio",
                        "--trackio-server-url",
                        "http://localhost:7860",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("use either --trackio-space-id", result.output)
        self.assertIn("--trackio-server-url, not both", result.output)
        auto_tuner.assert_not_called()

    def test_results_are_persistent_by_default(self):
        tuner = Mock()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("scenario: {}\n")
            with patch("auto_tune.cli.AutoTuner", return_value=tuner) as auto_tuner:
                result = self.runner.invoke(cli.app, ["--config", str(config), "--no-ui"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(auto_tuner.call_args.kwargs["result_dir"], "out")
        self.assertTrue(auto_tuner.call_args.kwargs["quiet"])
        tuner.run_auto_tune.assert_called_once_with()

    def test_help_lists_the_primary_options(self):
        result = self.runner.invoke(cli.app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--config", result.output)
        self.assertIn("--no-ui", result.output)
        self.assertIn("--trackio-project", result.output)


if __name__ == "__main__":
    unittest.main()
