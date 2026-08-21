import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune import cli


class AutoTuneCliTests(unittest.TestCase):
    def test_trackio_options_and_hf_token_are_forwarded(self):
        args = SimpleNamespace(
            config="config.yaml",
            result_dir="results",
            dataset_id=None,
            cache_dir="cache",
            hf_token="hf-token",
            trackio_project="benchmarks",
            trackio_space_id="user/trackio",
            trackio_server_url=None,
            trackio_group="nightly",
            no_ui=True,
        )
        tuner = Mock()

        with (
            patch.object(cli.parser, "parse_args", return_value=args),
            patch("auto_tune.cli.os.path.exists", return_value=True),
            patch("auto_tune.cli.AutoTuner", return_value=tuner) as auto_tuner,
        ):
            cli.main()

        auto_tuner.assert_called_once_with(
            config_path="config.yaml",
            result_dir="results",
            dataset_id=None,
            cache_dir="cache",
            hf_token="hf-token",
            trackio_project="benchmarks",
            trackio_space_id="user/trackio",
            trackio_server_url=None,
            trackio_group="nightly",
        )
        tuner.run_auto_tune.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
