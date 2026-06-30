import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from benchmarker.results import extract_inference_benchmarker_metrics


class BenchmarkExecutionError(RuntimeError):
    pass


def serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    return ",".join(
        f"{key}={serialize_metadata_value(value)}" for key, value in metadata.items() if value is not None
    )


def serialize_metadata_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    value = str(value)
    if any(separator in value for separator in [",", "=", "\n"]):
        return quote(value, safe="/:-_.")
    return value


@dataclass
class BenchmarkRun:
    args: Sequence[Any]
    extra_meta: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRequest:
    kind: str
    url: str
    max_vus: int
    duration: str
    tokenizer_name: str
    output_path: str
    run_id: str
    prompt_options: Optional[str] = None
    decode_options: Optional[str] = None
    dataset_file: Optional[str] = None
    rate: Optional[float] = None
    extra_meta: Optional[Dict[str, Any]] = None


class BenchmarkExecutor:
    name = "command"
    supports_extra_meta = False

    def __init__(self, command: Sequence[str], logger: Optional[logging.Logger] = None, verbose: bool = False):
        self.command = list(command)
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

    def build_command(self, run: BenchmarkRun) -> List[str]:
        return [str(part) for part in [*self.command, *run.args]]

    def build_request_run(self, request: BenchmarkRequest) -> BenchmarkRun:
        raise NotImplementedError(
            f"{self.name} cannot build structured benchmark requests yet. "
            "Use backend-specific arguments in a multi-benchmark config or implement this executor adapter."
        )

    def run(self, run: BenchmarkRun) -> subprocess.CompletedProcess:
        cmd = self.build_command(run)
        self.logger.info(f"Running benchmark: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None, check=True)
        except FileNotFoundError as exc:
            raise BenchmarkExecutionError(f"Benchmark command not found: {cmd[0]}") from exc
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"Benchmark failed with return code {exc.returncode}: {' '.join(cmd)}")
            self.logger.error(f"STDOUT:\n{exc.stdout if exc.stdout else 'None'}")
            self.logger.error(f"STDERR:\n{exc.stderr if exc.stderr else 'None'}")
            raise BenchmarkExecutionError(str(exc)) from exc

        if self.verbose:
            self.logger.debug(f"Return code: {proc.returncode}")
            self.logger.debug(f"STDOUT:\n{proc.stdout}")
            if proc.stderr:
                self.logger.debug(f"STDERR:\n{proc.stderr}")
        self.logger.info("Benchmark completed successfully")
        return proc

    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not provide a metrics parser yet")


class InferenceBenchmarkerExecutor(BenchmarkExecutor):
    name = "inference-benchmarker"
    supports_extra_meta = True

    def __init__(
        self,
        command: Sequence[str] = ("inference-benchmarker",),
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        super().__init__(command, logger, verbose)

    def build_command(self, run: BenchmarkRun) -> List[str]:
        cmd = super().build_command(run)
        extra_meta = serialize_metadata(run.extra_meta)
        if extra_meta:
            cmd.extend(["--extra-meta", extra_meta])
        cmd.append("--no-console")
        return cmd

    def build_request_run(self, request: BenchmarkRequest) -> BenchmarkRun:
        args = [
            "--url",
            request.url,
            "--benchmark-kind",
            request.kind,
            "--max-vus",
            request.max_vus,
            "--duration",
            request.duration,
            "--tokenizer-name",
            request.tokenizer_name,
            "--output-path",
            request.output_path,
            "--run-id",
            request.run_id,
        ]
        if request.kind == "rate":
            if request.rate is None:
                raise ValueError("Rate benchmark requests require a rate.")
            args.extend(["--rates", request.rate])
        if request.prompt_options:
            args.extend(["--prompt-options", request.prompt_options])
        if request.decode_options:
            args.extend(["--decode-options", request.decode_options])
        if request.dataset_file:
            args.extend(["--dataset-file", request.dataset_file])
        return BenchmarkRun(args=args, extra_meta=request.extra_meta)

    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return extract_inference_benchmarker_metrics(results)


class VllmBenchExecutor(BenchmarkExecutor):
    name = "vllm-bench"

    def __init__(
        self,
        command: Sequence[str] = ("vllm", "bench", "serve"),
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        super().__init__(command, logger, verbose)


def create_benchmark_executor(
    backend: str,
    logger: Optional[logging.Logger] = None,
    command: Optional[str] = None,
    verbose: bool = False,
) -> BenchmarkExecutor:
    command_parts = command.split() if command else None
    backend = backend.lower()
    if backend in ("inference", "inference-benchmarker"):
        return InferenceBenchmarkerExecutor(command_parts or ("inference-benchmarker",), logger, verbose)
    if backend in ("vllm", "vllm-bench"):
        return VllmBenchExecutor(command_parts or ("vllm", "bench", "serve"), logger, verbose)
    raise ValueError(f"Unknown benchmark backend: {backend}")
