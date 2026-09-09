"""Shared Docker lifecycle for serving-engine workloads."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import docker
import requests
from docker.types import DeviceRequest

_ENGINE_PARALLEL_ARGUMENTS = {
    "vllm": {"tp": "tensor_parallel_size", "dp": "data_parallel_size"},
    "sglang": {"tp": "tp_size", "dp": "dp_size"},
}
_ENGINE_DEFAULT_ENTRYPOINTS = {
    "sglang": ["python3", "-m", "sglang.launch_server"],
}
_ENGINE_DEFAULT_HEALTH_PATHS = {
    "vllm": "/health",
    "sglang": "/health_generate",
}


def engine_family(engine_config: dict[str, Any]) -> str:
    """Return the engine implementation while allowing descriptive run names."""
    configured = str(engine_config.get("kind") or engine_config.get("name", "vllm")).lower()
    for family in _ENGINE_PARALLEL_ARGUMENTS:
        if configured == family or configured.startswith((f"{family}-", f"{family}_")):
            return family
    return configured


def engine_parallel_arguments(engine_config: dict[str, Any]) -> dict[str, str] | None:
    """Return the engine-specific tensor/data parallel argument names."""
    return _ENGINE_PARALLEL_ARGUMENTS.get(engine_family(engine_config))


def engine_health_path(engine_config: dict[str, Any]) -> str:
    """Return an overrideable readiness endpoint for the configured engine."""
    path = str(
        engine_config.get("health_path")
        or _ENGINE_DEFAULT_HEALTH_PATHS.get(engine_family(engine_config), "/health")
    )
    return f"/{path.lstrip('/')}"


class DockerEngineRuntime:
    """Engine argument construction and Docker lifecycle shared by toolkit workflows.

    Subclasses provide ``config``, ``logger``, ``docker_client``, ``cache_dir``, and
    ``hf_token``. The deliberately small interface keeps benchmark policy out of the
    serving-engine lifecycle.
    """

    config: dict[str, Any]
    logger: logging.Logger
    docker_client: Any
    cache_dir: str
    hf_token: str
    container_name_prefix = "llm_tuning_engine"

    def _set_display_status(self, status: str) -> None:
        """Allow interactive workflows to publish engine lifecycle changes."""

    def _build_engine_args(self, param_config: dict[str, Any] | None = None) -> list[str]:
        """Build engine arguments from the fixed recipe and optional sweep values."""
        args = [str(argument) for argument in self.config["engine"]["base_args"]]
        parameters = param_config or {}
        for param, value in parameters.get("value_args", {}).items():
            args.extend([f"--{param.replace('_', '-')}", str(value)])
        for param, value in parameters.get("action_args", {}).items():
            if value:
                args.append(f"--{param.replace('_', '-')}")
        return args

    def _launch_docker_engine(self, engine_args: list[str]) -> Any | None:
        """Launch the configured serving engine, returning ``None`` on failure."""
        try:
            engine_config = self.config["engine"]
            port = self.config["port"]
            container_name = f"{self.container_name_prefix}_{uuid.uuid4().hex[:8]}"
            self.logger.info("Starting engine container: %s", container_name)

            docker_config = engine_config.get("docker", {})
            docker_args = {
                key: value
                for key, value in docker_config.items()
                if key in {"devices", "group_add", "security_opt", "ipc_mode"}
            }
            if "devices" in engine_config:
                docker_args["device_requests"] = [
                    DeviceRequest(device_ids=engine_config["devices"], capabilities=[["gpu"]])
                ]
            environment = {
                "HF_TOKEN": self.hf_token,
                "HF_HUB_CACHE": "/data/",
                **docker_config.get("environment", {}),
            }
            run_args = {
                "image": engine_config["image"],
                "command": engine_args,
                "shm_size": docker_config.get("shm_size", "2g"),
                "environment": environment,
                "volumes": {self.cache_dir: {"bind": "/data/", "mode": "rw"}},
                "ports": {f"{port}/tcp": port},
                "detach": True,
                "name": container_name,
                "stop_signal": "SIGTERM",
                **docker_args,
            }
            entrypoint = engine_config.get("entrypoint")
            if entrypoint is None:
                entrypoint = _ENGINE_DEFAULT_ENTRYPOINTS.get(engine_family(engine_config))
            if entrypoint is not None:
                run_args["entrypoint"] = entrypoint
            return self.docker_client.containers.run(**run_args)
        except Exception as error:
            self.logger.error("Failed to launch engine: %s", error)
            return None

    def _wait_for_server_ready(self, container: Any, port: int, timeout: int = 700) -> bool:
        """Wait until the container's configured health endpoint responds."""
        self.logger.info("Waiting for engine to be ready...")
        self._set_display_status("Waiting for engine health check")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status != "running":
                    self.logger.error("Engine container has stopped unexpectedly.")
                    return False
                response = requests.get(
                    f"http://localhost:{port}{engine_health_path(self.config['engine'])}", timeout=5
                )
                if response.status_code == 200:
                    self.logger.info("Engine is ready!")
                    self._set_display_status("Engine ready")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        self.logger.error("Timeout waiting for engine to be ready")
        return False

    def _cleanup_container(self, container: Any) -> None:
        """Stop and remove an engine container without hiding cleanup failures."""
        self._set_display_status("Stopping engine")
        try:
            self.logger.info("Stopping container...")
            container.stop(timeout=100)
        except Exception as error:
            self.logger.warning("Error stopping container; forcing removal: %s", error)
        try:
            self.logger.info("Removing container...")
            container.remove(force=True)
        except Exception as error:
            self.logger.warning("Error removing container: %s", error)
