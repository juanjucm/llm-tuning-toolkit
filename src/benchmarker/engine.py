import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import docker
import requests
from docker.models.containers import Container


@dataclass
class EngineSpec:
    name: str
    image: str
    command: Sequence[Any]
    port: int
    devices: Union[List[str], str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stop_signal: Optional[str] = None


def sanitize_name(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_") or "engine"


def cache_mount(cache_dir: Optional[str]) -> Dict[str, Dict[str, str]]:
    return {cache_dir: {"bind": "/data/", "mode": "rw"}} if cache_dir else {}


def token_environment(hf_token: str = "", cache_dir: Optional[str] = None) -> Dict[str, str]:
    env = {}
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if cache_dir:
        env["HF_HUB_CACHE"] = "/data/"
    return env


def env_list_to_dict(values: Optional[Sequence[str]]) -> Dict[str, str]:
    env = {}
    for value in values or []:
        if "=" in value:
            key, val = value.split("=", 1)
            env[key] = val
    return env


def normalize_devices(devices: Any) -> Union[List[str], str]:
    if not devices:
        return []
    if isinstance(devices, str):
        if devices.lower() == "all":
            return "all"
        return [device.strip() for device in devices.split(",") if device.strip()]

    normalized = [str(device) for device in devices]
    if any(device.lower() == "all" for device in normalized):
        return "all"
    return normalized


def spec_from_engine_config(
    engine: Dict[str, Any],
    port: int,
    extra_env: Optional[Dict[str, str]] = None,
    volumes: Optional[Dict[str, Dict[str, str]]] = None,
) -> EngineSpec:
    env = env_list_to_dict(engine.get("envs"))
    env.update(extra_env or {})
    return EngineSpec(
        name=engine.get("name", engine.get("image", "engine")),
        image=engine["image"],
        command=engine.get("args") or engine.get("base_args") or [],
        port=port,
        devices=normalize_devices(engine.get("devices", [])),
        environment=env,
        volumes=volumes or {},
        stop_signal=engine.get("stop_signal"),
    )


class DockerEngineRunner:
    def __init__(self, logger: Optional[logging.Logger] = None, docker_client=None):
        self.logger = logger or logging.getLogger(__name__)
        self.client = docker_client or docker.from_env()

    def _device_requests(self, devices: Union[List[str], str]):
        if not devices:
            return None
        if devices == "all":
            return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        return [docker.types.DeviceRequest(device_ids=list(devices), capabilities=[["gpu"]])]

    def launch(self, spec: EngineSpec, name_prefix: str = "engine") -> Optional[Container]:
        container_name = f"{sanitize_name(name_prefix)}_{sanitize_name(spec.name)}_{uuid.uuid4().hex[:4]}"
        self.logger.info(f"Starting engine container: {container_name}")
        kwargs = {
            "image": spec.image,
            "command": " ".join(str(arg) for arg in spec.command),
            "environment": spec.environment,
            "ports": {f"{spec.port}/tcp": spec.port},
            "detach": True,
            "name": container_name,
        }
        if spec.volumes:
            kwargs["volumes"] = spec.volumes
        if spec.stop_signal:
            kwargs["stop_signal"] = spec.stop_signal
        device_requests = self._device_requests(spec.devices)
        if device_requests:
            kwargs["device_requests"] = device_requests

        try:
            return self.client.containers.run(**kwargs)
        except Exception as exc:
            self.logger.error(f"Failed to launch engine: {exc}")
            try:
                container = self.client.containers.get(container_name)
                self.logger.warning(
                    f"Container {container_name} exists after launch error; continuing with status {container.status}."
                )
                return container
            except Exception:
                return None

    def log_tail(self, container: Container, lines: int = 80):
        try:
            logs = container.logs(tail=lines)
            logs = logs.decode(errors="replace") if isinstance(logs, bytes) else str(logs)
            if logs.strip():
                self.logger.error(f"Engine container logs (last {lines} lines):\n{logs.rstrip()}")
            else:
                self.logger.warning("Engine container produced no logs.")
        except Exception as exc:
            self.logger.warning(f"Failed to read engine container logs: {exc}")

    def wait_ready(self, container: Container, port: int, timeout: int = 600, health_path: str = "/health") -> bool:
        self.logger.info("Waiting for engine to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status in ("created", "restarting"):
                    time.sleep(2)
                    continue
                if container.status != "running":
                    self.logger.error(f"Engine container stopped with status: {container.status}.")
                    self.log_tail(container)
                    return False

                response = requests.get(f"http://localhost:{port}{health_path}", timeout=5)
                if response.status_code == 200:
                    self.logger.info("Engine is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)

        self.logger.error("Timeout waiting for engine to be ready")
        self.log_tail(container)
        return False

    def cleanup(self, container: Container, timeout: int = 30):
        try:
            self.logger.info("Stopping container...")
            container.stop(timeout=timeout)
        except Exception:
            self.logger.warning("Error stopping container, forcing removal.")
        try:
            container.remove(force=True)
        except Exception as exc:
            self.logger.warning(f"Failed to remove container {container.name}: {exc}")
