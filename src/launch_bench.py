import argparse
import json
import logging
import docker
from docker.models.containers import Container
import os
import requests
import subprocess
import time
import threading
from typing import Dict, List, Any
import uuid
import yaml

import coloredlogs
coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK_TOOL_CMD = "inference-benchmarker"


def setup_logging():
    """Configure logging for the benchmark launcher."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_bench_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch benchmarks based on configuration file"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='bench_config.yaml',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        help='Specific scenarios to run, comma separated (i.e: "s1,s2,s3") (if not specified, runs all scenarios)',
        default="all"
    )
    parser.add_argument(
        '--engines',
        type=str,
        help='Specific engines to test, comma separated (i.e: "e1,e2,e3") (if not specified, tests all engines)',
        default="all"
    )
    parser.add_argument(
        '--show-logs',
        action="store_true",
        help="Show engine container logs."
        
    )

    return parser.parse_args()

def get_server_config_from_logs(container: Container, engine_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """Extract server configuration from Docker container logs for TGI, SGLang, and vLLM."""
    try:  
        # Get more comprehensive logs
        logs = container.logs(tail=200).decode('utf-8')
        
        config = {'engine': engine_name, 'raw_config': {}}
        
        if engine_name.lower() == 'tgi':
            config.update(_parse_tgi_logs(logs))
        elif engine_name.lower() == 'sglang':
            config.update(_parse_sglang_logs(logs))
        elif engine_name.lower() == 'vllm':
            config.update(_parse_vllm_logs(logs))
        
        logger.info(f"Extracted {engine_name} config: {config}")
        return config
        
    except Exception as e:
        logger.warning(f"Failed to extract server config from logs: {e}")
        return {'engine': engine_name, 'error': str(e)}

def _parse_tgi_logs(logs: str) -> Dict[str, Any]:
    """Parse TGI (Text Generation Inference) logs for configuration."""
    config = {}
    
    for line in logs.split('\n'):
        line = line.strip()
        
        # TGI logs configuration at startup
        if 'Args {' in line or 'Args(' in line:
            # Extract args block
            config['startup_args'] = line
        
        # Model information
        if 'Loading model' in line or 'Model loaded' in line:
            config['model_loading'] = line
            
        # Memory and GPU info
        if 'memory' in line.lower() and ('gpu' in line.lower() or 'cuda' in line.lower()):
            config['memory_info'] = line
            
        # Quantization info
        if 'quantization' in line.lower() or 'dtype' in line.lower():
            config['quantization'] = line
            
        # Sharding info
        if 'shard' in line.lower() or 'tensor parallel' in line.lower():
            config['sharding'] = line
            
        # Server ready
        if 'Ready' in line and 'server' in line.lower():
            config['server_ready'] = line
            
        # Configuration summary (TGI often logs final config)
        if 'max_total_tokens' in line or 'max_input_length' in line:
            config['token_limits'] = line
    
    return config

def _parse_sglang_logs(logs: str) -> Dict[str, Any]:
    """Parse SGLang logs for configuration."""
    config = {}
    
    for line in logs.split('\n'):
        line = line.strip()
        
        # SGLang startup args
        if 'Arguments:' in line or 'Args:' in line:
            config['startup_args'] = line
            
        # Model loading
        if 'Loading model' in line or 'Model' in line and 'loaded' in line:
            config['model_info'] = line
            
        # Memory configuration
        if 'memory' in line.lower() and ('pool' in line.lower() or 'fraction' in line.lower()):
            config['memory_config'] = line
            
        # Context length
        if 'context' in line.lower() and ('length' in line.lower() or 'size' in line.lower()):
            config['context_config'] = line
            
        # CUDA graph info
        if 'cuda' in line.lower() and 'graph' in line.lower():
            config['cuda_graph'] = line
            
        # P2P and scheduling
        if 'p2p' in line.lower() or 'schedule' in line.lower():
            config['scheduling'] = line
            
        # Server listening
        if 'Uvicorn running' in line or 'server' in line.lower() and 'listening' in line.lower():
            config['server_info'] = line
            
        # Configuration summary
        if 'ServerArgs' in line or 'server_args' in line:
            config['server_args'] = line
    
    return config

def _parse_vllm_logs(logs: str) -> Dict[str, Any]:
    """Parse vLLM logs for configuration."""
    config = {}
    
    for line in logs.split('\n'):
        line = line.strip()
        
        # vLLM initialization args
        if 'Initializing an LLM engine' in line:
            config['engine_init'] = line
            
        # Model and tokenizer
        if 'Loading model' in line or 'Loaded tokenizer' in line:
            config['model_loading'] = line
            
        # GPU memory utilization
        if 'gpu_memory_utilization' in line.lower():
            config['gpu_memory'] = line
            
        # Model parallel and tensor parallel
        if 'tensor_parallel_size' in line or 'pipeline_parallel_size' in line:
            config['parallelism'] = line
            
        # Block management and KV cache
        if 'block' in line.lower() and ('manager' in line.lower() or 'allocator' in line.lower()):
            config['block_manager'] = line
            
        # Chunked prefill
        if 'chunked' in line.lower() and 'prefill' in line.lower():
            config['chunked_prefill'] = line
            
        # Max model length
        if 'max_model_len' in line.lower():
            config['max_model_len'] = line
            
        # Server configuration
        if 'Running on' in line or 'Uvicorn running' in line:
            config['server_info'] = line
            
        # Engine configuration summary
        if 'EngineArgs' in line or 'LLMEngine' in line:
            config['engine_args'] = line
    
    return config

def get_engine_config_dict(engine: Dict) -> Dict:
    config = {}
    config['image'] = engine.get('image')
    config['name'] = engine.get('name', config['image'])
    config['args'] = engine.get('args')
    config['cmd'] = ' '.join([str(a) for a in config['args']])
    environment = {}
    envs = engine.get('envs')
    if envs:
        for env in envs:
                if '=' in env:
                    key, value = env.split('=', 1)
                    environment[key] = value
    
    config['envs'] = environment
    config['envs']['HF_TOKEN'] = HF_TOKEN
    config['devices'] = engine.get('devices', [])

    return config

def launch_docker_engine(engine_name: str,
                         engine_config: Dict[str, Any],
                         port: int,
                         logger: logging.Logger) -> Container:
    try:
        container_name = f"bench_{engine_name.lower()}"
        logger.info(f"Starting {engine_name} container: {container_name}")
        
        # NOTE: update for supporting cpu and mps.
        device_requests = []
        if 'devices' in engine_config:
            device_config = engine_config['devices']
            if isinstance(device_config, str):
                device_ids = [device_config] if device_config == "all" else device_config.split(',')
                device_requests = [
                    docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
                ]
        else:
            device_requests = [
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
            ]

        client = docker.from_env()
        container = client.containers.run(
            image=engine_config.get('image'),
            command=engine_config.get('cmd'),
            environment=engine_config.get('envs'),
            ports={f'{port}/tcp': port},
            detach=True,
            name=container_name,
            remove=False,
            auto_remove=False,
            device_requests=device_requests
        )
        
        return container
        
    except Exception as e:
        logger.error(f"Failed to launch {engine_name}: {e}")
        return None

def wait_for_server_ready(port: int, logger: logging.Logger, timeout: int = 300) -> bool:
    """Wait for the server to be ready to accept requests."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            logger.info(f"Waiting for engine to be ready...")
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    return False

def generate_unique_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:8]}"

def run_benchmark(scenario_name: str,
                  scenario_description: str,
                  bench_args: List, 
                  engine_name: str,
                  engine_config: str,
                  run_id: str,
                  logger: logging.Logger):
    try:
        cmd = [BENCHMARK_TOOL_CMD]
        cmd.extend(['--no-console']) # disable UI so the process doesn't get stuck when finished.
        cmd.extend(bench_args)
        cmd.extend(['--run-id', run_id])
        
        metadata = f"engine={engine_name},scenario={scenario_name},scenario_description={scenario_description},engine_config={json.dumps(engine_config)}"
        cmd.extend(['--extra-meta', metadata])

        cmd = [str(c) for c in cmd]
        
        logger.info(f"Running benchmark: {' '.join(cmd)}")
        proc = subprocess.run(cmd,
                        capture_output=True,
                        text=True,
                        timeout=None,
                        check=True)
        
        logger.info(f"Benchmark completed successfully for {engine_name} - {scenario_name}")
        logger.info(proc.stdout)
    except Exception as e:
        logger.error(f"Error running benchmark for {engine_name} - {scenario_name}: {e}")

def cleanup_container(container: Container, logger: logging.Logger):
    try:
        container.stop(timeout=20)
        container.remove()
    except Exception as e:
        logger.warning(f"Failed to cleanup container {container.name}: {e}")

def stream_container_logs(container_name):
    def log_stream():
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            for line in process.stdout:
                print(f"[{container_name}] {line.rstrip()}")
        except Exception as e:
            print(f"Error streaming logs for {container_name}: {e}")

    thread = threading.Thread(target=log_stream, daemon=True)
    thread.start()
    return thread

def main():
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        config = load_bench_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        model = config.get('model')
        port = config.get('port', 8000)
        
        if not model:
            logger.error("No model specified in configuration")
            return 1
        
        # Filter scenarios
        scenarios_to_run = []
        requested_scenarios = args.scenarios.split(',')
        if requested_scenarios == ["all"]:
            scenarios_to_run = config['scenarios']
        else:
            for s in config['scenarios']:
                if s['name'] in requested_scenarios:
                    scenarios_to_run.append(s)
        
        if not scenarios_to_run:
            logger.error("No scenarios to run.")
            return 1
        
        logger.info(f"Running scenarios: {[s['name'] for s in scenarios_to_run]}")
        
        total_runs = 0
        successful_runs = 0
        requested_engines = args.engines.split(',')
        for scenario in scenarios_to_run:
            logger.info(f"Starting scenario: {scenario['name']} - {scenario.get('description', 'No description')}")

            scenario_run_id = generate_unique_run_id()
            scenario_name = scenario.get('name', '')
            scenario_description = scenario.get('description', '')
            scenario_bench_args = scenario.get('bench_args')
            
            # Filter engines
            engines_to_test = []
            if requested_engines == ["all"]:
                engines_to_test = scenario['engines']
            else:
                for e in scenario['engines']:
                    if e['name'] in requested_engines:
                        engines_to_test.append(e)

            if engines_to_test:
                logger.info(f"Running engines: {[e['name'] for e in engines_to_test]}")
            else:
                logger.error(f"Requested engine/s are not defined for scenario {scenario['name']}. Defined engines for scenario {scenario['name']}: {[e['name'] for e in scenario['engines']]}")
            
            for engine in engines_to_test:
                total_runs += 1
                try:
                    engine_name = engine['name']
                    engine_config = get_engine_config_dict(engine)
                    container = launch_docker_engine(
                        engine_name=engine_name,
                        engine_config=engine_config,
                        port=port,
                        logger=logger
                    )

                    logs_thread = None
                    if args.show_logs:
                        logs_thread = stream_container_logs(container.name)
                    
                    if not wait_for_server_ready(port, logger=logger):
                        logger.error(f"Timeout - Server {engine_name} failed to start properly.")
                        continue
                    
                    # Get server configuration from logs
                    server_config={}
                    # server_config = get_server_config_from_logs(container_name, engine_name, logger)
                    
                    # Run benchmark
                    run_benchmark(
                        scenario_name=scenario_name,
                        scenario_description=scenario_description,
                        bench_args=scenario_bench_args,
                        engine_name=engine_name,
                        engine_config=server_config,
                        run_id=scenario_run_id,
                        logger=logger
                    )
                    
                except Exception as e:
                    logger.error(f"Error testing '{engine_name}' in '{scenario_name}': {e}")
                
                finally:
                    # Clean up container
                    if container:
                        cleanup_container(container, logger)
                        if logs_thread:
                            logs_thread.join()
        
        # Summary
        logger.info(f"Benchmark execution completed.")
        logger.info(f"Total runs: {total_runs}, Successful: {successful_runs}, Failed: {total_runs - successful_runs}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())