#!/usr/bin/env python3

import argparse
import subprocess
import yaml
import json
import logging
import uuid
import time
import docker
from pathlib import Path
from typing import Dict, List, Any, Optional


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
        '--scenario',
        type=str,
        help='Specific scenario to run (if not specified, runs all scenarios)'
    )
    parser.add_argument(
        '--engine',
        type=str,
        help='Specific engine to test (if not specified, tests all engines)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands that would be executed without running them'
    )
    return parser.parse_args()


def get_server_config_from_logs(container_name: str, engine_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """Extract server configuration from Docker container logs for TGI, SGLang, and vLLM."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        
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


def launch_docker_engine(engine_name: str, engine_config: Dict[str, Any], 
                        model: str, port: int, logger: logging.Logger) -> Optional[str]:
    """Launch a Docker container for the specified engine."""
    try:
        client = docker.from_env()
        
        # Args are already resolved by YAML parser
        args = engine_config.get('args', [])
        
        # Set up environment variables
        environment = {}
        for env in engine_config.get('envs', []):
            if '=' in env:
                key, value = env.split('=', 1)
                environment[key] = value
        
        container_name = f"bench_{engine_name.lower()}_{int(time.time())}"
        
        logger.info(f"Starting {engine_name} container: {container_name}")
        
        # Set up device requests (GPU support)
        # NOTE: update for accepting cpu or mps.
        device_requests = []
        if 'devices' in engine_config:
            device_config = engine_config['devices']
            if isinstance(device_config, str):
                # Simple string format like "all" or "0,1"
                device_ids = [device_config] if device_config == "all" else device_config.split(',')
                device_requests = [
                    docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
                ]
        else:
            # Default: use all GPUs
            device_requests = [
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
            ]

        container = client.containers.run(
            engine_config['image'],
            command=' '.join(args) if args else None,
            environment=environment,
            ports={f'{port}/tcp': port},
            detach=True,
            name=container_name,
            remove=False,
            auto_remove=False,
            device_requests=device_requests
        )
        
        # Wait for container to be ready
        time.sleep(10)
        
        return container_name
        
    except Exception as e:
        logger.error(f"Failed to launch {engine_name}: {e}")
        return None


def wait_for_server_ready(port: int, timeout: int = 60) -> bool:
    """Wait for the server to be ready to accept requests."""
    import requests
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    return False


def generate_unique_run_id() -> str:
    """Generate a unique run ID for the benchmark."""
    return f"run_{uuid.uuid4().hex[:8]}_{int(time.time())}"


def run_benchmark(scenario_name: str, scenario_config: Dict[str, Any], 
                 engine_name: str, run_id: str, server_config: Dict[str, Any],
                 config: Dict[str, Any], logger: logging.Logger, 
                 dry_run: bool = False) -> bool:
    """Run the benchmark using inference-benchmarker command."""
    try:
        # Build the command
        cmd = ['inference-benchmarker']
        
        # Add benchmark configuration (YAML already resolved placeholders)
        cmd.extend(scenario_config['bench_config'])
        
        # Add run ID
        cmd.extend(['--run-id', run_id])
        
        # Add metadata
        metadata = {
            'engine': engine_name,
            'scenario': scenario_name,
            'config': {
                'model': config.get('model'),
                'scenario_description': scenario_config.get('description', ''),
                'server_config': server_config
            }
        }
        
        cmd.extend(['--extra-meta', json.dumps(metadata)])
        
        logger.info(f"Running benchmark: {' '.join(cmd)}")
        
        if dry_run:
            logger.info("DRY RUN: Would execute the above command")
            return True
        
        # Execute the benchmark
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info(f"Benchmark completed successfully for {engine_name} - {scenario_name}")
            logger.debug(f"Benchmark output: {result.stdout}")
            return True
        else:
            logger.error(f"Benchmark failed for {engine_name} - {scenario_name}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Benchmark timed out for {engine_name} - {scenario_name}")
        return False
    except Exception as e:
        logger.error(f"Error running benchmark for {engine_name} - {scenario_name}: {e}")
        return False


def cleanup_container(container_name: str, logger: logging.Logger):
    """Clean up Docker container."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        container.stop(timeout=10)
        container.remove()
        logger.info(f"Cleaned up container: {container_name}")
    except Exception as e:
        logger.warning(f"Failed to cleanup container {container_name}: {e}")


def main():
    """Main execution function."""
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        # Load configuration
        config = load_bench_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Get model and port from config
        model = config.get('model')
        port = config.get('port', 8000)
        
        if not model:
            logger.error("No model specified in configuration")
            return 1
        
        # Filter scenarios if specified
        scenarios_to_run = {}
        if args.scenario:
            if args.scenario in config['scenarios']:
                scenarios_to_run[args.scenario] = config['scenarios'][args.scenario]
            else:
                logger.error(f"Scenario '{args.scenario}' not found in configuration")
                return 1
        else:
            scenarios_to_run = config['scenarios']
        
        total_runs = 0
        successful_runs = 0
        
        # Run benchmarks for each scenario
        for scenario_name, scenario_config in scenarios_to_run.items():
            logger.info(f"Starting scenario: {scenario_name}")
            logger.info(f"Description: {scenario_config.get('description', 'No description')}")
            
            # Filter engines if specified
            engines_to_test = {}
            if args.engine:
                if args.engine in scenario_config['engines']:
                    engines_to_test[args.engine] = scenario_config['engines'][args.engine]
                else:
                    logger.error(f"Engine '{args.engine}' not found in scenario '{scenario_name}'")
                    continue
            else:
                engines_to_test = scenario_config['engines']
            
            # Test each engine in the scenario
            for engine_name, engine_config in engines_to_test.items():
                total_runs += 1
                run_id = generate_unique_run_id()
                container_name = None
                
                try:
                    logger.info(f"Testing {engine_name} for scenario {scenario_name}")
                    
                    # Launch Docker container for the engine
                    if not args.dry_run:
                        container_name = launch_docker_engine(
                            engine_name, engine_config, model, port, logger
                        )
                        
                        if not container_name:
                            logger.error(f"Failed to launch {engine_name}")
                            continue
                        
                        # Wait for server to be ready
                        if not wait_for_server_ready(port):
                            logger.error(f"Server {engine_name} failed to start properly")
                            continue
                        
                        # Get server configuration from logs
                        server_config = get_server_config_from_logs(container_name, engine_name, logger)
                    else:
                        server_config = {}
                    
                    # Run benchmark
                    success = run_benchmark(
                        scenario_name, scenario_config, engine_name, 
                        run_id, server_config, config, logger, args.dry_run
                    )
                    
                    if success:
                        successful_runs += 1
                    
                except Exception as e:
                    logger.error(f"Error testing {engine_name} in {scenario_name}: {e}")
                
                finally:
                    # Clean up container
                    if container_name and not args.dry_run:
                        cleanup_container(container_name, logger)
                    
                    # Brief pause between engine tests
                    if not args.dry_run:
                        time.sleep(5)
        
        # Summary
        logger.info(f"Benchmark execution completed")
        logger.info(f"Total runs: {total_runs}, Successful: {successful_runs}, Failed: {total_runs - successful_runs}")
        
        return 0 if successful_runs == total_runs else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())