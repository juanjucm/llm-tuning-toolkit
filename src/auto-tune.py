import argparse
import json
import logging
import os
import subprocess
import sys
import time
import docker
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import yaml
import coloredlogs

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK_TOOL_CMD = "inference-benchmarker"


parser = argparse.ArgumentParser(description="Auto-tune tool for finding optimal engine parameters.")
parser.add_argument(
        "--config", 
        default="src/benchmarker/auto-tune-config.yaml",
        help="Path to auto-tune configuration file"
)
parser.add_argument(
        "--result-dir",
        default="auto_tune_results",
        help="Directory to save tuning results"
)


class AutoTuner:
    def __init__(self, config_path: str, result_dir: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_dir = Path(result_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Best configuration tracking
        self.best_config = None
        self.best_throughput = 0
        
        # Docker client
        self.docker_client = docker.from_env()
        
    def _load_config(self) -> Dict:
        """Load and validate configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['goodput_thresholds', 'scneario', 'engine']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        return config
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        return f"autotune_{uuid.uuid4().hex[:8]}"
    
    def _build_engine_args(self, param_config: Dict) -> List[str]:
        """Build engine arguments from configuration
        """
        # TODO: move this outside to avoid rebuilding every time
        # Like, define base args in the main loop or something and just append params here.
        args = list(self.config['engine']['base_args'])
        
        # Handle value arguments (--param value)
        for param, value in param_config.get('value_args', {}).items():
            args.extend([f"--{param.replace('_', '-')}", str(value)])
        
        # Handle action arguments (boolean flags)
        for param, value in param_config.get('action_args', {}).items():
            if value:
                args.append(f"--{param.replace('_', '-')}")
        
        return args
    
    def _launch_docker_engine(self, param_config: Dict) -> Optional[docker.models.containers.Container]:
        """Launch a Docker container running the engine
        """
        try:
            engine_config = self.config['engine']
            engine_args = self._build_engine_args(param_config)
            port = self.config['port']
            
            container_name = f"autotune_engine_{int(time.time())}"
            self.logger.info(f"Starting engine container: {container_name}")
            
            device_requests = [
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
            ]
            
            container = self.docker_client.containers.run(
                image=engine_config['image'],
                command=' '.join([str(a) for a in engine_args]),
                environment={'HF_TOKEN': HF_TOKEN},
                ports={f'{port}/tcp': port},
                detach=True,
                name=container_name,
                remove=False,
                auto_remove=False,
                device_requests=device_requests
            )
            
            return container
            
        except Exception as e:
            self.logger.error(f"Failed to launch engine: {e}")
            return None
    
    def _wait_for_server_ready(self, port: int, timeout: int = 300) -> bool:
        """Wait for the server to be ready to accept requests"""
        import requests
        
        self.logger.info("Waiting for engine to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    self.logger.info("Engine is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        
        self.logger.error("Timeout waiting for engine to be ready")
        return False
    
    def _cleanup_container(self, container: docker.models.containers.Container):
        """Clean up Docker container"""
        try:
            container.stop(timeout=20)
            container.remove()
            self.logger.info(f"Cleaned up container: {container.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup container {container.name}: {e}")
    
    def _run_benchmark_command(self, bench_args: List[str], run_id: str, scenario_name: str) -> bool:
        """Run the benchmark command using inference-benchmarker"""
        try:
            cmd = [BENCHMARK_TOOL_CMD]
            cmd.extend(['--no-console'])  # disable UI
            cmd.extend(bench_args)
            cmd.extend(['--run-id', run_id])
            
            # Add metadata
            metadata = f"engine={self.config['engine']['name']},scenario={scenario_name},autotune=true"
            cmd.extend(['--extra-meta', metadata])
            
            cmd = [str(c) for c in cmd]
            
            self.logger.info(f"Running benchmark: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=None,
                check=True
            )
            
            self.logger.info("Benchmark completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            return False
    
    def _find_latest_results_file(self, run_id: str) -> Optional[Path]:
        """Find the latest results file for a given run_id
        """
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('config', {}).get('run_id') == run_id:
                        return json_file
            except:
                continue
        
        return None
    
    def _run_throughput_benchmark(self) -> Optional[Dict]:
        """Run throughput benchmark to discover maximum throughput
        """
        self.logger.info("Running throughput benchmark...")
        
        scenario = self.config['scneario']
        port = self.config['port']
        
        # Build benchmark arguments for throughput test
        run_id = self._generate_run_id()
        bench_args = [
            '--url', f"http://localhost:{port}",
            '--benchmark-kind', "throughput",
            '--max-vus', str(scenario['max_vus']),
            '--duration', scenario['throughput_duration'],
            '--prompt-options', scenario['prompt_options'],
            '--decode-options', scenario['decode_options'],
            '--tokenizer-name', self.config['model'],
            '--output-path', self.results_dir.joinpath(Path(f"throughput_{run_id}.json")).as_posix()
        ]
        
        if scenario.get('dataset_file'):
            bench_args.extend(['--dataset-file', scenario['dataset_file']])
        
        # Run benchmark
        success = self._run_benchmark_command(bench_args, run_id, "throughput")
        
        if not success:
            return None
        
        # Find and load results
        time.sleep(2)  # Wait for file to be written
        results_file = self._find_latest_results_file(run_id)
        if results_file and results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        
        self.logger.error("Could not find results file")
        return None
    
    def _run_rate_benchmark(self, rate: float) -> Optional[Dict]:
        """Run rate benchmark with specific request rate"""
        self.logger.info(f"Running rate benchmark at {rate:.2f} req/s")
        
        scenario = self.config['scneario']
        port = self.config['port']
        
        # Build benchmark arguments for rate test
        run_id = self._generate_run_id()
        bench_args = [
            '--url', f"http://localhost:{port}",
            '--benchmark-kind', 'rate',
            '--max-vus', str(scenario['max_vus']),
            '--duration', str(scenario['rate_duration']),
            '--rates', str(rate),
            '--prompt-options', scenario['prompt_options'],
            '--decode-options', scenario['decode_options'],
            '--tokenizer-name', self.config['model'],
        ]
        
        if scenario.get('dataset_file'):
            bench_args.extend(['--dataset-file', scenario['dataset_file']])
        
        # Run benchmark
        success = self._run_benchmark_command(bench_args, run_id, "rate")
        
        if not success:
            return None
        
        # Find and load results
        time.sleep(2)  # Wait for file to be written
        results_file = self._find_latest_results_file(run_id)
        if results_file and results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        
        self.logger.error("Could not find results file")
        return None
    
    def _extract_metrics(self, results: Dict) -> Dict:
        """Extract key metrics from benchmark results"""
        throughput_result = None
        for result in results.get('results', []):
            if result.get('id') == 'throughput':
                throughput_result = result
                break
        
        if not throughput_result:
            # Try to find any result with metrics
            for result in results.get('results', []):
                if result.get('request_rate'):
                    throughput_result = result
                    break
        
        if not throughput_result:
            return {}
        
        metrics = {
            'throughput': throughput_result.get('request_rate', 0),
            'total_requests': throughput_result.get('total_requests', 0),
            'successful_requests': throughput_result.get('successful_requests', 0),
            'failed_requests': throughput_result.get('failed_requests', 0),
            'success_rate': throughput_result.get('successful_requests', 0) / max(throughput_result.get('total_requests', 1), 1),
        }
        
        # Extract latency metrics
        ttft = throughput_result.get('time_to_first_token_ms', {})
        e2e = throughput_result.get('e2e_latency_ms', {})
        itl = throughput_result.get('inter_token_latency_ms', {})
        
        metrics.update({
            'ttft_p99': ttft.get('p99', float('inf')),
            'ttft_avg': ttft.get('avg', float('inf')),
            'e2e_p99': e2e.get('p99', float('inf')),
            'e2e_avg': e2e.get('avg', float('inf')),
            'itl_p99': itl.get('p99', float('inf')),
            'itl_avg': itl.get('avg', float('inf')),
        })
        
        return metrics
    
    def _meets_goodput_thresholds(self, metrics: Dict) -> bool:
        """Check if metrics meet goodput thresholds"""
        thresholds = self.config['goodput_thresholds']
        
        checks = [
            metrics.get('e2e_p99', float('inf')) <= thresholds['max_e2e_latency_ms'],
            metrics.get('ttft_p99', float('inf')) <= thresholds['max_ttft_ms'],
            metrics.get('success_rate', 0) >= thresholds['min_success_rate'],
            metrics.get('itl_p99', float('inf')) <= thresholds['max_inter_token_latency_ms'],
        ]
        
        return all(checks)
    
    def _find_optimal_rate(self, max_throughput: float) -> Tuple[float, Dict]:
        """Find optimal rate that meets goodput thresholds"""
        # Start with 90% of max throughput and decrease until goodput is met
        rate = max_throughput * 0.9
        best_rate = 0
        best_metrics = {}
        
        attempts = 0
        max_attempts = 10
        
        while rate > 0.1 and attempts < max_attempts:
            attempts += 1
            
            # Sleep between rate tests to let any pending requests clear
            time.sleep(3)
            
            results = self._run_rate_benchmark(rate)
            if not results:
                rate *= 0.8
                continue
            
            metrics = self._extract_metrics(results)
            if not metrics:
                rate *= 0.8
                continue
            
            if self._meets_goodput_thresholds(metrics):
                best_rate = rate
                best_metrics = metrics
                self.logger.info(f"Found optimal rate: {rate:.2f} req/s")
                break
            
            # Decrease rate by 20%
            rate *= 0.8
        
        return best_rate, best_metrics
    
    def _generate_parameter_combinations(self) -> List[Dict]:
        """Generate all parameter combinations to test
        """
        engine_config = self.config['engine']
        value_args_pool = engine_config.get('value_args_pool', {})
        action_args_pool = engine_config.get('action_args_pool', {})
        
        # Get parameter names and values for each type
        value_param_names = list(value_args_pool.keys())
        value_param_values = [value_args_pool[name] for name in value_param_names]
        
        action_param_names = list(action_args_pool.keys())
        action_param_values = [action_args_pool[name] for name in action_param_names]
        
        # Generate all combinations
        combinations = []
        
        # Generate combinations for value params (or empty if none)
        value_combinations = list(product(*value_param_values)) if value_param_values else [()]
        action_combinations = list(product(*action_param_values)) if action_param_values else [()]
        
        for value_combo in value_combinations:
            for action_combo in action_combinations:
                combination = {
                    'value_args': dict(zip(value_param_names, value_combo)) if value_param_names else {},
                    'action_args': dict(zip(action_param_names, action_combo)) if action_param_names else {}
                }
                combinations.append(combination)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations to test.")
        return combinations
    
    def run_auto_tune(self) -> Dict:
        """Run the complete auto-tuning process"""
        self.logger.info("Starting auto-tune process...")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        all_results = []

        for i, param_config in enumerate(param_combinations, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[{i}/{len(param_combinations)}] Testing parameter combination: {param_config}")
            self.logger.info(f"{'='*60}")
            
            container = None
            try:
                # TODO: save logs to debug if needed
                container = self._launch_docker_engine(param_config)
                if not container:
                    continue
                
                if not self._wait_for_server_ready(self.config['port']):
                    self.logger.error("Server failed to start properly")
                    continue
                
                # Step 3: Run throughput benchmark
                throughput_results = self._run_throughput_benchmark()
                if not throughput_results:
                    self.logger.error("Failed to run throughput benchmark")
                    continue
                
                # Step 4: Extract metrics
                throughput_metrics = self._extract_metrics(throughput_results)
                if not throughput_metrics:
                    self.logger.error("Failed to extract throughput metrics")
                    continue
                
                self.logger.info(f"Max throughput: {throughput_metrics['throughput']:.2f} req/s")
                
                # Step 5: Check if goodput thresholds are already met
                if self._meets_goodput_thresholds(throughput_metrics):
                    self.logger.info("Goodput thresholds met at max throughput!")
                    final_metrics = throughput_metrics.copy()
                    final_metrics['optimal_rate'] = throughput_metrics['throughput']
                else:
                    self.logger.info("Goodput thresholds not met, finding optimal rate...")
                    # Print current metrics for transparency
                    self.logger.info(f"Current metrics - E2E P99: {throughput_metrics['e2e_p99']:.2f}ms, "
                                   f"TTFT P99: {throughput_metrics['ttft_p99']:.2f}ms, "
                                   f"ITL P99: {throughput_metrics['itl_p99']:.2f}ms, "
                                   f"Success Rate: {throughput_metrics['success_rate']:.3f}")
                    
                    # Step 6: Find optimal rate
                    optimal_rate, rate_metrics = self._find_optimal_rate(throughput_metrics['throughput'])
                    
                    if optimal_rate > 0:
                        self.logger.info(f"Found optimal rate: {optimal_rate:.2f} req/s")
                        final_metrics = rate_metrics.copy()
                        final_metrics['optimal_rate'] = optimal_rate
                    else:
                        self.logger.info("Could not find rate that meets goodput thresholds")
                        continue
                
                # Compile final result
                result = {
                    'param_config': param_config,
                    'max_throughput': throughput_metrics['throughput'],
                    'metrics': final_metrics,
                    'meets_thresholds': self._meets_goodput_thresholds(final_metrics)
                }
                
                all_results.append(result)
                
                # Update best configuration if this is better
                if result['meets_thresholds'] and final_metrics['throughput'] > self.best_throughput:
                    self.best_throughput = final_metrics['throughput']
                    self.best_config = result
                    self.logger.info(f"NEW BEST CONFIG! Throughput: {self.best_throughput:.2f} req/s")
                
            except Exception as e:
                self.logger.error(f"Error testing parameter config {param_config}: {e}")
                
            finally:
                # Always cleanup container
                if container:
                    self._cleanup_container(container)
        
        # Save all results
        results_file = self.results_dir / "auto_tune_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'config_file': self.config_path,
                'goodput_thresholds': self.config['goodput_thresholds'],
                'best_config': self.best_config,
                'all_results': all_results
            }, f, indent=2)
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("AUTO-TUNE COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Tested {len(param_combinations)} parameter combinations")
        self.logger.info(f"Results saved to: {results_file}")
        
        if self.best_config:
            self.logger.info(f"\nBEST CONFIGURATION:")
            self.logger.info(f"Parameters: {self.best_config['param_config']}")
            self.logger.info(f"Throughput: {self.best_throughput:.2f} req/s")
            self.logger.info(f"Optimal Rate: {self.best_config['metrics']['optimal_rate']:.2f} req/s")
            self.logger.info(f"Success Rate: {self.best_config['metrics']['success_rate']:.3f}")
            self.logger.info(f"E2E P99 Latency: {self.best_config['metrics']['e2e_p99']:.2f} ms")
            self.logger.info(f"TTFT P99: {self.best_config['metrics']['ttft_p99']:.2f} ms")
            self.logger.info(f"ITL P99: {self.best_config['metrics']['itl_p99']:.2f} ms")
        else:
            self.logger.info(f"\nNo configuration met the goodput thresholds!")
        
        return {
            'best_config': self.best_config,
            'all_results': all_results,
            'results_file': str(results_file)
        }


def main() -> None :
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    tuner = AutoTuner(args.config, args.result_dir)
    tuner.run_auto_tune()
