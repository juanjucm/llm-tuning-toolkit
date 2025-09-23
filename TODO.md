
## TODOs
- Check correlation with [vllm auto-tunning tool](https://github.com/vllm-project/vllm/tree/main/benchmarks/auto_tune)
- Delegate logs to verbose mode.
- Add final `sweep` benchmark step after auto-tuning is finished. Test best config for that engine and save results for later comparison.
    - Control this through CLI arg.
- Add support for multi-engine tunning.
- Add plotting functionallity to `dashboard` tool:
    - Auto-tunning process -> throughput plots based on tunable params.
    - Plot final engine config run.
- Add support for testing ad-hoc functionallity configurable in yaml.
    - reasoning testing.
    - tool calling testing.
    - json output testing.
- Finish polishing `multi-benchmarking` tool

### Nice to have
- Module for initial configuration discovery
    - memory estimation.
    - suggested parameters to include in pool.
    - etc.
