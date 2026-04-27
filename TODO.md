
## TODOs
- Remove need of most of base-args from auto-tuning config yamls. Most of them such as max-model-len are already defined in the scenario config. If user include any, use it. If not, pass defaults / let the engine default.
- Add final `sweep` benchmark step after auto-tuning is finished. Test best config for that engine and save results for later comparison.
    - Control this through CLI arg.
- Add support for multi-engine tunning.
- Add plotting functionallity to `dashboard` tool:
    - Auto-tunning process -> SLOs plots based on tunable params.
    - Plot final engine config run.
- Add support for testing ad-hoc functionallity configurable in yaml.
    - reasoning testing.
    - tool calling testing.
    - json output testing.
- Finish polishing `multi-benchmarking` tool
