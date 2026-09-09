"""Typed configuration and model-aware selection for benchmark suites."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import yaml
from guidellm.benchmark.schemas.profiles import ProfileArgs
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from auto_tune.guidellm import metric_names

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelProfile(BaseModel):
    """Caller-supplied model traits used to select safe benchmark cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    tokenizer_model: str = Field(min_length=1)
    context_window: PositiveInt | None = None
    capabilities: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("capabilities", mode="before")
    @classmethod
    def normalize_capabilities(cls, value: object) -> object:
        if value is None:
            return frozenset()
        if isinstance(value, (list, tuple, set, frozenset)):
            return frozenset(str(item).strip().lower() for item in value if str(item).strip())
        return value


class BenchmarkSelector(BaseModel):
    """Conditions under which a benchmark is applicable to a model."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_patterns: list[str] = Field(default_factory=list)
    excluded_model_patterns: list[str] = Field(default_factory=list)
    min_context_window: PositiveInt | None = None
    max_context_window: PositiveInt | None = None
    capabilities: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("model_patterns", "excluded_model_patterns")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        if any(not pattern.strip() for pattern in patterns):
            raise ValueError("model patterns may not be empty")
        return patterns

    @field_validator("capabilities", mode="before")
    @classmethod
    def normalize_capabilities(cls, value: object) -> object:
        return ModelProfile.normalize_capabilities(value)

    @model_validator(mode="after")
    def validate_context_range(self) -> BenchmarkSelector:
        if (
            self.min_context_window is not None
            and self.max_context_window is not None
            and self.min_context_window > self.max_context_window
        ):
            raise ValueError("min_context_window may not exceed max_context_window")
        return self

    def skip_reason(self, profile: ModelProfile) -> str | None:
        if not self.enabled:
            return "disabled by suite configuration"
        model = profile.model.lower()
        if self.model_patterns and not any(
            fnmatch.fnmatchcase(model, pattern.lower()) for pattern in self.model_patterns
        ):
            return f"model '{profile.model}' does not match the allowed model patterns"
        if any(fnmatch.fnmatchcase(model, pattern.lower()) for pattern in self.excluded_model_patterns):
            return f"model '{profile.model}' matches an excluded model pattern"
        if self.min_context_window is not None:
            if profile.context_window is None:
                return f"requires context_window >= {self.min_context_window}; caller did not provide it"
            if profile.context_window < self.min_context_window:
                return f"requires context_window >= {self.min_context_window}"
        if self.max_context_window is not None:
            if profile.context_window is None:
                return f"requires context_window <= {self.max_context_window}; caller did not provide it"
            if profile.context_window > self.max_context_window:
                return f"requires context_window <= {self.max_context_window}"
        missing = self.capabilities - profile.capabilities
        if missing:
            return f"requires model capabilities: {', '.join(sorted(missing))}"
        return None


class BenchmarkCase(BaseModel):
    """One independently executable GuideLLM workload."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    tags: frozenset[str] = Field(default_factory=frozenset)
    selection: BenchmarkSelector = Field(default_factory=BenchmarkSelector)
    data: list[dict[str, Any] | str] = Field(min_length=1)
    profile: dict[str, Any]
    constraints: list[dict[str, Any] | str] = Field(min_length=1)
    slos: dict[str, float] | None = None
    backend: dict[str, Any] = Field(default_factory=dict)
    guidellm_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("must use letters, digits, '.', '_' or '-'")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        return ModelProfile.normalize_capabilities(value)

    @model_validator(mode="after")
    def validate_guidellm_contract(self) -> BenchmarkCase:
        try:
            ProfileArgs.model_validate(self.profile)
        except ValueError as error:
            raise ValueError(f"invalid GuideLLM profile: {error}") from error
        if self.slos:
            supported = metric_names()
            for name in self.slos:
                if not name.startswith(("min_", "max_")):
                    raise ValueError(f"SLO '{name}' must start with min_ or max_")
                if name[4:] not in supported:
                    raise ValueError(f"SLO '{name}' refers to unsupported metric '{name[4:]}'")
        arguments = self.guidellm_options.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("guidellm_options.arguments must be a mapping")
        if "constraint" in arguments or "constraints" in arguments:
            raise ValueError("use constraints, not guidellm_options.arguments.constraint")
        forbidden_backend_fields = {"target", "model", "api_key"} & self.backend.keys()
        if forbidden_backend_fields:
            fields = ", ".join(sorted(forbidden_backend_fields))
            raise ValueError(f"backend fields are runtime-supplied and may not appear in the suite: {fields}")
        return self


class BenchmarkSuiteConfig(BaseModel):
    """A reusable collection of benchmark cases with no deployment state."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    backend: dict[str, Any] = Field(default_factory=lambda: {"kind": "openai_http"})
    guidellm_options: dict[str, Any] = Field(default_factory=lambda: {"sample_size": 0})
    benchmarks: list[BenchmarkCase] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return BenchmarkCase.validate_name(value)

    @model_validator(mode="after")
    def validate_suite(self) -> BenchmarkSuiteConfig:
        names = [benchmark.name for benchmark in self.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("benchmark names must be unique")
        forbidden_backend_fields = {"target", "model", "api_key"} & self.backend.keys()
        if forbidden_backend_fields:
            fields = ", ".join(sorted(forbidden_backend_fields))
            raise ValueError(f"backend fields are runtime-supplied and may not appear in the suite: {fields}")
        arguments = self.guidellm_options.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("guidellm_options.arguments must be a mapping")
        if "constraint" in arguments or "constraints" in arguments:
            raise ValueError("use benchmark constraints, not guidellm_options.arguments.constraint")
        return self


def load_suite(path: str | Path) -> BenchmarkSuiteConfig:
    """Load and validate one benchmark-only YAML file."""
    with Path(path).open() as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("The benchmark suite file must contain a YAML mapping")
    return BenchmarkSuiteConfig.model_validate(loaded)


def select_benchmarks(
    suite: BenchmarkSuiteConfig,
    profile: ModelProfile,
    *,
    benchmark_names: frozenset[str] = frozenset(),
    include_tags: frozenset[str] = frozenset(),
    exclude_tags: frozenset[str] = frozenset(),
) -> list[tuple[BenchmarkCase, str | None]]:
    """Return every case with either no skip reason or an explicit one."""
    known_names = {benchmark.name for benchmark in suite.benchmarks}
    unknown_names = benchmark_names - known_names
    if unknown_names:
        raise ValueError(f"Unknown benchmark names: {', '.join(sorted(unknown_names))}")

    selections: list[tuple[BenchmarkCase, str | None]] = []
    for benchmark in suite.benchmarks:
        if benchmark_names and benchmark.name not in benchmark_names:
            reason = "not selected by caller"
        elif include_tags and not (benchmark.tags & include_tags):
            reason = "does not match caller include tags"
        else:
            excluded = benchmark.tags & exclude_tags
            if excluded:
                reason = f"excluded by caller tags: {', '.join(sorted(excluded))}"
            else:
                reason = benchmark.selection.skip_reason(profile)
        selections.append((benchmark, reason))
    return selections
