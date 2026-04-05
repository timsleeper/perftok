"""Pydantic models for benchmark configuration, results, and reports."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    model: str
    url: str
    api_key: str | None = None
    concurrency: int = Field(default=1, gt=0)
    num_requests: int = Field(default=1, gt=0)
    mean_input_tokens: int = Field(default=550, gt=0)
    stddev_input_tokens: int = Field(default=150, ge=0)
    mean_output_tokens: int = Field(default=150, gt=0)
    stddev_output_tokens: int = Field(default=10, ge=0)
    timeout: int = Field(default=300, gt=0)
    streaming: bool = True

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        v = v.rstrip("/")
        if v.endswith("/v1"):
            v = v[:-3]
        return v


class RequestResult(BaseModel):
    """Result of a single benchmark request."""

    success: bool
    ttft_ms: float | None = None
    e2e_latency_ms: float = 0.0
    output_tokens: int = 0
    inter_token_latencies_ms: list[float] = Field(default_factory=list)
    error: str | None = None

    @property
    def output_token_throughput(self) -> float:
        """Output tokens per second for this request."""
        latency_s = self.e2e_latency_ms / 1000.0
        if latency_s <= 0:
            return 0.0
        return self.output_tokens / latency_s


class LatencyStats(BaseModel):
    """Percentile statistics for a latency distribution."""

    mean: float
    stddev: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float


class BenchmarkReport(BaseModel):
    """Aggregated benchmark report."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    ttft_stats: LatencyStats | None = None
    itl_stats: LatencyStats | None = None
    e2e_latency_stats: LatencyStats | None = None
    output_throughput_per_request_stats: LatencyStats | None = None
    output_token_throughput: float = 0.0
    request_throughput: float = 0.0
    error_rate: float = 0.0
