# tokenflow

Lightweight benchmarking tool for OpenAI-compatible LLM endpoints.

tokenflow gives you the same metrics as [aiperf](https://github.com/NVIDIA/aiperf) (TTFT, ITL, throughput, latency percentiles) with a simple pip-installable CLI — no ZMQ services, no Ray dependency, no special binaries.

## Installation

```bash
pip install tokenflow
```

Or from source:

```bash
git clone https://github.com/timsleeper/tokenflow.git
cd tokenflow
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```bash
# Benchmark a local vLLM / TGI / Ollama endpoint
tokenflow run \
  --url http://localhost:8000 \
  --num-requests 100 \
  --concurrency 10

# Specify model explicitly
tokenflow run \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --url http://localhost:8000 \
  --num-requests 100 \
  --concurrency 10

# Hosted endpoint with API key
tokenflow run \
  --url https://api.example.com \
  --api-key $API_KEY \
  --num-requests 50 \
  --concurrency 5 \
  --output-format json \
  --output-file results.json
```

If `--model` is omitted, tokenflow queries `/v1/models` and prompts you to confirm.

## CLI Options

```
tokenflow run [OPTIONS]

  --model TEXT                    Model name (auto-discovered if omitted)
  --url TEXT                      Base URL of the endpoint [required]
  --api-key TEXT                  API key (or set API_KEY env var)
  --concurrency INTEGER           Max concurrent requests [default: 1]
  --num-requests INTEGER          Total number of requests [default: 1]
  --mean-input-tokens INTEGER     Mean input prompt tokens [default: 550]
  --stddev-input-tokens INTEGER   Stddev of input tokens [default: 150]
  --mean-output-tokens INTEGER    Mean output tokens [default: 150]
  --stddev-output-tokens INTEGER  Stddev of output tokens [default: 10]
  --timeout INTEGER               Request timeout in seconds [default: 300]
  --streaming / --no-streaming    Use streaming SSE responses [default: streaming]
  --output-format [table|json|csv]  Output format [default: table]
  --output-file TEXT              Write results to file
```

## Metrics

| Metric | Unit | Aggregation |
|--------|------|-------------|
| TTFT (Time to First Token) | ms | p50 / p75 / p90 / p95 / p99 / min / max / mean / stddev |
| ITL (Inter-Token Latency) | ms | same |
| E2E Request Latency | ms | same |
| Output Token Throughput | tokens/s | system-wide aggregate |
| Output Throughput Per Request | tokens/s | per-request with percentiles |
| Request Throughput | req/s | aggregate |
| Error Rate | % | aggregate |

## Output Formats

**Table** (default) — Rich-formatted summary printed to the terminal.

**JSON** — Full report as structured JSON, suitable for programmatic consumption.

**CSV** — Flattened single-row CSV with all metrics, handy for appending to spreadsheets.

Use `--output-file` to write results to disk in any format.

## TLS/SSL

tokenflow handles self-signed certificates gracefully. If TLS verification fails, it emits a warning and proceeds without verification — no `--insecure` flag needed. This is common with local inference servers behind self-signed certs.

## Architecture

- **Concurrency**: `asyncio` + `aiohttp` + `Semaphore` — no threads, no Ray, no ZMQ
- **Token counting**: `tiktoken` (cl100k_base) for exact prompt-length targeting
- **Models**: Pydantic v2 for config validation and report serialization
- **CLI**: Click with a `run` subcommand
- **Output**: Rich tables, JSON, CSV

## Development

```bash
pip install -e ".[dev]"
pytest -v              # 76 tests
ruff check src/ tests/ # lint
```

## License

Apache 2.0
