# perftok

Lightweight benchmarking tool for OpenAI-compatible LLM endpoints.

perftok gives you the same metrics as [aiperf](https://github.com/NVIDIA/aiperf) (TTFT, ITL, throughput, latency percentiles) with a simple pip-installable CLI — no ZMQ services, no Ray dependency, no special binaries.

## Installation

```bash
pip install perftok
```

Or from source:

```bash
git clone https://github.com/timsleeper/perftok.git
cd perftok
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```bash
# Benchmark a local vLLM / TGI / Ollama endpoint
perftok \
  --url http://localhost:8000 \
  --num-requests 100 \
  --concurrency 10

# Specify model explicitly
perftok \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --url http://localhost:8000 \
  --num-requests 100 \
  --concurrency 10

# Hosted endpoint with API key
perftok \
  --url https://api.example.com \
  --api-key $API_KEY \
  --num-requests 50 \
  --concurrency 5 \
  --output-format json \
  --output-file results.json
```

If `--model` is omitted, perftok queries `/v1/models` and prompts you to confirm.

## CLI Options

```
perftok [OPTIONS]

  --model TEXT                    Model name (auto-discovered if omitted)
  --url TEXT                      Base URL of the endpoint [required]
  --api-key TEXT                  API key (or set API_KEY env var)
  --concurrency INTEGER           Max concurrent requests [default: 1]
  --num-requests INTEGER          Total number of requests [default: 1]
  --mean-input-tokens INTEGER     Mean input prompt tokens [default: 550]
  --stddev-input-tokens INTEGER   Stddev of input tokens [default: 150]
  --mean-output-tokens INTEGER    Mean output tokens [default: 150]
  --stddev-output-tokens INTEGER  Stddev of output tokens [default: 10]
  --warmup-requests INTEGER       Requests to send and discard before measuring [default: 0]
  --timeout INTEGER               Request timeout in seconds [default: 300]
  --streaming / --no-streaming    Use streaming SSE responses [default: streaming]
  --ignore-eos                    Force exactly max_tokens output (vLLM/SGLang/TensorRT-LLM)
  --random-seed INTEGER           Seed for prompt and length sampling
  --insecure                      Skip TLS/SSL certificate verification
  --output-format [table|json|csv]  Output format [default: table]
  --output-file TEXT              Write results to file
```

## Output Length

Most models stop at end-of-sequence before reaching `max_tokens`, so a run configured for 150 output tokens may really measure much shorter responses. perftok reports the mean actual versus requested output length and counts requests that stray more than 5% (capped at 50 tokens) from the request. On vLLM, SGLang, and TensorRT-LLM, pass `--ignore-eos` to force exactly `max_tokens` tokens per request. Hosted OpenAI-style endpoints reject that parameter, so it is off by default.

## Metrics

| Metric | Unit | Aggregation |
|--------|------|-------------|
| TTFT (Time to First Token) | ms | p50 / p75 / p90 / p95 / p99 / min / max / mean / stddev |
| ITL (Inter-Token Latency): (latency - TTFT) / (output tokens - 1), per request | ms | same |
| ICL (Inter-Chunk Latency): gap between consecutive SSE chunks | ms | same |
| E2E Request Latency | ms | same |
| Output Token Throughput | tokens/s | system-wide aggregate |
| Output Throughput Per Request: output tokens / E2E latency | tokens/s | per-request with percentiles |
| Output Throughput Per User: 1 / ITL | tokens/s | per-request with percentiles |
| Request Throughput | req/s | aggregate |
| Error Rate | % | aggregate |
| Output Tokens Per Request (actual / requested) | tokens | mean |
| Output Length Mismatches | count, % | aggregate |

Output tokens come from the server's `usage.completion_tokens` when it reports usage in the stream, and are otherwise counted with the tokenizer. ITL and throughput definitions match [aiperf's metrics reference](https://github.com/ai-dynamo/aiperf/blob/main/docs/metrics-reference.md).

## Output Formats

**Table** (default) — Rich-formatted summary printed to the terminal.

**JSON** — Full report as structured JSON, suitable for programmatic consumption.

**CSV** — Flattened single-row CSV with all metrics, handy for appending to spreadsheets.

Use `--output-file` to write results to disk in any format.

## TLS/SSL

If TLS certificate verification fails, perftok exits with a clear error message. Pass `--insecure` to bypass verification — common with local inference servers behind self-signed certs.

## Architecture

- **Concurrency**: `asyncio` + `aiohttp` + `Semaphore` — no threads, no Ray, no ZMQ
- **Token counting**: `tiktoken` (cl100k_base) for exact prompt-length targeting
- **Models**: Pydantic v2 for config validation and report serialization
- **CLI**: Click single-command interface
- **Output**: Rich tables, JSON, CSV

## Development

```bash
pip install -e ".[dev]"
pytest -v              # 130 tests
ruff check src/ tests/ # lint
```

## License

Apache 2.0
