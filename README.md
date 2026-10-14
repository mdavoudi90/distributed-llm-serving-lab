# Distributed LLM Serving Lab

A small, reproducible lab for learning LLM inference serving. Start with a vLLM endpoint and measure the latency/throughput tradeoff under bounded concurrency. Extend to NVIDIA Dynamo, Model Streamer, and Grove after establishing a baseline.

## What works today

- A dependency-free Python benchmark client for OpenAI-compatible chat endpoints.
- Warmup, bounded concurrency, request timeouts, error accounting, JSON results.
- End-to-end latency p50/p95/p99, successful requests/s, and output tokens/s when server usage is present.
- Tests using a local HTTP fixture; no GPU needed for client tests.

This is a learning lab, not a new inference engine. GPU performance results have not been collected yet. Dynamo, Grove, and Model Streamer integrations are planned, not implemented.

## Start a real model server

Use a Linux machine with a compatible GPU, driver, and sufficient memory. Install vLLM in its own environment following its current installation guide; the benchmark client uses Python 3.10+ and only the standard library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 2048
```

The model downloads on first launch. The local endpoint is bound to loopback. For a remote GPU, use an SSH tunnel rather than exposing an unauthenticated endpoint.

## Run the client

From this directory, in another terminal:

```bash
python benchmark.py --requests 50 --concurrency 1 --hardware 'YOUR GPU' --engine-version 'YOUR VLLM VERSION' --output results/c1.json
python benchmark.py --requests 50 --concurrency 4 --hardware 'YOUR GPU' --engine-version 'YOUR VLLM VERSION' --output results/c4.json
python benchmark.py --requests 50 --concurrency 8 --hardware 'YOUR GPU' --engine-version 'YOUR VLLM VERSION' --output results/c8.json
python -m unittest discover -s tests -v
```

Override `--base-url`, `--model`, `--prompt`, and `--max-tokens` as needed. Set `INFERENCE_API_KEY` only if your endpoint requires it. Credentials and prompt contents are not written into results.

## Interpret results carefully

This is a closed-loop concurrency experiment: each worker sends its next request after the previous one finishes. It does not simulate a fixed arrival rate. Latency includes HTTP and server processing after the worker starts, excluding client executor queue time. Failed requests are counted separately; percentiles use successful requests only. Measurement duration includes all requests, including failures.

Responses are non-streaming, so this client does **not** measure time to first token or inter-token latency. Output tokens come from server usage; missing counts are reported as null, never estimated from words. Repeated prompts may benefit from prefix caching. Record whether it is enabled. A small sample is insufficient for reliable p99 conclusions.

For token-level metrics and controlled synthetic token lengths, use vLLM's official `vllm bench serve` and its current documentation. Repeat runs and record software versions, model revision, precision, GPU, driver, launch flags, and workload alongside results. Do not compare results from different configurations as if they were equivalent.

## Learning roadmap

1. Baseline: concurrency vs end-to-end latency and throughput.
2. Add streaming metrics and controlled prompt-token workloads.
3. Compare prefix caching on/off with repeated and unique prompts.
4. Add Dynamo routing; compare equivalent backend/workload configurations.
5. Compare default loading vs Model Streamer cold starts.
6. Use Grove to deploy and scale coordinated workers on Kubernetes.

See [experiment template](docs/experiment-template.md) for a reproducible write-up.

## References

- [vLLM server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM serving benchmarks](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
- [NVIDIA Grove](https://github.com/ai-dynamo/grove)
- [Model Streamer](https://github.com/dsx-ai-factory/model-streamer)

## Contributing

Keep experiments small and reproducible. Include setup, validation, and limitations with changes. Publish only public models and workloads. Please open an issue describing the experiment before a large integration.
