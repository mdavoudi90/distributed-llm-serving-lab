"""Dependency-free, bounded-concurrency benchmark for OpenAI-compatible servers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import time
from urllib.request import Request, urlopen


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def request_one(base_url, model, prompt, max_tokens, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": False}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("INFERENCE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["INFERENCE_API_KEY"]
    req = Request(base_url.rstrip("/") + "/chat/completions",
                  data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            body = json.load(response)
        if not body.get("choices"):
            raise ValueError("Response has no choices")
        tokens = body.get("usage", {}).get("completion_tokens")
        return {"ok": True, "latency_s": time.perf_counter() - start,
                "completion_tokens": tokens}
    except Exception as exc:
        # Do not persist response bodies, prompts, or credentials.
        return {"ok": False, "latency_s": time.perf_counter() - start,
                "error_type": type(exc).__name__}


def summarize(rows, duration):
    successful = [r for r in rows if r["ok"]]
    latencies = [r["latency_s"] for r in successful]
    known_tokens = all(isinstance(r.get("completion_tokens"), int) for r in successful)
    return {"requests": len(rows), "succeeded": len(successful),
            "failed": len(rows) - len(successful), "duration_s": duration,
            "successful_requests_per_s": len(successful) / duration,
            "output_tokens_per_s": (sum(r["completion_tokens"] for r in successful) / duration
                                    if successful and known_tokens else None),
            "latency_s": {"p50": percentile(latencies, .5),
                          "p95": percentile(latencies, .95), "p99": percentile(latencies, .99)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--prompt", default="Explain continuous batching in three sentences.")
    p.add_argument("--hardware", default="unspecified")
    p.add_argument("--engine-version", default="unspecified")
    p.add_argument("--output", default="results/run.json")
    args = p.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.timeout) <= 0 or args.warmup < 0:
        p.error("requests, concurrency, max-tokens, timeout must be positive; warmup nonnegative")
    def run(_):
        return request_one(args.base_url, args.model, args.prompt, args.max_tokens, args.timeout)
    for i in range(args.warmup):
        if not run(i)["ok"]:
            p.exit(1, "Warmup failed; verify server and model before benchmarking.\n")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(run, range(args.requests)))
    summary = summarize(rows, time.perf_counter() - start)
    config = vars(args).copy()
    config.pop("prompt")
    config.pop("base_url")
    result = {"config": config, "python": platform.python_version(),
              "summary": summary, "requests": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
