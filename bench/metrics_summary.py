"""Summarize observed cumulative metric deltas; never subtract percentiles."""
import argparse
import json
import math
import re
from pathlib import Path

STAGES = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "server_ttft": "vllm:time_to_first_token_seconds",
}


def total(text, metric):
    pattern = re.compile(r"^" + re.escape(metric) + r'(?:\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
    values = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            value = float(match[1])
            if not math.isfinite(value):
                return None
            values.append(value)
    return sum(values) if values else None


def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    valid = [r for r in rows if "prometheus_text" in r]
    out = {"snapshots": len(rows), "scrape_errors": len(rows) - len(valid),
           "scope": "aggregate across exposed label sets; this lab serves one model/engine"}

    def delta(metric):
        values = [total(row["prometheus_text"], metric) for row in valid]
        if len(values) < 2 or any(v is None for v in values):
            return None
        if any(b < a for a, b in zip(values, values[1:])):
            return None  # Observed reset/discontinuity: do not fabricate a rate.
        return values[-1] - values[0]

    for stage, name in STAGES.items():
        seconds, count = delta(name + "_sum"), delta(name + "_count")
        out[stage + "_seconds"] = seconds
        out[stage + "_samples"] = count
        out[stage + "_mean_ms"] = 1000 * seconds / count if seconds is not None and count else None
    hits = delta("vllm:prefix_cache_hits_total")
    queries = delta("vllm:prefix_cache_queries_total")
    out.update(prefix_hits=hits, prefix_queries=queries,
               prefix_hit_ratio=hits / queries if hits is not None and queries else None)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics_jsonl), indent=2))
