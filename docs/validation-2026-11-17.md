# First live validation: observations and confounds

Run date: 2026-11-17 America/Los_Angeles (raw timestamps are UTC on November 18).
This is an exploratory pilot, not a capacity certification or an optimization A/B.

## What was actually executed

- Existing RTX 5070, 12,227 MiB reported VRAM, driver 616.92, WSL2 Linux.
- Actual server and benchmark client version: vLLM 0.26.0.
- Existing Qwen2.5-1.5B-Instruct server: memory utilization 0.80, context limit 4096,
  `VLLM_USE_V2_MODEL_RUNNER=0`. No server tuning or deployment mutation was applied.
- A separate CPU-only Docker client used host networking to the Kubernetes Service.
- The cache contained model snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
  the existing server had no explicit revision argument. The revised manifests pin
  that snapshot, but those revised manifests were not deployed in this validation.
- 8 smoke requests plus 6 cells of 32 requests: **200 completed, 0 request failures**
  in these successful runs. Earlier failed setup attempts are described below.
- 226 metric snapshots across the seven successful runs; **0 scrape errors**.

The saved [summary](results/2026-11-17/summary.json) links conceptually to each run's
directory under [results](results/2026-11-17/). Each directory contains detailed
numeric request results, exact commands, status and gzip-compressed metric snapshots.
Synthetic generated text was omitted from the public copy for compactness; all
numeric fields remain. [Checksums](results/2026-11-17/checksums.json) identify the
original full local JSON and uncompressed metrics. Full raw local data remain in
`bench/raw/`. Environment metadata and client help/version are saved per batch;
trailing whitespace is normalized in the public help copies only.

## Observed pilot values

These are means from one small run per cell. Cache state differs between rows.
They must not be interpreted as a controlled rate comparison or reliable p99s.

| Input/output tokens | Offered req/s | Completed | Mean TTFT ms | Mean TPOT ms | Output tok/s | Prefix token hit ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / 128 | 1 | 32 | 22.92 | 6.89 | 124.70 | 21.48% |
| 128 / 128 | 4 | 32 | 20.03 | 6.77 | 458.34 | 87.50% |
| 2048 / 128 | 1 | 32 | 122.41 | 7.58 | 124.24 | 6.20% |
| 2048 / 128 | 4 | 32 | 30.16 | 7.41 | 459.13 | 99.22% |
| 128 / 512 | 1 | 32 | 21.31 | 6.79 | 461.53 | 87.50% |
| 128 / 512 | 4 | 32 | 22.03 | 7.20 | 1402.68 | 87.50% |

TTFT is client time to first output. TPOT is per-request average time per output
token after the first, not the ITL tail distribution. Offered rate is a configured
Poisson arrival parameter, not the measured completion rate. The last row completed
about 2.74 requests/s over the finite run including drain; it does not demonstrate
sustained capacity at 4 requests/s.

## Finding 1: repeated seeded prompts confounded the comparison

The same seed was reused for paired cells while the existing server reused
prefixes. The long-input test's hit ratio changed from 6.20% to 99.22%. Its mean
TTFT also changed from 122.41 ms to 30.16 ms. This is **not evidence that increasing
arrival rate improved latency**: the later test performed different amounts of
uncached input work, and run order/warmup also changed.

The useful finding is a flaw in the experimental comparison. Future baseline
runs must explicitly disable caching or independently verify cache reset, with
per-workload warmup and repeated/reordered runs. A cache-on study instead requires
defined cold/warm/shared-prefix groups. Changing seeds can reduce accidental reuse,
but does not by itself prove equivalent cache state or a causal performance effect.

The added `metrics_summary.py` computes interval deltas from raw counters and
reports hit ratios. It returns unknown for absent/nonfinite series or observed
counter decreases. It does not silently treat a reset as a valid measurement.

## Finding 2: large TTFT did not match measured scheduler queue time

The initial smoke had mean client TTFT **16,577.19 ms**, but mean TPOT **7.17 ms**.
For those eight requests the recorded scheduler queue sum was **0.000187427 s**
(about 0.0234 ms/request). Server TTFT averaged 16,573.89 ms and recorded prefill
averaged 3,242.01 ms. Therefore the observed scheduler-queue metric does not explain
the long first-output wait.

This does not rule out every kind of waiting or prove a particular compilation
event caused it. The phase metrics do not form an established complete partition
of TTFT in this investigation. Initialization/warmup, execution and metric-boundary
tracing remain follow-up work. No trace proving a sole cause was collected.

## Reliability and tooling validation

Before the successful runs the cluster intermittently reported no healthy GPU
devices and container availability failures. It later became usable while one
WSL session remained active. This was observed, not diagnosed as a WSL shutdown bug.

Real execution also found two client compatibility issues, both corrected:

1. Default CLI help is abbreviated; expand `--help=all` before validating flags.
2. In 0.26.0 a random range ratio of 0 fixes lengths; 1 is invalid. The rejected
   attempt produced no measured request result, and the failure was retained.

The wrapper pins the benchmark protocol version, captures a final metric snapshot,
and keeps failures even when no result JSON is emitted. **14 CPU tests pass**,
including partial HTTP success, timeouts, missing metrics, grouped help, the
version-specific ratio, final sampling and counter-reset handling. These tests are
separate from the 200 successful GPU requests. Revised manifest syntax/rollout and
the full 36-cell matrix have not been validated on the cluster.

## Claims this pilot does not support

No maximum capacity, stable tail-latency SLO, optimal batch budget, throughput
improvement, admission-queue bottleneck, mixed-workload isolation or multi-GPU
scaling result is claimed. The project now has an executable measurement path,
real evidence, and a concrete cache-state confound to explain and correct.
