# Experiment protocol

## Baseline

Capture image digest, actual client/server versions, model/tokenizer snapshot,
GPU/driver/WSL, complete server args, limits and network path. Record effective
prefix-cache and chunked-prefill settings. Verify health, models, one completion
and metrics; warm up separately. Smoke requests validate the pipeline, not tails.

The matrix compares 128/128, 2048/128 and 128/512 input/output token lengths,
initial rates 0.5/1/2/4 requests/s, 200 requests and three repeats per cell. These
are proposed settings, not capacity results. Pilot first and refine near the knee.
Increase samples for meaningful tails. Fixed output lengths ignore EOS; verify
actual returned lengths. Random token traffic does not measure answer quality.

The protocol pins the observed vLLM 0.26.0 benchmark client. Its length range
ratio is 0 for fixed lengths, not 1 (which is invalid in that version). Preserve
the full CLI help and verify semantics when changing benchmark versions.

Use matching seeds across configuration comparisons. Disable prefix caching for
baseline isolation or explicitly account for its state. The official client's
finite-rate arrival process is recorded by its version/help. The wrapper adds no
concurrency cap. Completed throughput is not evidence of achieved arrival rate.

The [first pilot](validation-2026-11-17.md) demonstrated why this cache control is
mandatory: paired seeds reused cached prefixes and changed hit ratios substantially.
Do not interpret another default-state run as a clean baseline. Inspect interval
hits/queries with `python bench/metrics_summary.py path/to/metrics.jsonl`; first
disable caching or verify cache clearing, warm each workload shape, and repeat
with reordered cells. Cache reset behavior is version-specific and was not tested
in the pilot.

## Controlled scheduling comparison

Choose a nontrivial rate from the pilot. With compatible chunked prefill enabled,
compare token budgets such as 512/2048/4096 while holding all other settings fixed.
Use separate output batches with their complete metadata. Documentation gives a
hypothesis about latency tradeoffs, not a guaranteed result on this GPU.

Start with separate workload families. A later 80/20 short/long mix must retain
request class labels and report each class separately. **Mixed workload generation
is not implemented in this wrapper.** Do not claim mixed-workload findings from
separate-family runs.

## Cache pressure and concurrency

Hold workload, arrival rate, memory allocation and token budget fixed, then vary
the server's sequence limit using settings valid for the installed version. Watch
active/waiting requests, KV utilization and preemption/recompute signals with
latency and throughput. If no cache pressure occurs, report it; do not invent a
cache-exhaustion incident. A later prefix-cache study needs shared-prefix, cold,
warm and nonsharing controls; the random matrix does not implement this study.

## Interpretation rules

- Client TTFT includes network and request processing. Server phase boundaries
  differ. Never subtract unrelated percentiles to reconstruct queue time.
- TTFT, end-to-end latency, per-request mean time per output token (TPOT), and
  inter-token latency (ITL) are different. A streaming event may hold many tokens.
- Report successful latency alongside errors/timeouts. Dropping failed requests
  can make successful-request tails misleadingly good.
- Preserve actual metric HELP/TYPE and names. Missing is unknown, not zero.
  Cumulative histograms need time-window deltas/rates, not just the final snapshot.
- Queue and TTFT growth suggest a hypothesis. Check prefill/decode, workload
  stability and client health, then intervene. Correlation is not exclusive cause.
- Check client CPU and compare a less constrained network/client path before
  claiming capacity. Port-forward and a colocated client can limit the result.
- On failures the runner aborts. Ensure the server drains before restarting;
  cooldown alone cannot prove abandoned requests have stopped computing.
- Define experimental SLOs before goodput analysis; they are not historical
  business SLAs. The exporter does not derive goodput without validated per-request
  field alignment. Failed runs stay visible; missing aggregate fields stay empty.

Required conclusion: under [environment/workload], changing [one setting]
changed [measurement] from [baseline] to [result], across [samples/repeats], while
[tradeoff]. [Evidence] supports [limited conclusion]; [alternatives] remain.
Only fill these fields after measurements exist, and attach raw data.
