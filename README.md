# Single-GPU LLM Serving: Reliability and Performance Experiments

A Kubernetes lab for understanding initialization, request latency, throughput,
and GPU memory when serving Qwen with vLLM. The goal is to make configuration
decisions from reproducible evidence. This is a single-node learning project,
not a production service, a custom inference engine, or a multi-GPU deployment.

## Evidence and scope

| Area | What exists | Boundary |
| --- | --- | --- |
| Deployment | Kubernetes Deployment/Service, GPU request, three health probes | Single-node WSL2/k3s lab |
| Failure analysis | Historical build notes, exit-state fragment and earlier screenshots | Full logs/events need recapturing for exact causal/timing claims |
| Memory accounting | Historical allocation numbers and model-based KV calculation | Capacity estimate, not measured serving concurrency |
| Benchmark tooling | Executable CLI wrapper, metric capture, CSV export; 200 real requests completed | CPU tests and GPU observations are documented separately |
| Performance study | First pilot exposed prefix-cache contamination in paired comparisons | No capacity or tuning-improvement claim; controls need correction |
| Prometheus/Grafana | Raw vLLM `/metrics` capture in the wrapper | No provisioned monitoring stack or dashboard in this checkout |
| Helm/autoscaling | Not implemented here | YAML does not demonstrate a Helm chart or scale-out |

See the [evidence ledger](docs/evidence-ledger.md). Empty results are not findings.

The [first live report](docs/validation-2026-11-17.md) includes raw numeric results
and metric snapshots. Its main finding is that repeated seeded prompts changed
cache hit ratios, making apparent latency improvements an invalid rate comparison.

## Research question

How do input length, output length and arrival rate affect time to first token
(TTFT), generation latency and completed throughput on one GPU? When latency
rises, what distinguishes waiting, prefill, decode and KV-cache pressure?
After changing one setting, which requests benefit and what gets worse?

The [experiment protocol](docs/experiments.md) specifies controls and decision rules.

## Architecture

```text
official vLLM benchmark client --streaming completions--> Service --> vLLM Pod
                                                                    | GPU
                                                                    | Qwen + KV cache
                                                                    + /metrics
                                                                        |
run_matrix.py --> exact commands + metadata + client JSON + raw metric timeline
summarize.py  --> reported aggregates in CSV; missing values remain missing
```

Keep the benchmark client separate from the resource-constrained serving process.
Port forwarding is convenient for smoke tests; evaluate its overhead before
interpreting capacity measurements.

## Deployment

Historical environment: RTX 5070 (12 GiB), WSL2, k3s with Docker and NVIDIA device
plugin, `Qwen/Qwen2.5-1.5B-Instruct`, context limit 4096. The manifest pins the
image digest found in the existing deployment. Record the actual vLLM version
inside it; an immutable artifact is not a guarantee of compatibility.

Prerequisites: working GPU container access and a healthy Kubernetes device
plugin. The hostPath `/home/liucz/.cache/huggingface` is machine-specific; update
it for another host. HostPath does not itself constrain Pod placement.

```bash
kubectl get nodes
kubectl describe node
kubectl apply --dry-run=client -f manifests/vllm.yaml
kubectl apply -f manifests/vllm.yaml
kubectl rollout status deployment/vllm --timeout=600s
kubectl port-forward svc/llm-inference 8000:8000
# In another terminal:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

The historical `VLLM_USE_V2_MODEL_RUNNER=0` workaround is retained as an
environment-specific setting. Old notes report a UVA initialization failure;
they do not establish a universal WSL2 limitation or a verified release boundary.
Validate it against the pinned image and retain the logs.

## Benchmark workflow

Python 3.10+ runs the wrapper and tests. Execution additionally needs the
official `vllm bench serve` CLI and its tokenizer dependencies in the client
environment. The matrices target the observed vLLM 0.26.0 client. The wrapper
checks its version and full advertised flags before sending traffic. In this
version, random range ratio 0 fixes lengths; 1 is invalid. Revalidate semantics
before changing versions instead of assuming the same flag keeps its meaning.

```bash
python -m unittest discover -s tests -v
# Plan only: no traffic.
python bench/run_matrix.py --config bench/smoke.json

# Replace every example field with actual observations.
cp bench/environment.example.json environment.local.json

# After readiness and a separate warmup:
python bench/run_matrix.py --config bench/smoke.json \
  --environment environment.local.json --tokenizer /path/to/matching/snapshot --execute
# Inspect smoke evidence before the longer sweep.
python bench/run_matrix.py --config bench/matrix.json \
  --environment environment.local.json --tokenizer /path/to/matching/snapshot --execute
python bench/summarize.py bench/raw --output summary.csv
```

Eight smoke requests validate the pipeline, not p99 or capacity. The full matrix
has 36 runs; pilot first, adjust rates around the knee and increase sample size
for tail estimates. Results go to new timestamped directories. Commands, metadata,
CLI help, detailed results, logs, metric snapshots and completion status are saved.
Nonzero exits, timeouts or incomplete request counts stop the sweep with evidence
preserved. Missing fields stay empty; failed result records stay in the CSV.

## Reliability lessons

### Initialization differs from runtime health

Old notes record about 163 seconds to application startup and a separate
185-second Pod-ready observation. The difference cannot be assigned entirely to
probe polling without a timeline. Startup gates the other probes, readiness
controls traffic eligibility, and liveness can trigger a restart.

A 10-second period and 30 failures provide an approximately 300-second startup
budget, which must be revisited for a 400-second start. Timing also depends on
scheduling, timeouts and termination behavior. Exit code 137 alone does not prove
who sent SIGKILL or why: correlate events, termination reason and previous logs.
Running and Ready are different states; neither alone explains restart backoff.

### An exclusive GPU changes rollout choices

A replacement cannot use the only GPU while the old Pod retains its allocation.
`Recreate` releases the old allocation first, accepting startup unavailability.
This project has no measured zero-downtime rollout. GPU sharing and alternative
rollout policies exist, but were not evaluated here.

### Separate application failure from probe failure

The old log records `VLLM_PORT=tcp://...` where vLLM expected an integer. A
same-namespace Service named `vllm` can introduce that name through service links.
The manifest disables service links and uses `llm-inference`. A failed health
check may be a symptom of an application that never started, not its cause.

## Memory model

For a conventional attention KV cache:

```text
bytes per cached token = 2 * layers * KV heads * head dimension * bytes per element
```

The historical Qwen configuration (28 layers, 2 KV heads, dimension 128, bf16)
gives 28,672 bytes = 28 KiB/token. Its reported 211,120 tokens correspond to about
5.64 GiB. Check the actual model revision before reusing these inputs.

`211120 / 4096 = 51.54` is a capacity ratio: at most 51 complete 4096-token
sequences under a simplified budget. It is not a promise of 51 requests meeting
a latency objective. Allocation granularity, active lengths, sharing, scheduler
limits and non-cache resources matter. Measure serving behavior separately.

## Limitations

- One GPU does not validate horizontal scaling, routing or prefill/decode
  disaggregation. Those remain design discussions.
- Slow cold starts may motivate warm capacity for short bursts. They do not
  prove HPA is categorically unsuitable; it can use custom metrics.
- GPU memory reserved at startup differs from KV blocks used by active requests.
  Utilization or memory occupancy alone cannot diagnose a bottleneck.
- A 60-second termination grace period is not proof of successful stream draining.
- Synthetic token workloads do not measure model answer quality or customer impact.

## Explain and verify the work

- [Chinese interview walkthrough](docs/interview-guide.zh-CN.md)
- [Experiment protocol](docs/experiments.md)
- [Live validation and measured limitations](docs/validation-2026-11-17.md)
- [Evidence ledger](docs/evidence-ledger.md)
- [Historical build notes](docs/build-log.md)

References: [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/),
[PagedAttention](https://arxiv.org/abs/2309.06180),
[vLLM tuning](https://docs.vllm.ai/en/stable/configuration/optimization/),
[metrics](https://docs.vllm.ai/en/stable/usage/metrics/),
[benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/).
Match documentation to the installed version.
