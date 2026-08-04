# LLM Inference on Kubernetes

Deploying and operating a vLLM inference service on Kubernetes — with the
observability, failure modes, and capacity limits that make GPU workloads
different from ordinary stateless services.

> **Why this exists.** I have production experience operating Kubernetes
> platforms, but not GPU-backed inference workloads. This project is a
> deliberately constrained environment built to surface the differences and
> reason about them, not a production system.

---

## Architecture

| Component | Role |
| --- | --- |
| vLLM server | OpenAI-compatible inference API; exposes `/metrics` |
| Kubernetes Deployment | Single replica, `nvidia.com/gpu: 1` (whole-GPU claim) |
| Service | ClusterIP fronting the inference pod |
| Prometheus | Scrapes the vLLM metrics endpoint |
| Grafana | Latency, queue depth, and KV-cache dashboards |
| Load generator | Open-loop request-rate benchmarking client |

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` — bfloat16, 2.98 GiB of weights,
4096-token context.

**GPU:** RTX 5070, 12 GiB VRAM. Roughly 0.9 GiB is permanently held by the
Windows desktop compositor, so `gpu_memory_utilization` is set to **0.80**
rather than the 0.90 default — at 0.90 the engine competes with the display
driver for memory it does not actually have.

### Memory budget

At `gpu_memory_utilization=0.80` vLLM reserves 9.55 GiB and reports the split:

| Allocation | Size |
| --- | --- |
| Model weights | 2.98 GiB |
| Peak activation | 0.55 GiB |
| CUDA graph pool | 0.20 GiB |
| Non-torch overhead | 0.06 GiB |
| **KV cache** | **5.64 GiB** |

That 5.64 GiB becomes **211,120 tokens** of KV cache — 51.5 concurrent
requests at the full 4096-token context, and proportionally more for the
shorter prompts that dominate real traffic.

Dividing gives roughly **28 KB per token**, which is exactly what the model
architecture predicts:

```
2 (K and V) x 28 layers x 2 KV heads x 128 head_dim x 2 bytes (bf16) = 28,672 B
```

The KV-head count is what matters here, not the attention-head count. Qwen2.5
uses grouped-query attention — 12 query heads share 2 KV heads — so the cache
is 6x smaller than an equivalent multi-head model would need. KV-cache cost
tracks layer count and KV heads, not parameter count.

---

## Setup

### Environment note — WSL2

vLLM 0.26.0 fails at engine initialization on WSL2 with
`RuntimeError: UVA is not available`. The V2 model runner, default since
0.26, allocates unified-virtual-addressing buffers for token-id staging, and
WSL2's CUDA layer does not expose UVA. The V1 runner is still present in the
codebase — it is simply no longer the default:

```
vllm/v1/worker/gpu_worker.py   ->  self.use_v2_model_runner
vllm/config/vllm.py:551        ->  envs.VLLM_USE_V2_MODEL_RUNNER
```

Setting `VLLM_USE_V2_MODEL_RUNNER=0` selects the V1 path and the engine starts
normally. Native Linux hosts are unaffected.

### Cluster

Single-node **k3s** (CNCF-conformant Kubernetes) running with the Docker
runtime, so the NVIDIA container toolkit is shared with the host Docker daemon
rather than configured twice. GPUs are scheduled through the standard NVIDIA
device plugin, which means the resource model exercised here —
`nvidia.com/gpu: 1`, whole-device allocation, no oversubscription — is
identical to any managed Kubernetes cluster.

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

curl -sfL https://get.k3s.io | sh -s - --docker --disable traefik \
  --write-kubeconfig-mode 644

kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
kubectl describe node | grep nvidia.com/gpu    # must report 1
```

### Deploy

```bash
kubectl apply -f manifests/vllm.yaml
kubectl port-forward svc/vllm 8000:8000
curl http://localhost:8000/v1/models
```

---

## Findings

### Startup breakdown

| Phase | Duration |
| --- | --- |
| Weight loading | 47.0 s |
| Engine init — profiling, KV-cache allocation, warmup | 91.7 s |
| Other — tokenizer fetch, server bootstrap | ~24 s |
| **Total to `Application startup complete`** | **163 s** |

Weight loading is only 29% of startup; the majority is memory profiling and
CUDA graph capture, of which 10.3 s is compilation. This number matters twice
over — it is the direct cause of the probe failure below, and it is why
horizontal autoscaling cannot react usefully to load.

This is also a 1.5B model. Startup scales with weight size, so a 7B or 70B
deployment is strictly worse.

### Throughput and latency under load

<!-- CAPTURE: sweep request rate from low to saturation. Record TTFT p50/p99,
     TPOT p50/p99, queue depth, and output throughput at each rate.
     Chart it and save to docs/latency-curve.png -->

![Latency vs request rate](docs/latency-curve.png)

| Request rate (req/s) | TTFT p99 | TPOT p99 | Queue depth | Output tok/s |
| --- | --- | --- | --- | --- |
| | | | | |

**Target workload.** Chat-style traffic: 200-token prompts, 300-token outputs,
single turn. **SLO:** TTFT p99 under 2 s, TPOT p99 under 50 ms. Saturation is
defined as the arrival rate at which p99 TTFT crosses the SLO — not simply the
point where the throughput curve flattens.

**Saturation point:** `[N]` req/s.

**What degrades first, and why.** `[FILL from observation. Expectation: TTFT
climbs sharply while TPOT stays comparatively flat, because requests queue for
prefill admission rather than decoding slowly. Record what actually happened,
including any divergence from that expectation.]`

### Benchmark methodology

Load was generated **open-loop** at a fixed arrival rate rather than with a
fixed concurrency pool. Closed-loop generators stop issuing new requests while
blocked on slow responses, which systematically under-reports tail latency —
the same coordinated-omission problem `wrk2` was written to correct.

Client saturation was ruled out by `[FILL: how you confirmed the load generator
was not itself the bottleneck]`.

```bash
[BENCHMARK COMMAND]
```

---

## Failure modes reproduced

Each of these was induced deliberately, observed through the monitoring stack,
and then fixed.

### 1. Model load time exceeds the liveness probe threshold

**Configuration.** A default-looking liveness probe —
`initialDelaySeconds: 30`, `periodSeconds: 10`, `failureThreshold: 3` — which
declares the container dead 60 seconds after start.

**Symptom.** `[FILL: restart count, CrashLoopBackOff, describe output]`

<!-- CAPTURE: kubectl describe pod events showing "Liveness probe failed",
     and kubectl get pods showing the restart counter climbing.
     Save to docs/crashloop.png -->
<img width="1107" height="304" alt="image" src="https://github.com/user-attachments/assets/7e867027-a0b7-4501-80d9-a25335f01c64" />
<img width="1364" height="489" alt="image" src="https://github.com/user-attachments/assets/35e5f7de-584b-46ae-8f20-e90addf9962a" />



**Root cause.** Startup takes 163 s; the probe kills the container at 60 s.
The container logs show a clean, in-progress load with no errors at the moment
of termination — the service was healthy, and the probe's assumption about
startup duration was wrong.

<img width="758" height="342" alt="image" src="https://github.com/user-attachments/assets/7d92efab-20aa-4987-8550-b41cf799a60e" />

**Fix.** `[FILL: startupProbe with a high failureThreshold — and why that is
the correct answer rather than simply raising livenessProbe
initialDelaySeconds. The distinction: startupProbe hands control to
livenessProbe the moment startup succeeds, whereas a long initialDelay is a
fixed guess that leaves the container unmonitored for that entire window and
breaks again the moment the model changes.]`

### 2. KV cache exhaustion under load

**Symptom.** `[FILL: throughput collapse, preemption counter climbing]`

<!-- CAPTURE: Grafana panel showing vllm:kv_cache_usage_perc and the
     preemption counter during the event -->

**Root cause.** `[FILL]`

**Fix.** `[FILL]`

### 3. `[OPTIONAL — long-prompt prefill blocking decode for other requests]`

---

## Autoscaling analysis

Native Kubernetes HPA is the wrong tool for this workload, for two independent
reasons.

**Wrong signal.** CPU utilization is uncorrelated with inference load — the GPU
saturates while CPU stays near idle. GPU utilization is a duty-cycle metric and
cannot distinguish a lightly-batched step from a heavily-batched one. GPU
memory is preallocated to the KV cache at startup and stays flat regardless of
load, so it never triggers a scaling decision either. The meaningful signals
are request queue depth (`vllm:num_requests_waiting`) and KV-cache utilization
(`vllm:kv_cache_usage_perc`), both of which need an external metrics adapter to
reach the HPA.

**Wrong time scale.** Scale-up latency is dominated by model load — **163
seconds measured here**, and worse for larger models. That is longer than the
traffic fluctuations HPA would be reacting to; by the time a new replica is
ready, the spike it was provisioned for has passed. When scale-up latency
exceeds the period of the load variation, autoscaling is not a slow solution,
it is the wrong category of solution.

`[FILL: what to do instead — provisioned headroom sized to peak, predictive
scaling against a known traffic curve, or queue-depth-driven KEDA with long
stabilization windows and a warm pool. State which you would pick and why.]`

Full write-up: [`docs/autoscaling.md`](docs/autoscaling.md)

---

## Known limitations

This is a single-GPU learning environment, not a production deployment.

- **Single node, single replica.** One GPU means no multi-replica routing, no
  load-balancing strategy validation, and no real autoscaling test. Single-node
  k3s also differs from a managed cluster in control-plane topology, though the
  workload-level concerns exercised here are distribution-independent.
- **Small model.** 1.5B in bf16. Behaviour at 70B scale with tensor parallelism
  is not exercised, and the memory arithmetic shifts substantially once weights
  dominate the budget.
- **WSL2, not bare metal.** Required the UVA workaround above; other CUDA
  feature gaps may exist that this project did not surface.
- **Synthetic load.** Prompt and output length distributions are generated, not
  drawn from production traffic. No shared prefixes, so prefix caching is not
  meaningfully exercised.
- **No multi-model serving**, no request prioritization, no quota enforcement.
- **Short duration.** Runs measured in minutes; no data on memory fragmentation
  or stability over days.

---

## What I would do differently at scale

- **Routing.** Round-robin is wrong when request durations vary by two orders
  of magnitude. Least-outstanding-requests, or prefix-cache-aware routing via
  the Kubernetes Gateway API Inference Extension.
- **Model weight distribution.** Baking multi-GB weights into images makes
  pulls slow; downloading on every restart is worse. A pre-warmed PVC or a
  node-local cache is the usual answer, and it attacks the startup-latency
  problem that breaks autoscaling.
- **Graceful shutdown.** A pod may hold dozens of in-flight streaming
  generations, each running for tens of seconds. The default 30-second
  termination grace period is not enough.
- **Prefill/decode disaggregation.** Splitting the two phases across separate
  GPU pools removes the head-of-line blocking a long prompt causes for everyone
  else — the approach taken by NVIDIA Dynamo and llm-d.
- **Cost accounting.** GPU cost dominates everything, so saturation throughput
  is better expressed as cost per million tokens than as tokens per second.

---

## Repo layout

```
manifests/    Kubernetes manifests, ServiceMonitor, Grafana dashboards
bench/        Load generation scripts and raw results
docs/         Build log, charts, autoscaling write-up
charts/       Helm chart
```

A chronological record of what was tried, what broke, and what the numbers were
at each step is in [`docs/build-log.md`](docs/build-log.md).
