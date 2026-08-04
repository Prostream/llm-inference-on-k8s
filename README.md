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
driver for memory that is not actually free.

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
kubectl port-forward svc/llm-inference 8000:8000
curl http://localhost:8000/v1/models
```

The deliberately misconfigured variant used for the probe experiment below is
kept alongside it as [`manifests/vllm-broken.yaml`](manifests/vllm-broken.yaml).

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

| Request rate (req/s) | TTFT p99 | TPOT p99 | Queue depth | Output tok/s |
| --- | --- | --- | --- | --- |
| | | | | |

**Target workload.** Chat-style traffic: 200-token prompts, 300-token outputs,
single turn. **SLO:** TTFT p99 under 2 s, TPOT p99 under 50 ms. Saturation is
defined as the arrival rate at which p99 TTFT crosses the SLO — not simply the
point where the throughput curve flattens.

**Saturation point:** `[N]` req/s.

**What degrades first, and why.** `[FILL from observation.]`

### Benchmark methodology

Load was generated **open-loop** at a fixed arrival rate rather than with a
fixed concurrency pool. Closed-loop generators stop issuing new requests while
blocked on slow responses, which systematically under-reports tail latency —
the same coordinated-omission problem `wrk2` was written to correct.

Client saturation was ruled out by `[FILL]`.

---

## Failure modes reproduced

### 1. Model load time exceeds the liveness probe threshold

**Configuration.** A default-looking liveness probe —
`initialDelaySeconds: 30`, `periodSeconds: 10`, `failureThreshold: 3` — which
declares the container dead 50 s after start, against a measured startup of
163 s.

**Symptom.** The container was killed and restarted on a perfectly regular
110-second cycle, five times before the run was stopped.

```
04:04:48  Started
04:05:18  Liveness probe failed        <- initialDelaySeconds elapsed
04:05:28  Liveness probe failed
04:05:38  Liveness probe failed        <- failureThreshold reached
04:05:38  Killing: failed liveness probe, will be restarted
```
<img width="959" height="353" alt="image" src="https://github.com/user-attachments/assets/fff37cfe-20ae-4b8c-a3d5-274b91c12fd7" />

`Last State: Terminated, Reason: Error, Exit Code: 137` — SIGKILL from the
kubelet, not a self-inflicted exit.

**Root cause.** Startup takes 163 s; the probe kills at 50 s. The container
logs show a clean, in-progress load with no errors at the moment of
termination. The service was healthy; the probe's assumption about startup
duration was wrong.

**Two things the reproduction surfaced that I did not expect:**

*It never entered CrashLoopBackOff.* Each run survived ~110 s — long enough
that the kubelet's restart back-off never escalated. `kubectl get pods` showed
`STATUS: Running` throughout. The only signals were the restart counter
climbing and `READY 0/1`. A permanently broken service that looks healthy in
the column most people scan is considerably more dangerous than one that
announces itself as CrashLoopBackOff.

*Sixty of those 110 seconds were self-inflicted.* The cycle decomposes into
~50 s of runtime plus the full `terminationGracePeriodSeconds: 60` — vLLM does
not respond to SIGTERM while loading the model, so every kill waited out the
entire grace period before SIGKILL. A setting added for a good reason
(draining in-flight streaming requests) doubled recovery time in the failure
case.

**Fix.** A `startupProbe` with `periodSeconds: 10, failureThreshold: 30`, and
`initialDelaySeconds` removed from the liveness probe entirely.

This is not equivalent to setting `initialDelaySeconds: 200`, for three
reasons:

- **A genuine hang stays visible.** With a long initial delay, nothing watches
  the container for that entire window. The startupProbe polls throughout and
  kills once the budget is exhausted.
- **It survives a model change.** `failureThreshold x periodSeconds` is a
  ceiling, not a fixed wait. A 7B model taking 400 s needs a new guess under
  the initialDelay approach; under a budget it does not.
- **It decouples two goals that `initialDelaySeconds` forces into one
  parameter** — tolerating slow startup, and detecting death quickly. Because
  the startupProbe owns the first, liveness can now be aggressive:
  `periodSeconds: 10, failureThreshold: 3` detects a hang within 30 s.

The handover is driven by startup actually completing rather than by a clock.
Probe semantics should match the question the probe answers.

**Result.** `READY 1/1` after 185 s with zero restarts. The extra ~22 s over
the 163 s baseline is probe polling granularity — a 10 s `periodSeconds` means
readiness can lag actual availability by up to a full period. Smaller periods
cut that lag at the cost of more probe traffic during startup.

<img width="768" height="348" alt="image" src="https://github.com/user-attachments/assets/b3d92684-964a-4db9-8e2a-002eb6412556" />


### 2. Rolling update deadlocks on a single GPU

**Symptom.** `kubectl rollout restart` left the new pod `Pending`
indefinitely.

**Root cause.** The default `RollingUpdate` strategy starts the replacement
before terminating the incumbent. With one GPU, the old pod holds
`nvidia.com/gpu: 1` and the new pod cannot be scheduled. The old pod was in a
restart loop and would never become ready, so the deadlock was permanent.

This is the clearest example in the project of a GPU workload breaking an
assumption that holds for ordinary stateless services: replicas cannot briefly
coexist when the resource is exclusive and indivisible.

**Fix.** `strategy.type: Recreate`.

**Tradeoff.** The update window becomes a full cold start — 165 s of zero
availability. Production would need a spare GPU for blue/green rather than
accepting that window.

### 3. `VLLM_PORT` collides with Kubernetes service links

**Symptom.** The pod exited at ~33 s on every attempt with:

```
ValueError: VLLM_PORT 'tcp://10.43.172.217:8000' appears to be a URI.
This may be caused by a Kubernetes service discovery issue
```

**Root cause.** Kubernetes injects legacy Docker-link-style environment
variables for every existing Service: `<SERVICE_NAME>_PORT`,
`<SERVICE_NAME>_SERVICE_HOST`, and so on. The Service was named `vllm`, so
pods received `VLLM_PORT=tcp://10.43.172.217:8000`. vLLM reads an environment
variable of the same name for its own internal port and expects an integer.
A namespace collision between a Kubernetes legacy feature and application
configuration.

**Fix.** `enableServiceLinks: false` on the pod spec — modern service
discovery is DNS, and these variables are dead weight — plus renaming the
Service to `llm-inference` as defence in depth.

**Why this one was instructive.** The identical configuration ran fine under
bare Docker. Having built that baseline first made the bisection immediate:
same image, same args, same GPU, different orchestrator, therefore the
orchestrator was injecting something. Without it I would have been debugging
vLLM instead of Kubernetes.

I also initially misread the resulting restart loop as the probe failure I was
trying to induce — the events did contain `Liveness probe failed`. But the
container was dying at 33 s when the probe could not act before 50 s, the
failure count had only reached 2 of 3, and `STATUS: Error` with a `BackOff`
event indicates a self-exit rather than a kubelet kill. Evidence that
superficially matches the expected failure mode is not the expected failure
mode.

### 4. KV cache exhaustion under load

**Symptom.** `[FILL: throughput collapse, preemption counter climbing]`

<!-- CAPTURE: Grafana panel showing vllm:kv_cache_usage_perc and the
     preemption counter during the event -->

**Root cause.** `[FILL]`

**Fix.** `[FILL]`

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

`[FILL: what to do instead.]`

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
- **`hostPath` weight cache.** Pins the pod to one node. Acceptable on a single
  node, wrong anywhere else.
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
  node-local cache DaemonSet is the usual answer, and it attacks the
  startup-latency problem that breaks autoscaling.
- **Graceful shutdown.** A pod may hold dozens of in-flight streaming
  generations. The 60 s grace period used here is enough to drain them, but as
  failure mode 1 showed, it also doubles recovery time when the process is not
  in a state to handle SIGTERM. A preStop hook that refuses new work and drains
  explicitly would separate those two cases.
- **Prefill/decode disaggregation.** Splitting the two phases across separate
  GPU pools removes the head-of-line blocking a long prompt causes for everyone
  else — the approach taken by NVIDIA Dynamo and llm-d.
- **Cost accounting.** GPU cost dominates everything, so saturation throughput
  is better expressed as cost per million tokens than as tokens per second.

---

## Repo layout

```
manifests/    Kubernetes manifests (working and deliberately broken)
bench/        Load generation scripts and raw results
docs/         Build log, captured evidence, charts
charts/       Helm chart
```

A chronological record of what was tried, what broke, and the numbers at each
step is in [`docs/build-log.md`](docs/build-log.md).
