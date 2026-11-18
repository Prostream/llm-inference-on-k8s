> Historical notes, not newly reproduced results. Several causal, timing and
> version statements need correction; see [evidence ledger](evidence-ledger.md).
> New benchmark tooling does not establish new GPU performance measurements.

RuntimeError: UVA is not available
at vllm/v1/worker/gpu/buffer_utils.py -> UvaBuffer.init


Traced it:

vllm/v1/worker/gpu_worker.py:378 logs "Using V2 Model Runner"
vllm/v1/worker/gpu_worker.py:174 self.use_v2_model_runner = vllm_config...
vllm/config/vllm.py:551 envs.VLLM_USE_V2_MODEL_RUNNER


The V2 model runner — default since 0.26 — allocates unified-virtual-
addressing buffers for token-id staging. WSL2's CUDA layer does not expose
UVA. The V1 runner is still in the codebase (`XPUModelRunnerV2 if
use_v2_model_runner else XPUModelRunner`), just no longer the default.

**Workaround:** `VLLM_USE_V2_MODEL_RUNNER=0`. Native Linux hosts unaffected.

### Startup timing

| Phase | Duration |
|---|---|
| Weight loading | 47.0 s |
| Engine init (profiling, KV-cache alloc, warmup) | 91.7 s (10.3 s compile) |
| Other (tokenizer fetch, server bootstrap) | ~24 s |
| **Total to `Application startup complete`** | **163 s** |

Weight loading is only 29% of it. This number drives both the probe
configuration and the autoscaling analysis.

### Memory budget

At `gpu_memory_utilization=0.80`, vLLM reserves 9.55 GiB:

| Allocation | Size |
|---|---|
| Model weights | 2.98 GiB |
| Peak activation | 0.55 GiB |
| CUDA graph pool | 0.20 GiB |
| Non-torch | 0.06 GiB |
| **KV cache** | **5.64 GiB** |

`nvidia-smi` after load: 11,284 / 12,227 MiB.

### KV cache capacity

GPU KV cache size: 211,120 tokens
Maximum concurrency for 4,096 tokens per request: 51.54x


5.64 GiB / 211,120 tokens ≈ **28 KB per token**, which matches the
architecture exactly:

2 (K and V) x 28 layers x 2 KV heads x 128 head_dim x 2 bytes (bf16) = 28,672 B


The KV-head count is the driver, not the attention-head count — Qwen2.5 uses
GQA, 12 query heads sharing 2 KV heads, so the cache is 6x smaller than an
equivalent MHA model would need.

### Verified

`/v1/models`, `/v1/chat/completions`, and `/metrics` all responding.

---

## Phase 2 — Kubernetes

Single-node k3s with `--docker`, so the NVIDIA container toolkit configured in
Phase 0 is reused rather than configured a second time for containerd.

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
curl -sfL https://get.k3s.io | sh -s - --docker --disable traefik \
  --write-kubeconfig-mode 644
kubectl apply -f .../nvidia-device-plugin.yml
```

Node reports `nvidia.com/gpu: 1`.

The device plugin is how Kubernetes handles resources it has no native concept
of. It advertises discrete devices, which is why GPUs cannot be
oversubscribed the way CPU can — there is no `500m` of a GPU.

---

## Phase 3 — probe misconfiguration (in progress)

**Intent:** deliberately configure a liveness probe that declares the
container dead at 60 s (`initialDelay 30 + 3 x period 10`) when startup
actually takes 163 s, and observe the resulting CrashLoopBackOff.

Three unrelated failures had to be cleared first.

### Incident 1 — misread the failure

First run produced CrashLoopBackOff, and `describe` showed
`Liveness probe failed`. Tempting to call it a successful reproduction.

It was not:

- Container died at **33 s**, but the probe cannot act before **60 s**
  (initialDelay 30 + 3 x period 10).
- Event showed `Liveness probe failed (x2)` — with `failureThreshold: 3` the
  probe had not yet acted.
- `STATUS=Error` plus a `BackOff` event indicates a non-zero self-exit. A
  probe kill produces a `Killing` event and exit code 137.

The probe was failing, but only because the port was not open yet during a
normal slow start. Something else was killing the container.

**Takeaway:** evidence that superficially matches the expected failure mode is
not the expected failure mode. Check the timing and the exit path.

### Incident 2 — rolling update deadlock on a single GPU

`kubectl rollout restart` hung with the new pod `Pending`.

Default `RollingUpdate` starts the replacement before terminating the
incumbent. With one GPU, the old pod holds `nvidia.com/gpu: 1` and the new pod
cannot be scheduled. The old pod was in CrashLoopBackOff and would never
become ready, so the deadlock was permanent.

**Fix:** `strategy.type: Recreate`.

**Tradeoff worth stating:** Recreate means the update window is a full 163 s
of zero availability. Production would need a spare GPU for blue/green.

This is a clean example of a GPU workload breaking an assumption that holds
for ordinary stateless services — replicas cannot briefly coexist when the
resource is exclusive and indivisible.

### Incident 3 — YAML indentation

`replicas`, `strategy`, and `selector` were nested under `template.spec`
instead of the Deployment `spec`. `kubectl apply --dry-run=client` catches
this class of error in seconds; adopted as a habit.

### Root cause — VLLM_PORT collision with Kubernetes service links

ValueError: VLLM_PORT 'tcp://10.43.172.217:8000' appears to be a URI.
This may be caused by a Kubernetes service discovery issue


Kubernetes injects legacy Docker-link-style environment variables for every
existing Service: `<SERVICE_NAME>_PORT`, `<SERVICE_NAME>_SERVICE_HOST`, etc.

The Service was named `vllm`, so pods received
`VLLM_PORT=tcp://10.43.172.217:8000`. vLLM uses an environment variable of the
same name for its own internal port and expects an integer. Namespace
collision between a Kubernetes legacy feature and application config.

**Fix:** `enableServiceLinks: false` on the pod spec, plus renaming the
Service to `llm-inference` so nothing else trips over the same prefix.

**Why this one is interesting:** the identical configuration ran fine under
bare Docker in Phase 1. The difference was purely the orchestrator injecting
environment into the container. Having the Docker baseline made that
bisection immediate.

### Open observation

`QoS Class: BestEffort` — the pod declares only `nvidia.com/gpu` and no
CPU/memory requests or limits. Extended resources do not count toward QoS, so
the inference pod sits in the lowest eviction tier despite a 163 s restart
cost. Should be Guaranteed or at least Burstable in anything real.

### Status

Blockers cleared. The intended probe experiment has not run yet.
