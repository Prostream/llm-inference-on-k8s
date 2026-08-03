# Build log — 2026-08-03

## Phase 0 — environment
- WSL2 Ubuntu on F: drive, 24GB RAM, systemd enabled
- Docker Engine + nvidia-container-toolkit, nvidia set as default runtime
- RTX 5070: 12227 MiB total, ~900 MiB held by Windows desktop
- Driver 610.53, CUDA UMD 13.3
- k3s with --docker, NVIDIA device plugin → node reports nvidia.com/gpu: 1

## Phase 1 — bare Docker baseline
- Blocker: vLLM 0.26.0 fails at engine init with "UVA is not available"
  - V2 model runner (default since 0.26) allocates UVA buffers
  - WSL2's CUDA layer does not expose UVA
  - Traced gpu_worker.py -> config/vllm.py:551 -> envs.VLLM_USE_V2_MODEL_RUNNER
  - V1 runner still present; workaround VLLM_USE_V2_MODEL_RUNNER=0
- Startup 163s total: weights 47.0s, engine init 91.7s (compile 10.3s), rest ~24s
- Memory: weights 2.98 / activation 0.55 / cudagraph 0.20 / non-torch 0.06 GiB
- KV cache: 5.64 GiB = 211,120 tokens; max concurrency @4096 ctx = 51.54x
- Derived 28 KB/token; matches 28 layers x 2 KV heads x 128 dim x 2 x 2 bytes
