# Evidence ledger

Revision work began on 2026-11-17 (America/Los_Angeles). New work is not evidence
of what was complete before the interview or a previously submitted resume.

| Claim | Evidence | Status |
| --- | --- | --- |
| Deployment, probes and Service | `manifests/vllm.yaml` | Configuration exists; observe running state separately |
| Slow initialization | Historical build log reports 163 seconds | Historical note, not new measurement |
| Probe-triggered restart | Historical screenshots and exit-state fragment | Partial; full events and previous logs needed |
| Service-link collision | Historical error string and corrective manifest | Recorded diagnosis, separate from probe experiment |
| GPU rollout resource conflict | Historical notes, current Recreate setting | Mechanism and recorded observation; outage not remeasured |
| KV capacity | Historical log numbers plus model calculation | Capacity estimate, not serving performance |
| Admission queueing bottleneck | No full sweep or intervention in original checkout | Unverified; do not claim as established |
| Helm chart, Grafana dashboard | Absent in original checkout | Not demonstrated |
| New benchmark tooling | `bench/`, tests and live report | 200 real requests completed in smoke/pilot; CPU tests tracked separately |
| Cache-state confound | Live hit-counter deltas and seeded workload commands | Reused prefixes invalidate a simple rate-only interpretation |
| Performance improvement | Requires real controlled runs | Unmeasured until evidence is attached |

Corrections to historical text: 137/Error/BackOff alone do not determine root
cause; a 300-second budget does not tolerate 400 seconds; the startup-to-ready gap
is not fully explained; 51.54 is a capacity ratio; hostPath does not pin placement;
whole-device allocation here does not rule out GPU sharing elsewhere; WSL/UVA
version claims need rechecking; HPA can use custom metrics; a grace period does
not implement or validate draining.

For every run retain immutable image and model revision, actual versions and
server args, workload lengths/arrival/seed/cache/warmup, raw client results and
errors, metrics, client location/resources and one-variable comparison controls.
Preserve failed and adverse results. A plausible explanation is not an observation.

See [first live validation](validation-2026-11-17.md) for measured values, raw data
and the remaining controls. Initial large TTFT was not explained by the measured
scheduler queue; its full cause remains unresolved.
