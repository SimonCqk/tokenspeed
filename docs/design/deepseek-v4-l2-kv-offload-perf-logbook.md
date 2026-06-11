# DeepSeek V4 L2 KV Offload Performance Logbook

Status: living diagnostic log for `feat/ds-v4-l2-offload`.
Last updated: 2026-06-11.

Read this document before starting new DeepSeek V4 L2/prefix-cache performance
work. It captures what has already been tried, what was ruled out, and where
the current evidence points.

## How To Use This Logbook

1. Do not restart from old hypotheses unless new logs contradict the evidence
   below.
2. When adding a new optimization, record the before/after result here with the
   exact log markers used.
3. Keep conclusions evidence-based. Prefer `slow_attn`, `slow_layer`,
   `slow_detail`, `loadback done`, and paged-L2 transfer logs over raw TPS
   alone.
4. During this session, every new perf exploration must append an entry to this
   file before the investigation is considered complete.

## Baseline And Functional Notes

- DeepSeek V4 KVStore uses the existing default-on behavior. Do not add an
  `--enable-kvstore` flag; use `--disable-kvstore` only for ablation.
- `origin/main` with `--disable-kvstore` was used as a working baseline.
  Earlier stalls on this branch with `--disable-kvstore` were treated as branch
  bugs until fixed; do not use those early failures as performance evidence.
- Enable-KVStore and disable-KVStore runs both eventually reached functional
  completion. The current unresolved issue is not correctness or hang, but that
  enable-KVStore does not yet produce the expected long-context TPS advantage.
- Small-context or no-eviction workloads are not useful L2 perf tests. Use a
  constrained `--max-total-tokens` and repeated agentic long-context turns to
  force device eviction and host hits.
- If a request's live working set exceeds the configured device token budget,
  the run can appear blocked during prefill. That is a capacity/scheduling
  stress condition, not evidence that L2 transfer is slow.
- When `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton` is
  changed, remote perf validation must reinstall or rebuild the kernel package.
  Otherwise the runtime may execute stale Triton wrappers.

## Benchmark Setup Used In This Session

Representative server arguments:

```bash
SPDLOG_LEVEL=debug NCCL_DEBUG=INFO tokenspeed serve \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --port 30101 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --attention-use-fp4-indexer-cache \
  --moe-backend mega_moe \
  --dist-init-addr 127.0.0.1:14013 \
  --prometheus-port 11111 \
  --enable-mixed-batch \
  --gpu-memory-utilization 0.85 \
  --max-total-tokens 81920
```

Use `--disable-kvstore` only for the ablation run. To stress eviction more
aggressively, lower `--max-total-tokens`, but keep enough budget for the active
request working set.

Representative eval command:

```bash
evalscope perf \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --url http://localhost:30101/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path /tmp/agentic_dataset.json \
  --max-tokens 500 \
  --multi-turn \
  --number 6 \
  --parallel 2 \
  --dataset-offset 68 \
  --extra-args '{"ignore_eos": true}'
```

## Current Root-Cause Summary

The L2 host cache path is functionally active and usually not the dominant
runtime cost. The current TPS gap is mostly explained by DeepSeek V4 cached
prefill forward cost after host loadback:

- Layer 2 CSA has a stable fixed update cost. In the 2026-06-11 09:08 log,
  `layer=2 kind=csa` appears 42 times with `indexer_stream_scope` around
  0.58 s. The time splits roughly into `indexer` around 0.31 s and
  `compressor` around 0.27 s, even when the cached-prefill tail has only a few
  hundred or a few thousand new tokens.
- DP synchronization can hide per-request L2 gains. Cached-prefill requests are
  often co-scheduled with another DP rank doing an 8192-token prefill, so the
  global forward step follows the slowest rank.
- A secondary cost remains in CSA/HCA prefill workspace materialization.
  `flashmla` is tiny in these samples; the expensive phase is workspace
  gather/dequantize/materialize, not scalar metadata sync.

## Explored Directions And Verdicts

| Direction | Verdict | Evidence / Notes |
| --- | --- | --- |
| Origin/main / `--disable-kvstore` comparison | Baseline only | `origin/main --disable-kvstore` established that the long-context workload can complete without the V4 L2 path. Use it as a control, but remember it cannot validate V4 KVStore behavior because V4 KVStore support lives on this branch. |
| Claude branch `feat/v4-l2-kv-offload` comparison | Informational | The alternate branch was reviewed for design ideas. The current branch kept the scheduler-owned visibility model and group-paged host-pool control flow rather than importing a separate transfer design. |
| Mamba L2 overlap model | Reference pattern | Mamba L2 uses the same broad architecture: scheduler-visible cache state plus async host/device transfer, not a custom transfer kernel. This supports the current staged loadback/writeback direction. |
| L2 loadback/writeback transfer as primary bottleneck | Ruled out for current logs | Loadback is usually tens to low hundreds of ms. In the 2026-06-11 09:08 log, loadback p50 was about 95 ms and p90 about 119 ms, while forward slow paths were about 1-1.5 s. |
| Paged transfer span ordering | Implemented, not sufficient | `_ordered_page_pairs()` ordering reduced avoidable span fragmentation, and logs now print pages/spans. Remaining TPS gap is dominated by forward cost, not host transfer. |
| Selectively offload only some page groups | Deferred | The bottleneck moved to cached-prefill forward after loadback. Reducing L2 bytes may help later, but current evidence does not justify dropping cache groups. |
| Custom transfer kernel | Out of current implementation scope | vLLM-style staged transfer/loadback first is still the right baseline. Do not add transfer kernels until forward bottlenecks are fixed. |
| Indexer logits workspace cap (`--deepseek-v4-indexer-prefill-max-logits-mb 128`) | Ruled out as primary knob | User-run benchmark showed almost no end-to-end TPS change. The remaining issue is not a simple logits workspace cap. |
| HCA cache insert materialization | Mostly addressed | HCA tiled materialization and compact insert were added. Later logs showed cache-insert CUDA time is tiny when wall time looks large, pointing to backlog attribution or different workspace costs. |
| CSA cache insert compact rows | Implemented, not current bottleneck | Compact rows are active. Cache insert itself is sub-ms in the key samples; layer 2 time remains in indexer/compressor update and sparse indexer work. |
| CSA compressor/indexer overlap | Reverted | The experiment did not produce useful overlap and introduced extra variables, including independent slot-cache recomputation. Commit `2b0bf57` removed `csa_update_overlap` and `run_compressor({})`. |
| `_prefill_workspace` scalar `.item()` sync | Fixed / no longer current root cause | DeepSeek V4 model/backend/cache paths have no `.item()` in the audited files. Workspace sizing uses `metadata.seq_lens_cpu` and `metadata.query_lens_cpu`; missing CPU metadata fails fast instead of falling back to GPU scalar reads. |
| Python debug log expansion | Diagnostic-only | More Python slow logs helped locate forward phases, but should remain slow-path/debug gated. Do not keep broad noisy logs in normal perf runs. |
| Health-check timeout outliers | Track separately | One 7 s forward outlier coincided with gateway health timeout and was not explained by layer phase totals. Do not treat it as the steady-state L2 root cause unless it reproduces. |

## Timeline Of Material Findings

### Functional Bring-Up

- `--disable-kvstore` initially still hit stalls on the branch, while
  `origin/main --disable-kvstore` completed. That was a branch correctness
  signal, not a KVStore performance signal.
- After fixes, both disable-KVStore and enable-KVStore paths completed on the
  agentic workload. Subsequent work focused on why enable-KVStore did not
  improve TPS enough.

### Transfer And Capacity Experiments

- `--max-total-tokens` was used to force device-cache pressure and host hits.
  Extremely small budgets can create prefill admission/capacity stalls if the
  active request working set is larger than the configured budget.
- Paged L2 logs showed real host hits and transfers. Loadback latency remained
  far below the forward slow path in the useful perf runs.
- Sorting `_ordered_page_pairs()` and printing transfer pages/spans improved
  observability and reduced unnecessary fragmentation, but did not materially
  move end-to-end TPS.

### Attention/Indexer Experiments

- HCA cache insert and materialization were optimized first because early logs
  showed large wall time near HCA compressor/cache insert. Later CUDA timings
  showed actual cache insert kernels were small; remaining HCA/CSA cost moved
  to workspace materialization or backlog attribution.
- CSA compact cache insert reduced unnecessary insert work and is active, but
  layer 2 CSA still pays a stable `indexer + compressor` update cost.
- CSA compressor/indexer overlap was tried and then removed. It did not overlap
  usefully and made phase attribution harder.
- Removing `.item()` from DeepSeek V4 prefill workspace sizing was necessary,
  but later logs still showed workspace materialization can be expensive even
  without scalar sync.

### Latest Evidence As Of 2026-06-11

- Enable-KVStore runs show host prefix hits such as thousands of new tokens with
  tens of thousands of cached tokens.
- The dominant repeated cost is layer 2 CSA `indexer_stream_scope` around
  0.58 s, which is too fixed relative to the cached-prefill tail length.
- When a cached-prefill rank is co-scheduled with an 8192-token full prefill
  rank, DP synchronization makes the cached request wait for the slow rank, so
  local prefix reuse does not translate directly into higher TPS.

## Current Optimization Targets

### 2026-06-11 Next-Step Decision

The current goal is to make L2 cache show a clear benefit in the
capacity-bound multi-turn benchmark, preferably in host-hit TTFT / prefill to
first decode first, and then in aggregate TPS once scheduling stops hiding the
per-request win. Steady decode throughput is not the primary target because the
latest logs keep decode around the same stable band after prefill completes.

The next implementation should not start from L2 transfer bytes or cache insert
again. The best path is:

1. Attack layer 2 CSA cached-prefill fixed cost first. This is the repeated
   local cost paid by host-hit turns, so reducing it directly improves L2
   value. The success condition is that layer 2 `indexer_stream_scope` no
   longer stays near 0.6 s for small tail prefills and instead scales with the
   effective tail / compact rows.
2. Add a minimal prefix-reuse-aware DP scheduling heuristic second. The current
   logs show cached-prefill ranks can be co-scheduled with full 8192-token
   prefill ranks; then the DP step follows the slowest rank and hides the L2
   benefit. This should be a conservative binning heuristic, not a scheduler
   redesign: avoid mixing host-hit short-tail prefills with large uncached
   prefills when decode priority and overlap scheduling allow it.
3. Treat CSA/HCA workspace materialization as a follow-up only after layer 2
   fixed cost and DP straggler amplification are measured. Workspace remains a
   secondary suspect, but it is less directly tied to the L2 capacity win than
   layer 2 CSA fixed update cost and DP co-scheduling.

Use these pass/fail signals before considering the work complete:

- Enable-KVStore host-hit turns have visibly lower TTFT / prefill-to-first
  decode than disable-KVStore under the same capacity-bound benchmark.
- Layer 2 CSA `indexer_stream_scope` drops materially from the current
  ~0.58 s fixed band on small cached-prefill tails.
- Scheduler debug logs show fewer global steps where a host-hit cached prefill
  is paired with another rank's 8192-token uncached prefill.

### 2026-06-11 Prefix-Reuse-Aware DP Scheduling Attempt

Implemented a minimal two-phase DP scheduling heuristic in the Python event
loop:

- First DP sync gathers proposed local work plus two CPU-only counters:
  prefill tokens and true prefix-reuse tokens.
- If the proposal contains both a prefix-hit short-tail prefill and an
  uncached full chunk, uncached full-prefill ranks defer their already built
  `forward_op`, participate in a second DP sync as idle, and run the deferred
  op on the next iteration.
- Deferred ops are not allowed to defer again on their replay iteration, which
  bounds starvation risk.
- Decode rows hidden in mixed batches are not deferred; this preserves decode
  priority.

The implementation adds `prefix_reuse_lens` to scheduler forward ops so the
heuristic does not confuse ordinary chunked-prefill continuation with real
prefix reuse. For first prefill chunks, `prefix_reuse_lens` is the reused cache
prefix length; for later chunks it remains zero even when `extend_prefix_lens`
is non-zero.

Expected validation markers:

- New debug marker appears only when the heuristic fires:
  `[Scheduler][dp_prefix_reuse] defer uncached prefill`.
- In the following final DP sync for that step, the deferring rank should have
  local/global mode idle for that rank and `global_prefill_tokens` should no
  longer include its uncached full chunk.
- Host-hit turns should reach first decode earlier; aggregate TPS may improve
  only if the saved TTFT outweighs the extra separated DP step.

### 1. Layer 2 CSA Cached-Prefill Fixed Cost

Primary target. The path to inspect is:

- `DeepseekV4Attention.forward`: layer 2 CSA `indexer_stream_scope`.
- `DeepseekV4Indexer.forward`: indexer compressor state update, compressed slot
  mapping, indexer cache insert, and sparse indexer custom op.
- `_forward_sparse_indexer_custom_op`: FP4 indexer preparation and prefill
  sparse top-k path.

The target behavior is that cached-prefill cost scales with effective tail
tokens and required compact rows. Avoid fixed ~0.6 s update cost for small
tails.

### 2. Prefix-Reuse-Aware DP Scheduling

Second target. Even when a cached-prefill request is faster locally, a global
forward step can still be gated by another DP rank doing full 8192-token
prefill. Future scheduler work should avoid mixing host-hit cached prefill with
large uncached prefill when that removes the cached-prefill benefit.

### 3. CSA/HCA Workspace Materialization

Secondary target. Slow prefill workspace logs with `flashmla` near zero mean the
cost is gather/dequantize/materialize into the workspace, not attention compute
or CPU scalar metadata. Optimize this after the layer 2 fixed update cost is
under control.

## Recommended Log Checklist

Use the same workload and compare enable/disable KVStore where possible.

Required markers:

- `[cache_op] loadback done`: H2D loadback latency.
- `[cache_op][paged_l2] loadback schedule`: pages, spans, and group split.
- `[DeepSeekV4][slow_attn]`: layer/kind totals and phase split.
- `[DeepSeekV4][slow_layer]`: layer-level attribution.
- `[Scheduler][forward][slow_detail]`: confirm whether `forward_step` dominates.
- `[Scheduler][forward][dp_sync]`: detect DP-rank straggler amplification.
- `Prefill batch`: correlate `#new-token` and `#cached-token`.

Useful one-pass parsing questions:

- Is `loadback done` below the forward slow path by an order of magnitude?
- Does layer 2 CSA still show `indexer_stream_scope` around 0.6 s?
- Is `prefill_backend` near zero or near hundreds of ms?
- Is a cached-prefill request paired with another rank doing 8192-token prefill?
- Are any one-off long tails accompanied by health-check timeouts?

## Guardrails

- Do not add a new `--enable-kvstore` flag. The existing user-facing off switch
  is `--disable-kvstore`.
- Do not reintroduce CSA compressor/indexer overlap without new evidence and a
  clean design for shared metadata reuse.
- Do not add global `torch.cuda.synchronize()` or scheduler-wide stream waits.
- Do not optimize L2 transfer bytes before proving transfer is the current
  dominant cost.
- Keep new diagnostics slow-path-only or debug-level; logs must not flood normal
  runs.
