# DeepSeek V4 Cached-Prefill Mixed-Batch Perf

## Problem

DeepSeek V4 agentic long-context traffic often alternates between decode steps
and cached-prefill tail chunks. With mixed prefill/decode enabled, a single DP
global step can combine decode work on one rank with a large prefill chunk on
another rank. The global step then completes at the pace of the slowest rank, so
decode latency and per-user TPS can suffer even when the local decode work is
small.

The scheduler should still preserve decode priority and overlap scheduling. The
fix should therefore bound only the prefill work admitted into a global step
while decode pressure exists, without adding GPU synchronization or model-specific
cost modeling.

## Final Approach

The event loop now computes a CPU-only mixed-prefill token budget before asking
the scheduler for the next execution plan.

The budget is active only when all of these conditions hold:

- mixed prefill/decode is enabled;
- chunked prefill has a positive configured size;
- at least one DP rank has pending decode work.

When there is no global decode pressure, the budget is disabled and the existing
prefill scheduling behavior remains unchanged.

When decode pressure exists, the budget is:

```text
align_down_to_block_size(chunked_prefill_size / active_prefill_ranks)
```

where `active_prefill_ranks` is the number of DP ranks whose next candidate set
contains real prefill work. The lower bound is one block, so the prefill side
continues to make progress.

This is a fair-share bound, not a DeepSeek-V4-specific heuristic. It uses only
CPU scheduler state:

- `has_decode`;
- `has_prefill`;
- `chunked_prefill_size`;
- `block_size`;
- DP world size.

The only cross-rank operation is the same CPU-side DP all-gather shape used by
the event loop for scheduling metadata. No GPU tensor is read on CPU and no new
CUDA stream synchronization is introduced.

## Scheduler Contract

The C++ scheduler receives `mixed_prefill_token_budget` as an optional limit for
prefill tokens in the next mixed forward plan.

- Negative budget: disabled; keep existing scheduling behavior.
- Non-negative budget: share this budget across local prefill candidates in the
  step.
- Decode candidates keep their existing priority.
- Existing page, batch-size, Mamba slot, and token-budget constraints remain in
  force.

`peek_next_forward_workload()` is intentionally small. It reports whether the
next candidate set has decode and whether it has prefill work. For submitted or
prefetch-done requests, the prefill check uses the existing prefix-match estimate
without touching prefix-cache state.

## Related DeepSeek V4 Optimizations

This branch also keeps the DeepSeek V4 cached-prefill improvements that are
independent of temporary logging:

- CPU metadata sizing for prefill workspaces, avoiding `.item()` GPU syncs in the
  workspace construction path.
- Compact-row bounds for CSA/HCA cache updates in mixed/cached-prefill paths.
- HCA direct slot mapping for pure prefill when the compact row range is known.
- Paged compressed slot mapping and kernel/API support for compact cache-update
  rows.

## Validation

Use the long-context mixed-batch benchmark with `--disable-kvstore` for this
branch when L2 is not part of the test:

```bash
tokenspeed serve --model deepseek-ai/DeepSeek-V4-Flash \
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
  --disable-kvstore
```

```bash
evalscope perf --model deepseek-ai/DeepSeek-V4-Flash \
  --url http://localhost:30101/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path /tmp/agentic_dataset.json \
  --max-tokens 500 \
  --multi-turn \
  --number 6 \
  --parallel 2 \
  --dataset-offset 68 \
  --outputs-dir "$WARMUP_DIR" \
  --extra-args '{"ignore_eos": true}'
```

### Origin Main Baseline

Baseline runs below were collected on `origin/main` after one warmup run, then
two measured runs with:

```bash
evalscope perf --model deepseek-ai/DeepSeek-V4-Flash \
  --url http://localhost:30101/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path /tmp/agentic_dataset.json \
  --max-tokens 500 \
  --multi-turn \
  --number 4 \
  --parallel 2 \
  --dataset-offset 68 \
  --outputs-dir "$WARMUP_DIR" \
  --extra-args '{"ignore_eos": true}'
```

| Metric | Run 1 | Run 2 | Mean |
| --- | ---: | ---: | ---: |
| Test duration (s) | 227.31 | 254.00 | 240.66 |
| Req throughput (req/s) | 0.21 | 0.19 | 0.20 |
| Avg latency (s) | 9.07 | 10.25 | 9.66 |
| TTFT (ms) | 1540.63 | 2087.29 | 1813.96 |
| TPOT (ms) | 15.09 | 16.35 | 15.72 |
| ITL (ms) | 15.07 | 16.41 | 15.74 |
| Output throughput (tok/s) | 105.58 | 94.49 | 100.04 |
| Total throughput (tok/s) | 13923.87 | 12460.52 | 13192.20 |
| KV cache hit rate (%) | 89.30 | 88.63 | 88.97 |
| First-turn TTFT (ms) | 3760.94 | 3759.33 | 3760.14 |
| Subsequent-turn TTFT (ms) | 1338.78 | 1935.28 | 1637.03 |
| Per-trace decode TPS | 66.48 | 63.14 | 64.81 |
| Workload completion tok/s | 105.58 | 94.49 | 100.04 |
| Workload cached prompt tok/s | 12339.18 | 10960.35 | 11649.77 |

Latency percentile baseline:

| Metric | Run | p50 | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Latency (s) | 1 | 7.91 | 11.03 | 11.73 | 11.74 | 11.96 | 11.96 |
| Latency (s) | 2 | 10.51 | 11.26 | 15.45 | 15.67 | 15.73 | 15.73 |
| TTFT (ms) | 1 | 564.64 | 3615.32 | 4200.92 | 4441.05 | 4736.23 | 4736.23 |
| TTFT (ms) | 2 | 999.91 | 3618.53 | 4323.17 | 8143.64 | 8366.13 | 8366.13 |
| TPOT (ms) | 1 | 14.83 | 15.02 | 15.26 | 19.17 | 20.69 | 20.69 |
| TPOT (ms) | 2 | 14.72 | 15.22 | 20.56 | 27.42 | 30.59 | 30.59 |
| ITL (ms) | 1 | 14.60 | 14.71 | 14.81 | 14.88 | 19.08 | 442.30 |
| ITL (ms) | 2 | 14.60 | 14.71 | 14.81 | 14.89 | 29.04 | 4976.73 |

Throughput percentile baseline:

| Metric | Run | p50 | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Output (tok/s) | 1 | 63.44 | 63.97 | 64.19 | 64.34 | 67.27 | 67.27 |
| Output (tok/s) | 2 | 47.68 | 63.50 | 64.04 | 64.14 | 64.37 | 64.37 |
| Total (tok/s) | 1 | 8058.95 | 8866.50 | 9182.70 | 9326.80 | 9887.95 | 9887.95 |
| Total (tok/s) | 2 | 6064.43 | 8786.05 | 9172.55 | 9266.05 | 9364.56 | 9364.56 |
| Decode (tok/s) | 1 | 67.63 | 68.68 | 68.82 | 68.97 | 71.20 | 71.20 |
| Decode (tok/s) | 2 | 68.13 | 68.50 | 68.86 | 68.92 | 71.35 | 71.35 |

Workload throughput baseline:

| Metric (tok/s) | Run 1 overall | Run 2 overall | Mean overall | Run 1 steady | Run 2 steady | Mean steady |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total prompt | 13818.35 | 12366.04 | 13092.20 | 14448.34 | 12763.11 | 13605.73 |
| New prompt | 1479.17 | 1405.68 | 1442.43 | 1171.89 | 1146.85 | 1159.37 |
| Cached prompt | 12339.18 | 10960.35 | 11649.77 | 13276.46 | 11616.26 | 12446.36 |
| Completion | 105.58 | 94.49 | 100.04 | 107.66 | 95.10 | 101.38 |

The two `origin/main` measured runs have visible tail variance. Run 2 shows a
worse TTFT tail (`p99=8366.13ms`) and ITL max (`4976.73ms`) than run 1
(`p99 TTFT=4736.23ms`, `ITL max=442.30ms`). Treat the mean as a convenience
baseline and the run-to-run range as the more useful comparison envelope.

### Optimal Version Comparison

After `6d2dc1e` showed a small regression, the remote benchmark was reset to the
previous optimal version and measured twice with the same `origin/main` command
and warmup method. This optimal version keeps the paged compressed slot mapping
fast path.

Raw summaries are also preserved under:

```text
/private/tmp/tokenspeed-ds-v4-origin-main-baseline-20260624
```

Summary mean comparison:

| Metric | Origin mean | Optimal mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| Test duration (s) | 240.66 | 229.92 | -10.74 | -4.46% |
| Avg latency (s) | 9.66 | 9.26 | -0.40 | -4.12% |
| TTFT (ms) | 1813.96 | 1666.88 | -147.08 | -8.11% |
| TPOT (ms) | 15.72 | 15.22 | -0.51 | -3.21% |
| ITL (ms) | 15.74 | 15.23 | -0.52 | -3.27% |
| Output throughput (tok/s) | 100.04 | 104.40 | +4.37 | +4.36% |
| Total throughput (tok/s) | 13192.20 | 13768.07 | +575.87 | +4.37% |
| First-turn TTFT (ms) | 3760.14 | 3689.64 | -70.50 | -1.87% |
| Subsequent-turn TTFT (ms) | 1637.03 | 1482.99 | -154.04 | -9.41% |
| Per-trace decode TPS | 64.81 | 66.21 | +1.40 | +2.16% |
| KV cache hit rate (%) | 88.97 | 89.34 | +0.37 pp | +0.42% |

Tail-latency mean comparison:

| Metric | Origin mean | Optimal mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| Latency p50 (s) | 9.21 | 7.89 | -1.32 | -14.39% |
| Latency p95 (s) | 13.71 | 12.03 | -1.68 | -12.26% |
| Latency p99 (s) | 13.85 | 12.50 | -1.35 | -9.75% |
| TTFT p50 (ms) | 782.28 | 527.17 | -255.11 | -32.61% |
| TTFT p75 (ms) | 3616.93 | 3658.46 | +41.53 | +1.15% |
| TTFT p90 (ms) | 4262.05 | 4224.52 | -37.53 | -0.88% |
| TTFT p95 (ms) | 6292.35 | 4565.92 | -1726.43 | -27.44% |
| TTFT p99 (ms) | 6551.18 | 5103.31 | -1447.87 | -22.10% |
| TPOT p50 (ms) | 14.78 | 14.71 | -0.07 | -0.47% |
| TPOT p90 (ms) | 17.91 | 15.95 | -1.97 | -10.97% |
| TPOT p95 (ms) | 23.30 | 19.79 | -3.51 | -15.07% |
| TPOT p99 (ms) | 25.64 | 22.91 | -2.74 | -10.67% |
| ITL p50 (ms) | 14.60 | 14.60 | 0.00 | 0.00% |
| ITL p99 (ms) | 24.06 | 24.04 | -0.03 | -0.10% |
| ITL max (ms) | 2709.52 | 1047.11 | -1662.41 | -61.36% |

Throughput mean comparison:

| Metric | Origin mean | Optimal mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| Output throughput (tok/s) | 100.04 | 104.40 | +4.37 | +4.36% |
| Total throughput (tok/s) | 13192.20 | 13768.07 | +575.87 | +4.37% |
| Output p50 (tok/s) | 55.56 | 63.70 | +8.14 | +14.65% |
| Total p50 (tok/s) | 7061.69 | 8141.58 | +1079.89 | +15.29% |
| Decode p50 (tok/s) | 67.88 | 68.08 | +0.20 | +0.29% |
| Workload total prompt overall (tok/s) | 13092.20 | 13663.73 | +571.53 | +4.37% |
| Workload cached prompt overall (tok/s) | 11649.77 | 12206.23 | +556.46 | +4.78% |
| Workload completion overall (tok/s) | 100.04 | 104.40 | +4.37 | +4.36% |
| Workload total prompt steady (tok/s) | 13605.73 | 14239.94 | +634.22 | +4.66% |
| Workload cached prompt steady (tok/s) | 12446.36 | 13091.26 | +644.90 | +5.18% |
| Workload completion steady (tok/s) | 101.38 | 106.11 | +4.73 | +4.66% |

Run stability:

| Metric | Origin runs | Optimal runs | Readout |
| --- | ---: | ---: | --- |
| Output throughput (tok/s) | 105.58 / 94.49 | 103.16 / 105.64 | Optimal does not raise best-run much, but removes the bad-run drop. |
| Output throughput range | 11.09 | 2.48 | Optimal is materially more stable. |
| TTFT p99 (ms) | 4736.23 / 8366.13 | 5792.35 / 4414.27 | Optimal avoids the very bad second-run TTFT tail. |
| ITL max (ms) | 442.30 / 4976.73 | 1167.56 / 926.65 | Optimal removes the multi-second ITL outlier. |
| Subsequent-turn TTFT (ms) | 1338.78 / 1935.28 | 1606.53 / 1359.45 | Optimal has lower mean and tighter spread. |

The strongest evidence for the optimal version is stability rather than peak
throughput. The best `origin/main` run and best optimal run are almost identical
on output throughput (`105.58` vs `105.64 tok/s`), but the optimal version raises
the worse run from `94.49` to `103.16 tok/s` and reduces the TTFT/ITL bad-tail
behavior. Keep `6d2dc1e` out of the final PR unless a separate microbenchmark
can prove that removing the paged mapping fast path is neutral.

Expected behavior:

- decode-only steps are not packed with full 8192-token prefill chunks;
- mixed steps under decode pressure admit a bounded prefill chunk;
- full prefill throughput is unchanged when there is no decode pressure;
- runtime debug logs from the exploration phase are not required for normal PR
  validation.

## Remaining Scope

This patch deliberately does not introduce an adaptive cost model or per-model
work class system. The current fair-share bound is simple, deterministic, and
uses only scheduler-visible CPU metadata. If future traces show stable residual
long-tail stalls inside DeepSeek V4 cached-prefill kernels, those should be
handled as model/kernel optimizations rather than scheduler admission policy.
