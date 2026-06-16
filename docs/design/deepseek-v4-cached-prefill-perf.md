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
- Per-step reuse of SWA and compressed slot mappings when cache group shape is
  identical across layers.
- Compact-row bounds for CSA/HCA cache updates in mixed/cached-prefill paths.
- HCA direct slot mapping for pure prefill when the compact row range is known.
- Kernel/API support for compact cache-update rows.

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
