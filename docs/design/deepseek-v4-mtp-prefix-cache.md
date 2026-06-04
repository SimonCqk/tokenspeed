# DeepSeek V4 MTP Prefix Cache Design

## Scope

This document describes the phase-1 design and implementation for making
DeepSeek V4 MTP speculative decoding use prefix cache without publishing
unaccepted draft state. It is based on `origin/main` at
`11767316d0b77428d5691b140ca3cb23a7eb9cf6`.

The scope is intentionally narrow:

- DeepSeek V4 target verify and MTP draft extend;
- scheduler prefix cache, hybrid prefix cache, paged cache groups, and request
  lifecycle;
- DeepSeek V4 SWA KV, compressed KV, compressor state, and indexer state;
- Mamba/linear-attention checkpoint paths where they intersect with MTP.

Non-goals:

- no new speculative algorithm or forward mode;
- no broad prefix-cache abstraction rewrite;
- no overlap-schedule enablement for speculative + paged-cache groups in the
  first phase.

Items marked "Needs Further Validation" were not fully proven by this phase-1
implementation and should be checked before production rollout.

## Current MTP Speculative Decoding Flow

### Forward modes and scheduling

Key files:

- `python/tokenspeed/runtime/execution/forward_batch_info.py`
- `python/tokenspeed/runtime/execution/model_executor.py`
- `python/tokenspeed/runtime/execution/drafter/eagle.py`
- `python/tokenspeed/runtime/execution/input_buffer.py`
- `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py`
- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/engine/generation_output_processor.py`

`ForwardMode` defines `TARGET_VERIFY` and `DRAFT_EXTEND` as speculative modes.
`ForwardMode.from_num_extends(..., has_drafter=True, use_target_verify=True)`
maps pure decode batches to `TARGET_VERIFY`; `is_speculative()` returns true for
`TARGET_VERIFY` and `DRAFT_EXTEND`.

When speculative decoding is enabled, `EventLoop` configures the scheduler with
`decode_input_tokens = server_args.speculative_num_draft_tokens`. For DeepSeek
V4 this means each decode scheduling event reserves and builds metadata for the
full verify width, not only for the tokens that will be accepted.
`ServerArgs.resolve_config_aliases` sets
`speculative_num_draft_tokens = speculative_num_steps + 1` when the explicit
draft-token count is unset, so `--speculative-num-steps 3` gives a decode input
width of 4.

The runtime currently disables overlap scheduling when both speculative decoding
and paged-cache groups are enabled:

- `python/tokenspeed/runtime/engine/scheduler_utils.py`
  `should_use_overlap_schedule(...)` returns false for
  `speculative_algorithm is not None and paged_cache_groups`.

That is a useful phase-1 constraint: accepted-token commit and cleanup can be
designed without previous-step/current-step overlap races.

### Target verify

`ModelExecutor._forward_step` runs the target model first. For decode-only
speculative batches, the target forward mode is `TARGET_VERIFY`. The target
receives packed tokens from `InputBuffers.future_input_map`, whose width is
`speculative_num_draft_tokens`.

The sampler verifies candidates in:

- `python/tokenspeed/runtime/sampling/backends/greedy.py`
- `python/tokenspeed/runtime/sampling/backends/flashinfer.py`
- `python/tokenspeed/runtime/sampling/backends/flashinfer_full.py`

The `verify(...)` contract returns accepted output tokens and `accept_lengths`.
`accept_lengths` includes the verified/base token. `ModelExecutor` updates
`runtime_states.valid_cache_lengths` by those accepted lengths for the next GPU
runtime step. This is request-local runtime state; it is not a prefix-tree
publish point.

For Mamba-style state, `ModelExecutor._snapshot_mamba_checkpoints` already uses
`accept_lengths` to choose the correct intermediate MTP output slot when a page
boundary is crossed. `hybrid_linear_attn.update_mamba_state_after_mtp_verify`
updates request-local current-input pointers after verify. This is not yet a
DeepSeek V4 SWA/compressor prefix-cache transaction.

### Draft extend

`python/tokenspeed/runtime/execution/drafter/eagle.py` maps MTP onto the `Eagle`
drafter wrapper:

- `_DRAFTER_MAPPING = {"EAGLE3": Eagle, "MTP": Eagle}` in
  `model_executor.py`;
- `Eagle._run_first_step` uses `ForwardMode.DRAFT_EXTEND` when target verify
  mode is active;
- later draft steps run with `ForwardMode.DECODE`;
- `_run_multi_step_decode` computes `cache_start` from
  `runtime_states.valid_cache_lengths + accept_lengths` for decode rows.

DeepSeek V4 MTP draft model code lives in
`python/tokenspeed/runtime/models/deepseek_v4_mtp.py`.
`DeepseekV4ForCausalLMNextN` consumes `captured_hidden_states`, runs
`DeepseekV4MultiTokenPredictorLayer`, and calls the normal
`DeepseekV4DecoderLayer` for the MTP block. Draft tokens therefore use the same
DeepSeek V4 attention/cache helpers as target decode.

This shared helper path was also where the SWA mapping gap appeared:
`DeepseekV4MultiTokenPredictorLayer.forward` called the decoder without passing
a paged SWA slot mapping, so `DeepseekV4Attention` could fall back to
`out_cache_loc`. Phase 1 fixes this by routing both MTP first-step
`DRAFT_EXTEND` and later draft `DECODE` steps through
`_deepseek_v4_swa_slot_mapping`, which uses `v4.swa_kv` paged group tables and
the existing per-request/per-token metadata expansion.

### Idle forward and DP

`EventLoop._dp_sync_and_check` gathers per-rank `[num_tokens, batch_size,
forward_mode]`. `TARGET_VERIFY` is treated as decode/idle-compatible. If this
rank has no work while another rank has decode work, `execute_idle_forward`
participates in collectives.

`ModelExecutor.execute_idle_forward` has two paths:

- CUDA graph path: target graph runs in `TARGET_VERIFY` or `DECODE` mode with
  dummy decode buffers and no sampler/drafter result.
- eager idle path: target runs `ForwardMode.IDLE`, then draft steps run so
  draft-side collectives also execute.

This is not only a validation concern. Active draft step 1+ uses one token per
request and switches DP token counts to the draft batch width. Phase 1 makes
idle draft step 1+ mirror active draft decode by using `global_bs` for
`global_num_tokens` after step 0.

### DeepSeek V4 speculative metadata

Key files:

- `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`
- `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`
- `python/tokenspeed/runtime/models/deepseek_v4.py`

`DeepseekV4AttentionBackend.uses_paged_cache_groups = True`.

For speculative modes, `DeepseekV4AttentionBackend.init_forward_metadata`
requires explicit `num_tokens` or `positions`, creates uniformly packed
per-request token metadata, and normalizes speculative metadata internally to
`ForwardMode.DECODE`. CUDA graph capture/replay preallocates persistent packed
metadata buffers sized by `max_tokens_per_req`, plus per-group paged-cache
block-table buffers and sliding-window base offsets.

The phase-1 runtime fix names the actual call sites: the main model path and
the MTP predictor now both compute paged SWA mapping before calling decoder
layers. `_group_slot_mapping_from_raw` already supports per-request metadata by
expanding values across packed tokens when the token count is a multiple of the
request count, so no separate MTP-only metadata shape is needed for this fix.

## Current Prefix Cache and Hybrid State Flow

### Scheduler publish contract

Key files:

- `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.h`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- `tokenspeed-scheduler/csrc/fsm/forward_events.h`
- `tokenspeed-scheduler/csrc/fsm/forward_states.h`
- `tokenspeed-scheduler/csrc/core/token_container.cpp`

`TokenContainer` is the scheduler truth for request tokens. Python emits
accepted tokens through `make_extend_result_event`; C++ applies
`ExtendResultEvent` by extending the `TokenContainer`. Rejected draft tokens are
not appended.

`OutputProcesser.post_process_forward_op` advances the output-token pointer by
the full speculative stride for decode slots, but emits only accepted `new_ids`.
For decode slots it also emits `UpdateReserveNumTokensEvent(output_length)`, so
the next scheduling event reserves the accepted length instead of the previous
full draft width.

Prefix-tree publish is scheduler-side:

- prefill transitions insert full pages into the KV prefix tree;
- ordinary `Decoding -> Decoding` transitions currently acquire local reserve
  pages for the next decode step but do not publish new accepted pages into the
  radix tree;
- `HybridPrefixCache::CommitChunk` is currently called on prefill paths and the
  first decode scheduled from `PrefillDone`, not on every continuing decode;
- finish and retract insert full accepted KV pages from `TokenContainer`, but
  they need explicit paged-cache snapshot hooks if DeepSeek V4 compressed/state
  group snapshots should be attached at those terminals;
- `GetFullPagedTokens(page_size, except_last)` only returns full token pages.

The intended accepted-token commit point is:

1. GPU target verify produces `accept_lengths`.
2. Python output processing emits accepted `new_ids`.
3. Scheduler applies `ExtendResultEvent`.
4. A new accepted-aware decode/publish hook rewinds request-local visibility to
   the accepted cursor and, when a full page/history-aligned boundary is reached,
   publishes only accepted pages and paged-cache snapshots. Without this new hook,
   continuing decode remains request-local until finish, retract, or another
   existing insertion path.

### Hybrid paged cache

Key files:

- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp`
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h`
- `tokenspeed-scheduler/csrc/resource/allocator/paged_cache_group.cpp`
- `tokenspeed-scheduler/csrc/resource/allocator/paged_cache_group.h`
- `tokenspeed-scheduler/csrc/resource/radix_tree/paged_cache_snapshot.h`
- `tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h`

`HybridPrefixCache` wraps KV prefix cache matching and augments it with Mamba and
paged-cache snapshots. A request table is kept per request and per paged group.

`PagedCacheGroupTable` separates borrowed page ids imported from tree-owned
snapshots from owned pages allocated for this request. It also tracks compact
sliding-window `base_logical_page`, `raw_token_cursor`, and
`committed_prefix_len_tokens`.

`AcquireForRequest(...)` imports borrowed prefix pages on a fresh table, applies
`ReleaseSkipped(...)` for sliding groups, then acquires owned pages up to the
requested target. `PopulateOp(...)` writes compact per-group block tables and
base offsets into the forward operation.

`CommitChunk(...)` publishes only aligned full segments. It uses the LCM of
history-family `RawTokensPerPage()` as the history alignment. For required state
groups it checkpoints the sliding-window state segment at the same target. It
can adopt an existing snapshot and attaches complete snapshots to radix-tree
nodes. `OnKVEvict(...)` detaches paged-cache snapshots when the corresponding
KV tree node is evicted.

### DeepSeek V4 cache groups

Key files:

- `python/tokenspeed/runtime/configs/deepseek_v4_cache_spec.py`
- `python/tokenspeed/runtime/configs/paged_cache_spec.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`

DeepSeek V4 defines multiple paged-cache groups:

| Group | Family | Retention | Role |
| --- | --- | --- | --- |
| `v4.swa_kv` | State | Sliding window | SWA KV tail |
| `v4.c{ratio}a.compressor_state` | State | Sliding window | compressor state |
| `v4.c{ratio}a.compressed_kv` | History | Full history | compressed KV history |
| `v4.c4a.indexer_compressor_state` | State | Sliding window | c4 indexer state |

`DeepseekV4TokenToKVPool.prefix_cache_required_group_ids` currently returns
history-family group ids. State groups are still registered as paged-cache
groups and can be terminal continuation groups in `HybridPrefixCache`, but the
pool-side required set is history-only.

The target and draft runtime paths are separate physical pool instances today
when `draft_attn_config` is present: registry construction creates a target
`DeepseekV4TokenToKVPool` and a draft `DeepseekV4TokenToKVPool`, and the drafter
receives `draft_token_to_kv_pool`. Keeping those two pools separate is the
right phase-1 boundary because target verify and draft prediction write
different model/layer paths. MTP layers still use the same pool class, group
specs, page-size contract, and compact draft cache slots by passing
`cache_layer_index=local_idx` in
`python/tokenspeed/runtime/models/deepseek_v4_mtp.py`.

The design goal is therefore not to create a third independent MTP cache pool,
and not to merge target and draft into one physical pool. It reuses the existing
DeepSeek V4 pool/group contract, scheduler page-id semantics, and
request-local paged-cache table machinery wherever the layer mapping allows it.
Prefix-cache coverage is added by extending accepted scheduler publication and
DeepSeek V4 metadata mapping for the relevant MTP `cache_layer_index` values,
not by inventing an MTP-only pool model.

Needs Further Validation: confirm whether DeepSeek V4 MTP prefix reuse should
require state-family group completeness for all hits, or only for terminal
continuation and state recovery. The current `HybridPrefixCache` policy can
fall back to root when transport-only state groups prevent safe continuation.

### State recovery and cleanup

Retraction and finish are handled by the C++ FSM:

- `ScheduleRetractEvent` inserts full accepted pages, saves Mamba checkpoint or
  working state to the prefix tree, and resets request-local Mamba ownership.
- `ScheduleDecodeFromRetractedEvent` matches with
  `MatchIntent::StateRecovery`, reacquires request-local tables from matched
  tree snapshots, and sets `hist_token_len = request->TokenSize() - 1`.
- `AbortEvent` transitions live states to `Finished`, letting destructors and
  request release paths drop local ownership.
- `HybridPrefixCache::ReleaseRequest` drops request paged-cache tables, clears
  borrowed ids, and returns owned pages through RAII.

Needs Further Validation: confirm every abort/cancel/removal path invokes
`HybridPrefixCache::ReleaseRequest` for DeepSeek V4 paged-cache tables. The
release surface exists, but this exploration did not fully trace every removal
path.

## Pre-Phase-1 Gaps and Remaining Risks

Before phase 1, the code had the right building blocks, but not a complete
accepted transaction for DeepSeek V4 MTP. Phase 1 addresses accepted decode
publication, MTP SWA mapping, and DP idle draft sizing; state-family reuse and
aggressive tail-page release remain conservative.

1. Decode admission reserves full speculative width.

   `Scheduler::scheduleDecode` sets `target = request->TokenSize() +
   config_.decode_input_tokens`. In speculative mode, `decode_input_tokens` is
   the full draft width. `HybridPrefixCache::AcquireForRequest` then advances
   paged-cache group tables to that full target.

2. The accepted truth is shorter and arrives later.

   Python emits only accepted `new_ids`. `TokenContainer` only grows by accepted
   tokens. A partial accept leaves request-local paged-cache tables and
   GPU-written state potentially covering a wider range than the scheduler
   token truth.

3. Tree publish is accepted-length based, but temporary state visibility is not
   explicit.

   `CommitChunk` publishes up to accepted radix-tree terminal depth. That
   prevents rejected tokens from becoming tree tokens. However, there is no
   explicit MTP rewind surface for DeepSeek V4 state-family groups after
   partial accept.

4. MTP draft SWA mapping had to use paged-group slots.

   The model-side fallback from `swa_slot_mapping` to `out_cache_loc` on the MTP
   draft path bypassed the proper DeepSeek V4 paged SWA mapping when paged SWA
   groups were active. Phase 1 now computes the same paged SWA mapping for main
   and MTP paths.

5. Prefix snapshot reads must be leased, not raw tree reads.

   `HybridPrefixCache::Match` and `AcquireForRequest` make borrowed pages safe
   by attaching them to request-local tables while tree nodes are pinned through
   node refs. Any MTP-specific direct tree lookup during draft would need the
   same ownership/refcount boundary.

6. CUDA graph and idle paths need fixed-shape metadata.

   DeepSeek V4 CUDA graph replay uses preallocated packed token metadata and
   per-group block-table buffers. Any design that adds temporary state must
   update in-place tensors within capture-time bounds and mark padding or
   rejected slots invalid.

## Evaluation of the Proposed Three-Point Idea

| Proposal | Assessment | Required correction |
| --- | --- | --- |
| MTP fixed-depth draft reads and reuses current request cache from the prefix tree | Directionally correct for committed prefix state, but draft should not re-read the tree directly. | Scheduler should import a read-only prefix snapshot into the request table before forward. Target verify and draft reuse that request-local leased table. |
| MTP temporary SWA/compressor state is stored temporarily | Correct and necessary. | The accepted cursor is the scheduler `TokenContainer` plus runtime sequence length, not the full speculative acquisition width. Rejected tail state is kept as overwrite-only request-local scratch while the request continues. |
| Accepted tokens publish or update the tree after confirmation | Correct, but publish must happen at the scheduler accepted-token commit point. | Publish after `ExtendResultEvent` through the accepted decode publish hook. Partial accepts below alignment stay request-local until the next full history alignment, finish, or retract. |

The core correction is: do not add a draft-time tree mutation. Add an accepted
transaction boundary between full-width speculative allocation and accepted-width
tree publication.

## Proposed Design

### Principle

Treat every speculative decode step as a request-local transaction:

- read prefix snapshots through scheduler-owned borrowed pages;
- write target verify and draft outputs into request-local acquired pages;
- commit only the accepted prefix to scheduler token truth and tree snapshots;
- rewind or mask rejected tail state before it can influence prefix
  matching or state recovery. In the normal partial-accept path this is a
  logical rewind of accepted length/sequence length, with the next draft/verify
  overwriting reserved slots, not a prefix-cache eviction.

### Data structures

Use existing request paged-cache tables as the primary ownership object. Phase 1
does not need a new speculative transaction object or a second table cursor:
`TokenContainer` remains the accepted truth, while the existing request-local
tables may stay acquired up to the fixed verify/draft width. Rejected tail slots
are logical scratch for the next step and are overwritten; they are not visible
to prefix-tree publication because publication reads accepted tokens from
`TokenContainer`.

The concrete phase-1 scheduler change is an accepted-aware `Decoding`
`ExtendResultEvent` transition. After accepted tokens are appended, it:

- computes full accepted pages with `GetFullPagedTokens(..., except_last=true)`;
- compares against the matched prefix pages already present at the request's
  device-node ref;
- inserts only newly accepted full pages with `InsertHybridCache`;
- calls `HybridPrefixCache::CommitChunk(request_id, terminal_node)` so DeepSeek
  V4 history-family paged-cache snapshots are attached only at accepted,
  history-aligned terminals.

This preserves the existing acquire/populate behavior needed by fixed-width
verify/draft computation while adding continuous accepted-prefix reuse during
decode. Page release beyond accepted length remains cleanup/capacity work for
abort, finish, retract, or future optimization; it is not the normal
partial-accept path.

Python/runtime additions implemented in phase 1:

- derive accepted target from `TokenContainer` after `ExtendResult`;
- make DeepSeek V4 main and MTP paths use `_deepseek_v4_swa_slot_mapping` so
  SWA slot mapping uses `_group_slot_mapping_from_raw` and the paged
  `v4.swa_kv` table, not the `out_cache_loc` fallback;
- make DP idle draft step 1+ use `global_bs` as `global_num_tokens`, matching
  active multi-step draft decode.

No new `ForwardMode` is needed.

### Lifecycle

1. Prefix match and read-only snapshot import.

   `HybridPrefixCache::Match` returns the deepest usable KV and paged-cache
   snapshot. `AcquireForRequest` imports matched pages as borrowed pages on
   fresh request tables. MTP draft code must reuse these request tables.

2. Speculative acquire.

   Decode scheduling acquires tables up to `base + draft_width`, matching the
   current scheduler behavior. Sliding-window groups may release old front
   pages, but must not release pages required by the current forward.

3. Target verify.

   Target `TARGET_VERIFY` reads prefix/history from request tables and writes
   target outputs for the speculative token pack. Existing Mamba accepted-state
   updates continue to use `accept_lengths`.

4. Draft extend.

   MTP `DRAFT_EXTEND` and later draft decode steps use the same request-local
   tables and DeepSeek V4 per-token metadata. Draft writes are temporary until
   accepted. Draft steps must not call `Match` or mutate `TreeNode` snapshots.

5. Accepted commit.

   `OutputProcesser` emits accepted `new_ids` and reserve hint `output_length`.
   Scheduler applies `ExtendResultEvent`; the `Decoding` overload publishes
   accepted full pages and commits eligible paged-cache snapshots immediately.
   `TokenContainer` is the source of accepted length.

6. Publish.

   Publishing is separate from accepted scheduler truth. Phase 1 adds an
   explicit decode accepted-page insertion path plus a `CommitChunk` hook after
   `ExtendResultEvent`. Finish and retract continue to insert accepted KV pages.
   Only full history-aligned targets are attached as `PagedCacheSnapshot`.
   `commitTerminalContinuationSnapshot` may attach terminal state only when the
   accepted terminal has complete history and all continuation state groups are
   available.

7. Rewind and cleanup.

   Any request-local state beyond accepted length is logically rewound by the
   accepted `TokenContainer` length and runtime sequence lengths, then retained
   as overwrite-only owned scratch for the next decode, or masked invalid for
   CUDA graph padding/rejected tokens. It must never be converted into borrowed
   pages or stored in a tree snapshot. Releasing pages is a cleanup/capacity
   fallback, not the normal accepted-token rewind.

### Accepted token commit point

The authoritative accepted point is scheduler-side, after
`ExtendResultEvent`. GPU-side `valid_cache_lengths` can lead the scheduler by
one iteration and is only runtime-local.

Implementation detail:

- Python may continue to emit `UpdateReserveNumTokensEvent(output_length)`.
- C++ uses accepted `TokenContainer::Size()` after `ExtendResult` as the
  accepted raw target.
- The accepted publish hook runs before the next `scheduleDecode` admission, so
  prefix-tree visibility is accepted-only while retained speculative tail slots
  remain overwrite-only scratch. Page release is only a fallback when the
  request finishes, aborts, or capacity pressure requires it.
- Snapshot publication should be triggered only from accepted scheduler truth,
  not from GPU-local `valid_cache_lengths` or speculative acquired width.

### Rewind behavior

Partial accept:

- accepted length includes the verified/base token;
- publish only accepted tokens;
- rewind `seq_lens` and accepted cursor to `base + accepted_length`;
- stop referencing `[base + accepted_length, base + draft_width)` and let the
  next draft/verify overwrite those reserved slots when the request continues;
- do not mark the rejected tail for prefix-cache eviction, because it was never
  tree-published;
- keep Mamba current input at accepted slot, as existing Mamba path already
  does;
- ensure DeepSeek V4 SWA/compressor/indexer state for rejected slots is not
  published to prefix snapshots; future request-local forwards either ignore it
  by sequence length or overwrite it.

Request cancel or abort:

- stale `ExtendResultEvent` and reserve hints for `Finished` already no-op;
- `ReleaseRequest` must drop all paged-cache request tables and return owned
  pages.

Finish:

- finish inserts full accepted pages excluding the tail where appropriate;
- terminal continuation state can publish only accepted terminal state;
- rejected tail pages must be released before or during finish cleanup.

Retract:

- before retraction, copy accepted draft Mamba state to working state through
  existing `flush_mamba_draft_to_working_on_retract`;
- insert only full accepted pages from `TokenContainer`;
- state recovery must match accepted prefix only;
- rejected DeepSeek V4 state must be absent from `MatchIntent::StateRecovery`.

Tree eviction:

- borrowed prefix pages are safe only while owning tree nodes are pinned;
- snapshot pruning must skip pinned nodes as current
  `tryPrunePagedCacheSnapshot` does;
- draft code must not hold raw `TreeNode*` or page ids outside scheduler
  request-table ownership.

Multi-rank and DP:

- all ranks derive identical request order and forward modes from the scheduler
  plan;
- cleanup events are deterministic and keyed by request id/epoch;
- idle ranks do not require real request snapshots, but must run compatible
  target and draft metadata for collectives.

CUDA graph:

- speculative packed metadata stays fixed shape;
- per-group block tables and base offsets refresh in-place;
- padding and rejected tokens use `is_valid_token` and `-1` slots;
- no allocation is allowed during capture.

Idle draft:

- idle forward must not mutate prefix-cache request tables;
- draft metadata for idle ranks uses dummy or empty tables that satisfy shape
  contracts;
- step 1+ draft metadata uses the same global token/batch semantics as active
  ranks.

## Invariants

| Invariant | Enforced at |
| --- | --- |
| Only accepted tokens mutate `TokenContainer`. | `OutputProcesser.post_process_forward_op` emits accepted `new_ids`; `ExtendResultEvent` applies them. |
| Prefix-tree snapshots never include rejected draft tokens. | `CommitChunk`, `FinishEvent`, and `ScheduleRetractEvent` operate on accepted `TokenContainer` full pages only. |
| Draft reads prefix cache only through request-local borrowed pages. | `HybridPrefixCache::Match` plus `AcquireForRequest`; no draft-time direct tree match. |
| Borrowed pages are never returned to group allocators by request cleanup. | `PagedCacheGroupTable::ReleaseAll` drops borrowed ids and RAII returns only owned pages. |
| State-family snapshots are complete only at accepted aligned targets. | `CommitChunk`, `CheckpointStateToSnapshot`, and `RefreshPagedCacheSnapshotCompleteness`. |
| Sliding-window base offsets match compact block tables. | `PagedCacheGroupTable::BaseLogicalPage`, `PopulateOp`, and DeepSeek V4 metadata split helpers. |
| CUDA graph replay never indexes invalid padded or rejected tokens. | DeepSeek V4 `is_valid_token`, `-1` slot mappings, and preallocated graph metadata buffers. |
| DP cleanup and publish are deterministic across ranks. | Scheduler request ordering, request-id keyed events, and stale-event no-ops for `Finished`. |
| Request abort/finish releases speculative owned pages. | Existing `ReleaseRequest` paths and allocator RAII. |

## Failure Modes and Mitigations

| Failure mode | Risk | Mitigation |
| --- | --- | --- |
| Partial accept publishes rejected state | Later request reuses invalid SWA/compressor state. | Accepted cursor comes from `TokenContainer` and runtime sequence length; `CommitChunk` only after accepted `TokenContainer`; rejected tail is not tree-published. |
| Partial accept leaves owned tail pages allocated forever | Paged-cache group leak and capacity collapse. | Retain tail pages as overwrite-only scratch in the normal path; abort/finish/capacity fallback releases pages; metrics assert pages return eventually. |
| MTP draft SWA fallback writes to wrong group | Incorrect SWA state when paged SWA groups active. | Main and MTP paths use `_deepseek_v4_swa_slot_mapping` and paged SWA block tables. |
| Request cancel races with accepted cleanup | Stale event mutates finished request. | No-op on `Finished`, mirroring existing stale `ExtendResultEvent` behavior. |
| Tree eviction detaches snapshot while draft uses it | Use-after-free page ids. | Only borrowed request-table pages are used; tree nodes remain pinned by request refs during active use. |
| State recovery restores rejected tail | Retracted request resumes from invalid state. | Retraction inserts only accepted pages and accepted terminal state; rejected tail is not in prefix snapshots. |
| Multi-rank idle draft metadata mismatch | NCCL hang or wrong collective shape. | Keep phase 1 non-overlap; add DP idle tests for target verify plus draft steps with paged groups. |
| CUDA graph capture sees dynamic allocation | Capture failure or replay corruption. | Preallocate max-width metadata; refresh in-place; use invalid masks. |
| Transport-only state blocks reuse | Prefix hit falls back to root, reducing benefit. | Preserve current conservative fallback; later phase may expand continuation snapshot policy. |

## Metrics and Debug Tracing

Add metrics and debug logs behind normal metrics/debug gates:

- MTP prefix-cache decode attempts;
- matched history tokens for MTP target verify;
- accepted length, draft width, and partial-accept count;
- rewind/rejected-tail token count;
- per-group speculative pages acquired, committed, released;
- per-group failed allocation count during speculative decode;
- snapshot publish count and terminal continuation publish count;
- snapshot prune count blocked by active request refs;
- CUDA graph fallback or replay metadata refresh errors;
- idle draft iterations under MTP prefix-cache mode.

Trace fields should include request id, request pool index, forward mode, base
raw tokens, accepted target, speculative target, group id, base logical page,
and page count.

## Configuration

Phase 1 uses the existing prefix-cache and speculative-decoding configuration
surfaces:

- require DeepSeek V4, `speculative_algorithm=MTP`, target-verify mode,
  prefix cache, and paged-cache groups;
- use `--speculative-algorithm MTP --speculative-num-steps 3` as the baseline
  validation shape unless a test explicitly varies draft depth;
- use `--enable-metrics` in performance validation so `benchmark_summary.json`
  or the output summary can track `Decoded Tok/Iter` and speculative accept
  rate;
- keep overlap schedule disabled when speculative + paged-cache groups;
- keep prefix-cache fallback conservative when state-family completeness is
  missing.

## Implementation Phases

### Phase 0: Guardrails and tests first

- Add scheduler unit tests that prove rejected speculative tails do not publish.
- Add runtime metadata tests for DeepSeek V4 MTP draft per-token
  `token_to_req_indices` with paged SWA groups.
- Add leak/capacity tests for partial accept and cancel.

### Phase 1: Minimal behavior alignment

Goal: make DeepSeek V4 MTP prefix cache correct in non-overlap mode without new
forward modes or broad abstractions.

Implemented work:

- add an accepted-page publish path for continuing decode after
  `ExtendResultEvent`;
- commit eligible DeepSeek V4 history-family paged-cache snapshots from the
  accepted decode terminal with `HybridPrefixCache::CommitChunk`;
- keep speculative tail pages request-local and overwrite-only instead of
  tree-published;
- fix DeepSeek V4 MTP draft SWA slot mapping so it uses paged `v4.swa_kv`
  tables and existing request-index expansion;
- fix idle DP draft step 1+ to mirror active draft decode
  `global_num_tokens`/`global_bs` semantics;
- keep tree publish accepted-only; do not rely on existing continuing decode to
  call `CommitChunk`, because the new `ExtendResultEvent` path does it
  explicitly;
- keep state-family reuse conservative, falling back to root when continuation
  state is incomplete.

This phase should not add a new speculative mode and should not re-enable
overlap scheduling.

### Phase 2: State-family continuation quality

- Decide whether DeepSeek V4 prefix hits require SWA/compressor state
  completeness at all reuse points or only terminal continuation/state recovery.
- Extend `HybridPrefixCache` snapshot completeness policy if history-only hits
  are correct but lower value.
- Add focused tests for `v4.swa_kv`, compressor state, compressed KV, and
  indexer state combinations.

### Phase 3: DP, idle, CUDA graph hardening

- Add DP tests where one rank is idle and another rank runs MTP verify/draft
  with paged-cache groups.
- Verify CUDA graph capture/replay with packed target verify and draft extend
  metadata, compact sliding tables, and invalid padding tokens.
- Add trace-based checks for no dynamic allocation during capture.

### Phase 4: Optional overlap schedule

- Revisit overlap scheduling only after phase 1-3 invariants are proven.
- Add epoch-based stale cleanup for previous-step commit versus current-step
  dispatch.
- Prove grammar, abort, finish, and retract interactions under overlap.

## Test Plan

Scheduler C++ tests:

- partial accept shorter than draft width does not attach snapshots beyond the
  accepted token depth;
- continuing decode publishes accepted full pages through the new accepted-page
  hook;
- full accept crossing history alignment publishes exactly one complete
  snapshot;
- rejected tail slots are rewound or marked overwrite-only and do not reduce
  available pages permanently after request cleanup;
- request cancel after speculative acquire releases request paged-cache tables;
- finish after partial accept publishes only accepted full pages and accepted
  terminal continuation paged-cache state when the design expects finish-time
  prefix reuse;
- retract after partial accept recovers accepted prefix only;
- tree eviction cannot prune a snapshot pinned by an active borrowed request;
- state-only starvation prunes state snapshots without corrupting history reuse.

Python/runtime metadata tests:

- DeepSeek V4 MTP SWA slot mapping expands per-request request indices when
  packed token count is a multiple of request count;
- MTP draft SWA slot mapping uses `v4.swa_kv` block tables and base offsets,
  not `out_cache_loc` fallback, when paged SWA groups are active;
- idle draft step 1+ uses `global_bs` as `global_num_tokens`;
- CUDA graph replay requires explicit speculative `num_tokens`, refreshes
  compact per-group tables in-place, and masks padding tokens;
- decode compressed slot mappings refresh after `advance_draft_forward_metadata`.

End-to-end runtime tests:

- DeepSeek V4 MTP with prefix cache off vs on returns identical tokens on a
  deterministic small prompt set;
- second request sharing a prefix reports prefix-cache hit tokens for MTP target
  verify only after accepted pages and required paged-cache snapshots have been
  published; otherwise the test should assert the conservative fallback;
- forced partial accept does not leak SWA/compressor state into the next request;
- cancel, finish, and retract paths release speculative pages;
- DP run with one idle rank completes without collective shape mismatch.

Operational validation:

- enable debug metrics and verify acquired, accepted, rewound, and published
  token counts balance per request;
- stress with low paged-cache group capacity to force snapshot pruning;
- run with CUDA graph enabled and disabled.

## Open Validation Points

- Needs Further Validation: whether all DeepSeek V4 abort/removal paths call
  `HybridPrefixCache::ReleaseRequest` before request state disappears.
- Needs Further Validation: exact state-family completeness required for safe
  DeepSeek V4 SWA/compressor reuse under MTP.
- Needs Further Validation: whether retaining rejected tail pages as
  overwrite-only scratch is worth the complexity under capacity pressure, or
  whether some paths should eagerly release them after partial accept.
