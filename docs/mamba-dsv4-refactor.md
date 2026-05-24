# tokenspeed-scheduler Hybrid Cache Refactor Plan

> Finalized design note for converging Mamba/GDN-style state cache and
> DeepSeek-V4 grouped cache in `tokenspeed-scheduler`.
>
> Target families:
> - Standard MHA / MLA token-page cache.
> - DeepSeek-V4 CSA / HCA compressed history plus SWA / tail state.
> - Qwen 3.5 / GDN / Mamba-like recurrent state plus conv state.
> - Future hybrid variants with page-like and state-like cache families.
>
> Baseline: tokenspeed-scheduler `main`, plus the in-flight
> `feat/v4-prefix-cache` bridge as a prototype to preserve and generalize.

---

## 1. Goal

Refactor `tokenspeed-scheduler/csrc/resource/` so the scheduler critical path is
model-neutral. `Scheduler` should call one `CacheCoordinator` facade; per-model
differences belong in registered `CacheResourceSpec` records and internal
`CacheFamilyPolicy` implementations.

The scheduler should optimize for the **longest recoverable prefix**, not the
longest equal-token prefix. A token prefix is usable only if every required
cache family can either attach an exact reusable slice, restore an aligned
checkpoint, or provide an explicit replay plan that rebuilds the missing state
before normal execution resumes.

Design principle: **converge the scheduler-facing substrate, not the cache
semantics**. Cache families may share integer resource ids, RAII ownership,
per-request facades, TreeNode attachments, commit-boundary plumbing, and
coordinator control flow. Allocator geometry, recoverability, split behavior,
and publication policy remain family-specific.

## 2. Non-Goals

- Do not force all cache types into one KV page model.
- Do not collapse all allocators into one semantics-blind allocator.
- Do not make `TreeNode` aware of model names such as Mamba or DeepSeek-V4.
  Model-specific names live in spec ids.
- Do not publish auxiliary / window / tail state to the global prefix index by
  default. They stay request-local unless an explicit recovery contract exists.
- Do not change forward-op tensor ABI in PRs 1-7. Existing Mamba fields and
  flattened per-group block tables remain compatibility views until the worker
  ABI is deliberately migrated.
- Do not introduce per-token or per-layer virtual dispatch on the scheduler hot
  path. A bounded startup-built loop over active families is acceptable.

## 3. Model Semantics To Preserve

DeepSeek-V4 and Mamba/GDN layouts share a State/Classical split, but the
shareable object differs.

| Aspect | DeepSeek-V4 CSA/HCA/SWA | Qwen 3.5 / GDN / Mamba-like |
|---|---|---|
| Classical history | CSA/HCA compressed KV blocks. CSA has main compressed KV plus indexer KV; HCA has main compressed KV. | Standard token KV pages only on full-attention layers. |
| Sequence state | SWA window plus uncompressed compression tail. | Recurrent state plus conv ring buffer. |
| Reuse cap | Last complete compression block / LCM boundary. Incomplete tails are replay work. | Last aligned checkpoint (`mamba_cache_chunk_size` / FLA chunk multiple), plus full-attention page hits. |
| Shareable unit | Canonical compressed-history bundle; optional SWA state only with an explicit policy. | Atomic recurrent+conv checkpoint; conv and recurrent state must not be split. |
| Request-local unit | SWA/tail state unless checkpointed. | In-flight working state. |

Recoverability mapping:

- `Recoverability::Exact`: standard KV pages, V4 compressed history, Qwen
  full-attention pages.
- `Recoverability::AlignedCheckpoint`: Mamba/GDN state checkpoints and optional
  V4 SWA periodic checkpoints.
- `Recoverability::WindowRepairable`: V4 SWA zero-cache style rebuild.
- `Recoverability::RequestLocalOnly`: scratch state that never enters the global
  prefix index.

## 4. Current TokenSpeed State

Current `main` has three scheduler-visible cache concepts:

- Standard KV: `Scheduler` owns `KVPrefixCache`; normal KV pages are inserted
  into the radix tree.
- Mamba/GDN: `HybridPrefixCache` wraps `KVPrefixCache` for Mamba-aware matching.
  `MambaChunkAllocator` allocates integer state slots; `LocalMambaAllocator`
  keeps one working slot and one checkpoint slot per request. `InsertMamba`
  attaches a checkpoint slot to a block-aligned `TreeNode`.
- DeepSeek-V4 paged groups: `PagedCacheGroupTable` is per `(request, group)` and
  owns compact `OwnedPages`, `raw_token_cursor_`, and `base_logical_page_`.
  Scheduler helpers allocate, compact, and flatten these request-local tables
  into `paged_cache_block_tables` and `paged_cache_block_table_base_offsets`.

The important current-state boundary: on `main`, V4 paged-cache groups are not
tree-attached prefix-cache entries. They are a parallel request-local address
space keyed by `group_id`.

Current scheduler-facing structure on `main`:

```text
Scheduler
  |
  +-- KVPrefixCache
  |     |
  |     +-- RadixTree
  |           |
  |           +-- TreeNode
  |                 |
  |                 +-- DeviceResource / HostResource     (standard KV pages)
  |                 `-- MambaSlot                         (Mamba checkpoint)
  |
  +-- Mamba side
  |     |
  |     +-- MambaChunkAllocator                           (slot allocator)
  |     `-- LocalMambaAllocator per request
  |           |
  |           +-- working slot
  |           `-- checkpoint slot
  |
  `-- DeepSeek-V4 paged-cache-group side
        |
        +-- PagedCacheGroupAllocator[group_id]            (page allocator)
        `-- PagedCacheGroupTable[request_id][group_id]
              |
              +-- OwnedPages                              (request-owned only)
              +-- raw_token_cursor_
              `-- base_logical_page_
```

Current critical path on `main`:

```text
Scheduler::newForwardOperation
  |
  +-- Match
  |     |
  |     +-- KVPrefixCache::Match
  |     `-- HybridPrefixCache::augmentMatch               (Mamba checkpoint)
  |
  +-- Admission
  |     |
  |     +-- KVPrefixCache::EnsureCapacityByEvict
  |     +-- HybridPrefixCache::EnsureMambaCapacityByEvict
  |     `-- Scheduler::checkPagedCacheGroupAdmission       (V4 request-local)
  |
  +-- Apply FSM event
  |
  +-- Cache state mutation
  |     |
  |     +-- Scheduler::acquirePagedCachePagesForRequest    (ReleaseSkipped + Acquire)
  |     `-- Scheduler::populatePagedCachePagesForOp        (forward metadata)
  |
  `-- Finish / retract cleanup
        |
        +-- KV node refs / cache-op tracking
        +-- Mamba slot release
        `-- PagedCacheGroupTable::ReleaseAll
```

The in-flight `feat/v4-prefix-cache` work should be treated as a Phase-0 bridge,
not the final abstraction. Preserve its useful semantics: opt-in switch,
History/State completeness split, state-only prune, split guards,
borrowed-prefix import ordering, and RAII ownership of pages moved from
request-local tables into tree-node attachments.

## 5. Target Architecture

`Scheduler` sees only five coordinator operations:

| Operation | Scheduler-facing meaning |
|---|---|
| `MatchPrefix` | Produce a `RecoveryPlan`: exact hits, checkpoint hits, and replay ranges. |
| `Admit` | Decide whether the next step can allocate, must evict, or must reject. |
| `StepCommit` | Publish complete pages, compressed bundles, or eligible checkpoints. |
| `FinishRequest` | Release request-local state while leaving shared attachments managed by refcount/LRU. |
| `Statistics` | Report family-level hit, replay, allocation, and eviction counters. |

`CacheCoordinator` owns:

- `FamilyRegistry`: startup-registered `CacheResourceSpec` records.
- `RadixTree`: token prefix tree with family slots on each node.
- Existing allocators: `PageAllocator`, `PagedCacheGroupAllocator`,
  `MambaChunkAllocator`, and their request-local users stay resource-specific
  in this refactor.
- The mutation protocol for request-local cache state: admission, detach,
  release, tree attach, finish cleanup, and atomic cohort transitions.
- Startup-built active-family policy arrays for match, admit, commit, evict,
  and finish phases.

Expected simplification:

- `scheduler.cpp` no longer branches over KV vs Mamba vs V4 paged groups.
- `MatchResult::mamba_*` and existing paged-cache metadata become compatibility
  views derived from `RecoveryPlan`.
- `Scheduler` keeps a `RequestCacheContext` per request. Flatten reads
  snapshots from that context; mutating operations pass the context by reference
  to `CacheCoordinator`.
- There is no separate indirection registry. `RequestCacheContext` itself is
  the request-local API surface; ownership-changing methods still live on
  `CacheCoordinator`.
- `PagedCacheSnapshot` becomes a bridge representation of generic
  `TreeAttachment` slots, with V4 roles encoded by spec id.

The wrong abstraction is a universal allocator. The right shared substrate is:
per-request state shape, explicit commit boundaries, radix-node attachments,
recovery/admission plans, and one coordinator-owned control-flow order.

Target scheduler-facing structure:

```text
Scheduler
  |
  +-- RequestCacheContext[request_id]
  |     |
  |     +-- page-array state entries
  |     `-- checkpoint-pair state entries
  |           |
  |           +-- read-only snapshot helpers
  |           `-- no tree attach / detach authority
  |
  +-- Forward metadata flattening step
  |     |
  |     `-- asks RequestCacheContext for read-only snapshots
  |         no detach, release, or tree attach
  |
  `-- CacheCoordinator
        |
        +-- FamilyRegistry
        |     |
        |     `-- CacheResourceSpec[FamilyId]
        |           |
        |           +-- family: TokenPage / CompressedPage /
        |           |           SlidingWindowState / CompressionTailState /
        |           |           RecurrentState / ConvState
        |           `-- attachment_kind: TokenPageAttachment /
        |                       CompressedBlockAttachment /
        |                       SlidingWindowStateAttachment /
        |                       StateSlotAttachment /
        |                       NoneForRequestLocal
        |
        +-- RadixTree
        |     |
        |     `-- TreeNode
        |           |
        |           `-- slots[FamilyId] -> TreeAttachment
        |                 (shape follows spec.attachment_kind)
        |                 |
        |                 +-- TokenPageAttachment
        |                 +-- CompressedBlockAttachment
        |                 +-- SlidingWindowStateAttachment
        |                 `-- StateSlotAttachment
        |
        +-- Existing allocators
        |     |
        |     +-- PageAllocator / LocalKVAllocator
        |     +-- PagedCacheGroupAllocator / PagedCacheGroupTable
        |     `-- MambaChunkAllocator / LocalMambaAllocator
        |
        `-- CacheFamilyPolicy[FamilyId]
              |
              +-- Match / ComputeSlice
              +-- ComputeDemand
              +-- OnCommit
              +-- OnEvict
              `-- OnFinish
```

Target critical path:

```text
Scheduler::newForwardOperation
  |
  +-- RecoveryPlan plan = CacheCoordinator::MatchPrefix(request_key)
  |     |
  |     +-- walk the token radix tree once
  |     +-- validate each active family against the terminal node
  |     |     |
  |     |     +-- exact page / compressed-bundle hit
  |     |     +-- aligned recurrent+conv checkpoint hit
  |     |     `-- window/tail replay plan
  |     |
  |     `-- return slices + replay ranges + execution resume point
  |
  +-- AdmissionVerdict verdict = CacheCoordinator::Admit(ctx, next_step)
  |     |
  |     +-- compute new demand by family
  |     +-- subtract borrowed_prefix_units
  |     +-- credit sliding releasable_units
  |     `-- evict/prune through family policies if needed
  |
  +-- Apply FSM event
  |
  +-- CacheCoordinator::StepCommit(ctx, plan, step_tokens)
  |     |
  |     +-- attach complete canonical bundles at commit boundaries
  |     +-- publish eligible checkpoints
  |     `-- keep RequestLocalOnly state local
  |
  +-- ForwardMetadata::Flatten(ctx, plan)
  |     |
  |     +-- old KV block tables
  |     +-- old mamba_pool_indices / cow_src / branching seqlen
  |     `-- old V4 paged_cache_block_tables / base_offsets
  |
  `-- CacheCoordinator::FinishRequest(ctx)
        |
        `-- release request-local state; shared TreeNode attachments remain
```

## 6. Core Data Model

Only the stable scheduler-facing model is shown here. Implementation helpers and
test adapters should stay local to the PR that introduces them.

Data-structure relationship:

```text
┌──────────────────────────────────────────────────────────────┐
│ scheduler.cpp / operations::forward                         │
│   owns ctx; calls coordinator(ctx)                           │
└──────────────┬──────────────────────────┬────────────────────┘
               │ owns                     │ calls with ctx&
               ▼                          ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ RequestCacheContext          │  │ CacheCoordinator             │
│   - PageArrayState entries   │◄─┤   - mutates ctx via APIs     │
│   - CheckpointPair entries   │  │   - FamilyRegistry           │
│   - read-only state views    │  │   - active family policies   │
│   - no allocator ownership   │  │   - produces RecoveryPlan    │
│   - no tree access           │  └──────────────┬───────────────┘
└──────────────┬───────────────┘                 │ owns
               │ request-local lifetime          ▼
               │                  ┌────────────────────────────────────────┐
               │                  │ RadixTree / TreeNode slots             │
               │                  │   slots[FamilyId] -> TreeAttachment   │
               │                  │                                        │
               │                  │   - TokenPageAttachment                │
               │                  │   - CompressedBlockAttachment          │
               │                  │   - SlidingWindowStateAttachment       │
               │                  │   - StateSlotAttachment                │
               │                  └───────────────────┬────────────────────┘
               │                                      │ shared lifetime
               └──────────────────┬───────────────────┘
                                  ▼
                   ┌────────────────────────────────────────┐
                   │ Existing allocator layer                │
                   │   PageAllocator                         │
                   │   PagedCacheGroupAllocator              │
                   │   MambaChunkAllocator                   │
                   │   request-local adapters                │
                   └────────────────────────────────────────┘
```

There is no standalone request-local entity beside `CacheCoordinator`.
`PageArrayState` and `CheckpointPair` are entries inside
`RequestCacheContext`. The coordinator can mutate them only because the
scheduler passes the context by reference to coordinator methods. The tree side
stores reusable `TreeAttachment` shapes; the request side stores current
per-request state.

`RequestCacheContext` is split out because its lifecycle is different from both
the tree and the allocators:

- Request-local state is created at admission, changes every scheduling step,
  and is fully cleaned up at request finish or abort.
- Tree attachments are shared artifacts. They are created only at commit
  boundaries and outlive the request until refcount/LRU eviction releases them.
- Allocators are global resource owners. They provide and reclaim pages or
  slots, but they should not encode request progress, prefix-match decisions,
  or tree publication state.

Keeping these lifecycles separate lets the scheduler carry one per-request
object through the forward path while preserving coordinator ownership of
mutating operations.

The code model should stay small. It needs five stable concepts:

| Concept | Purpose | Essential fields |
|---|---|---|
| `CacheResourceSpec` | Static declaration of one cache family role. | `id`, `family_index`, `family`, `attachment_kind`, `recoverability`, `publication`, `split_policy`, `rows_per_page`, `entry_stride_tokens`, optional `sliding_window_tokens`, `checkpoint_chunk_tokens`, `layer_indices`, optional `state_cohort_id`, `required_for_recovery`. |
| `RequestCacheContext` | Scheduler-held per-request cache facade. | request id, family-indexed page-array/checkpoint-pair entries, read-only state views; no tree attach/detach authority. |
| `FamilySlice` | One family contribution inside a match result. | `family`, `hit_node`, `recoverable_end_tokens`, `replay_from_tokens`, `replay_to_tokens`, `replay_mode`, borrowed page/slot ids, optional page-table base. |
| `RecoveryPlan` | Scheduler-facing answer to prefix matching. | `kv_terminal`, `matched_prefix_tokens`, `reusable_classical_end_tokens`, `recoverable_prefix_end_tokens`, `execution_resume_tokens`, `replay_ranges`, `slices`. |
| `ResourceDemand` / `AdmissionVerdict` | Admission accounting by family and by atomic state cohort. | family id, new units needed, releasable units, borrowed prefix units, optional `state_cohort_id`, verdict kind. |

The small enum set behind those structs is:

| Enum | Values |
|---|---|
| `CacheFamily` | `TokenPage`, `CompressedPage`, `SlidingWindowState`, `CompressionTailState`, `RecurrentState`, `ConvState`. |
| `AttachmentKind` | `TokenPageAttachment`, `CompressedBlockAttachment`, `SlidingWindowStateAttachment`, `StateSlotAttachment`, `NoneForRequestLocal`. |
| `Recoverability` | `Exact`, `AlignedCheckpoint`, `WindowRepairable`, `RequestLocalOnly`. |
| `PublicationPolicy` | `CanonicalPrefixIndex`, `AuxiliaryLocalOnly`, `RequestLocalOnly`. |
| `SplitPolicy` | `SplitPrefix`, `KeepOnSuffix`, `DropOnSplit`, `NoSplitAllowed`. |
| `ReplayMode` | `None`, `PrefillRecompute`, `StateForwardPropagate`, `WindowRebuild`. |

`CacheFamily` and `AttachmentKind` should not drift. `CacheFamily` names the
model semantic role; `AttachmentKind` names the reusable shape stored on a
`TreeNode`. The mapping is declared once in `CacheResourceSpec`:

| Cache family | Tree attachment shape |
|---|---|
| `TokenPage` | `TokenPageAttachment` |
| `CompressedPage` | `CompressedBlockAttachment`; CSA/HCA specs for the same raw span may be committed as one atomic bundle. |
| `SlidingWindowState` | `NoneForRequestLocal` by default; `SlidingWindowStateAttachment` if explicit SWA sharing is enabled. Full SWA Caching and Periodic Checkpointing are strategies on that one attachment kind, not separate tree state. |
| `CompressionTailState` | `NoneForRequestLocal` by default; incomplete tails are replay work, not shared attachments. |
| `RecurrentState` | `StateSlotAttachment` as part of the recurrent+conv checkpoint cohort. |
| `ConvState` | `StateSlotAttachment` as part of the same checkpoint cohort; it must not own a separate tree lifetime from `RecurrentState`. |

Spec-id naming keeps model roles out of scheduler branches:

- `kv.device`, `kv.host` -> `TokenPage`
- `v4.csa.main.compressed_kv`, `v4.csa.indexer.compressed_kv` ->
  `CompressedPage`
- `v4.hca.main.compressed_kv` -> `CompressedPage`
- `v4.swa.window_state` -> `SlidingWindowState`
- `v4.compression_tail.state` -> `CompressionTailState`
- `qwen35.gdn.recurrent_state` -> `RecurrentState`
- `qwen35.gdn.conv_state` -> `ConvState`

Layout comparison:

```text
1. Standard KV / MLA token pages

   logical tokens:     [0 ........ 63][64 ....... 127][128 ...... 191]
   page table:          page A          page B           page C
   tree attachment:     exact reusable token-page slices
   growth:              linear with context length


2. DeepSeek-V4 compressed history + sequence state

   raw tokens:        [---------- LCM span ----------][tail not complete]
                           |                 |                 |
                           v                 v                 v
                    CSA compressed KV   HCA compressed KV   SWA/tail state
                    + indexer KV        main history        request-local

   tree attachment:  complete compressed-history bundle only
   replay boundary:  incomplete compression tail / SWA window policy
   growth:           compressed history grows by completed spans;
                     sequence state stays fixed-size per request


3. Mamba/GDN recurrent + conv checkpoint

   raw tokens:        [0 ........ checkpoint N][N .... current tail]
                                  |                    |
                                  v                    v
                         state slot checkpoint       working slot
                         conv_state + recurrent      runtime updates every step

   tree attachment:   atomic state slot checkpoint at aligned boundary
   replay boundary:   nearest checkpoint <= token prefix
   growth:            state size is independent of sequence length;
                      checkpoints are sparse reusable snapshots
```

### 6.1 Allocators And Request Facades

Allocator classes should remain resource-specific in this refactor. Do not
introduce a generic allocator layer, and do not rename existing allocators into
pool-like abstractions.

Keep the current allocator ownership model:

- `PageAllocator` remains the allocator for standard KV pages.
- `PagedCacheGroupAllocator` remains the allocator for V4 paged-cache groups.
- `MambaChunkAllocator` remains the allocator for Mamba/GDN state slots.
- `OwnedPages` and `OwnedStateSlots` keep their concrete RAII ownership names.
- Low-level free-list or counter helpers may be factored later, but they are
  implementation details, not part of this design's public model.

The shared scheduler-facing layer is `RequestCacheContext`, above the
allocators:

- Page-array state: request-local state whose visible shape is borrowed prefix
  pages, owned suffix pages, raw-token cursor, base logical page, and optional
  sliding-window release. This covers standard KV suffix state and V4
  `PagedCacheGroupTable` behavior.
- Checkpoint-pair state: request-local state whose visible shape is one working
  slot plus one checkpoint slot. This covers `LocalMambaAllocator` behavior.

Those names describe context entry shapes, not new allocator base classes. At
runtime, `Scheduler` stores a `RequestCacheContext` per request and passes it by
reference to `CacheCoordinator`. `RequestCacheContext` may expose read-only
state views, but detach, release, attach, admission, and cohort mutation must go
through `CacheCoordinator`.

Do not add an intermediate indirection layer in this rollout. The context
object is passed directly to coordinator methods, and the coordinator mutates
the page-array or checkpoint-pair entries through explicit function calls on
that context.

The lifetime rule is still explicit:

- Page-like family: existing allocator -> request-local page-array entry in
  `RequestCacheContext` -> Tree attachment at commit boundary -> eviction
  returns pages to the same allocator.
- Slot-like family: existing allocator -> request-local checkpoint-pair entry in
  `RequestCacheContext` -> `StateSlotAttachment` at aligned checkpoint boundary
  -> eviction returns the slot to the same allocator.

`RecurrentState` and `ConvState` are separate families but one atomic checkpoint
cohort. A request must not publish, transfer, admit, or evict one without the
other, so the coordinator treats them as one checkpoint-pair recovery unit even
though the underlying allocator remains `MambaChunkAllocator`.

### 6.2 Tree Attachments

Tree-node family slots are the reusable side of the design.

- `TokenPageAttachment`: normal KV page ownership.
- `CompressedBlockAttachment`: canonical V4 compressed-history bundle. It may
  own multiple spec ids for the same raw-token span, because V4 history is
  reusable only when the required CSA/HCA groups are complete together.
- `SlidingWindowStateAttachment`: optional SWA storage when SWA is published to
  the tree. Full SWA Caching and Periodic Checkpointing are two strategies for
  the same logical attachment; an implementation should not keep both for the
  same SWA span.
- `StateSlotAttachment`: Mamba/GDN checkpoint slot. It is leaf-like and uses
  `SplitPolicy::KeepOnSuffix`.

The Phase-0 `PagedCacheSnapshot` bridge maps to `CompressedBlockAttachment` plus
optional sliding-window-state attachments. Preserve its no-split guard until
generic attachment slots are ready.

### 6.3 Recovery And Admission

The important fields in `RecoveryPlan` are conceptual, not syntactic:

```text
matched_prefix_tokens
  raw token equality from the radix walk

reusable_classical_end_tokens
  furthest canonical page/bundle boundary that can be attached directly

recoverable_prefix_end_tokens
  min boundary across all required families after checkpoint/replay rules

execution_resume_tokens
  first token the worker must execute after materializing borrowed state

replay_ranges
  explicit family-local work needed to rebuild missing sequence state

slices
  per-family attachments, borrowed page/slot ids, base offsets, and replay mode
```

Admission consumes the same family view:

```text
RecoveryPlan
      |
      v
ResourceDemand per family
  new_units_needed
  borrowed_prefix_units
  releasable_units
  state_cohort_id
      |
      v
AdmissionVerdict
  Ok / NeedEvict / Reject
```

Demands with the same `state_cohort_id` are admitted as one unit. V4
SlidingWindowState + CompressionTailState share one sequence-state block per
request; Mamba/GDN RecurrentState + ConvState share one checkpoint slot.

## 7. Critical-Path Discipline

The critical path is the target pipeline in Section 5. This section constrains
where indirection is allowed.

```text
HOT: per request scheduled, per token committed
  - fixed loop over startup-built ActiveFamilies
  - at most one family hook per Match / Admit / Commit phase
  - no string lookup, dynamic family discovery, or per-layer scan
  - no scheduler-level model branch

WARM: per insert decision, per eviction batch
  - finalize recovery slices
  - drive family-aware eviction
  - update LRU/refcount/lock accounting

COLD: radix split, HiCache, PD transfer, remote import
  - validate layout signatures
  - serialize immutable transfer descriptors
  - use split policy before attachment-specific split logic
```

Derived `CacheFamilyPolicy` implementations may simplify family-specific code,
but they must not create a second family-specific control flow outside
`CacheCoordinator`.

## 8. Invariants

| # | Invariant |
|---|---|
| I1 | Auxiliary-family lock implies its canonical carrier lock on the same node: `aux_lock_ref <= carrier_lock_ref`. |
| I2 | A family may attach to the global index only at its commit boundary. |
| I3 | `PublicationPolicy::RequestLocalOnly` families never call `TreeNode::Attach`. |
| I4 | `Recoverability::RequestLocalOnly` families return no global hit. |
| I5 | Auxiliary attachments are evicted before their carrier attachment is freed. |
| I6 | Borrowed prefix ids are imported before sliding release or fresh allocation. |
| I7 | `NoSplitAllowed` and Phase-0 `PagedCacheSnapshot` nodes are not radix-split. |

## 9. Cold-Path Layout And Transfer

HiCache, prefill/decode disaggregation, remote cache import, and future
cross-instance transfer paths must validate layout compatibility before treating
a cache hit as materializable. This is cold-path metadata; it must not add
string lookup, serializer probing, or topology checks to the scheduler hot path.

Every exported transfer descriptor should carry:

- `CacheResourceId` / `FamilyId` and raw logical token span.
- `Recoverability`, `PublicationPolicy`, and checkpoint / replay mode.
- Page or slot geometry: `rows_per_page`, `entry_stride_tokens`,
  `checkpoint_chunk_tokens`, and optional `sliding_window_tokens`.
- Tensor layout signature: dtype, layer membership, TP shard layout, PP stage
  ownership, and serializer version.
- Storage references: page ids, state-slot blobs, or external object ids.

Descriptors are immutable compatibility records. Local runtime state such as
`ref_count`, `lock_ref`, LRU position, pin state, and in-flight ownership stays
local to the receiving cache manager.
