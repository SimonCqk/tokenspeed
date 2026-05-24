# Scheduler Hybrid Cache Refactor Specification

## Purpose

This spec turns `docs/mamba-dsv4-refactor.md` into the implementation contract
for the current scheduler hybrid-cache PR.

The core goal is to make the scheduler cache critical path model-neutral:
`Scheduler` should interact with one cache facade, while model-specific cache
semantics live behind registered cache families and internal family policies.
The refactor must preserve current worker ABI, allocator ownership, Mamba
checkpoint behavior, and DeepSeek-V4 paged-cache behavior while moving the code
toward the terminal design.

The current branch already established two pieces of work:

- `docs: add hybrid cache refactor draft`: adds the high-level design note in
  `docs/mamba-dsv4-refactor.md`.
- `refactor(scheduler): consolidate hybrid cache lifecycle boundaries`: makes
  `HybridPrefixCache` always present, moves many cache lifecycle operations
  behind that facade, removes dormant scheduler prefetch helpers, adds
  `RequestCacheContext`, and adds focused C++ tests.

The current implementation keeps `HybridPrefixCache` as the concrete
scheduler-facing facade/coordinator for this PR, while splitting internal
family behavior out of the main translation unit. Registry construction and
dense active-family views live in `family_registry`; Mamba-specific behavior
lives in `mamba_family_ops`; paged-cache configuration, matching, admission,
attach/detach, and commit helpers live in `paged_cache_family_ops`.

This spec intentionally replaces slice-by-slice bookkeeping with durable
contracts. The PR should implement the parts below that are feasible without a
worker ABI migration or a full tree-slot rewrite.

## Primary Goals

### G1: Complete Family Registry

The cache facade must expose a complete startup-built registry of active cache
families. `HybridPrefixCache` owns the registry instance as part of the
scheduler-facing lifecycle boundary, while internal `family_registry` code owns
registry construction and dense active-family view setup. The registry is the
only place that translates model/config-specific cache roles into
scheduler-visible family records.

A `CacheResourceSpec` record must include at least:

| Field | Meaning |
|---|---|
| `id` | Stable string id used for config, diagnostics, and compatibility mapping. |
| `family_index` | Dense integer index used by hot-path arrays. |
| `family` | Semantic family: token page, compressed page, sliding-window state, compression-tail state, recurrent state, conv state. |
| `attachment_kind` | Reusable tree shape, or `NoneForRequestLocal`. |
| `recoverability` | Exact, aligned checkpoint, window repairable, or request-local only. |
| `publication` | Canonical prefix index, auxiliary local only, or request-local only. |
| `split_policy` | How the family behaves when a radix node is split. |
| `rows_per_page` / `entry_stride_tokens` | Page geometry when page-like. |
| `checkpoint_chunk_tokens` | Alignment when checkpoint-like. |
| `sliding_window_tokens` | Optional window bound for state families. |
| `state_cohort_id` | Atomic cohort for families that must be admitted/published/evicted together. |
| `required_for_recovery` | Whether this family gates recoverable-prefix selection. |

The registry must produce startup-built active-family arrays for match, admit,
commit, evict, finish, stats, and compatibility-flattening phases through the
internal registry module. Hot-path scheduler code must not discover families
through string lookup or model-name branches.

Current active-family inventory:

| Family or vocabulary | Current PR status | Registry treatment |
|---|---|---|
| Standard KV / token pages | Active for the existing KV prefix cache. | Required carrier family for token-prefix match, admission, publication, eviction, stats, and worker block-table compatibility. |
| Mamba/GDN checkpoint cohort | Active when the current model/config enables Mamba/GDN cache state. | Registered as an aligned checkpoint cohort; recurrent and conv state semantics must share the same `state_cohort_id` when split into separate policy records. |
| DeepSeek-V4 paged compressed history groups | Active when configured through V4 paged-cache group specs. | Registered as page-like history families with contiguous complete snapshots at configured history-alignment boundaries. |
| DeepSeek-V4 paged state/window groups | Active when configured through V4 paged-cache group specs. | Registered according to configured recoverability: required state gates prefix recovery; request-local-only state stays out of the global tree. |
| Recurrent, conv, sliding-window, compression-tail, and related state names | Policy vocabulary unless the current model/config wires them into active families. | They may appear in `CacheResourceSpec.family` and policy code, but must not appear in hot-path active arrays or tests as live families without current configuration wiring. |

An active family means a family produced by scheduler construction from the
current model/config. Future vocabulary is allowed in the registry type system,
but it is not evidence that dormant behavior is wired.

### G2: Small Scheduler-Facing Facade

The public scheduler-facing cache facade must converge to a small lifecycle
surface:

| Operation | Contract |
|---|---|
| `MatchPrefix` | Walk token prefix once and return a recovery plan across all required families. |
| `Admit` | Decide and execute the cache-side admission/eviction plan needed for a scheduling step. |
| `StepCommit` | Publish completed reusable state, acquire request-local suffix state, and populate worker compatibility metadata for the selected step. |
| `FinishRequest` | Release request-local cache state while tree attachments remain refcount/LRU managed. |
| `Stats` | Return read-only cache statistics and diagnostics through the facade. |

The implementation may keep temporary internal helper methods when moving all
scheduler call sites at once would be too disruptive. Those helpers must be
private, under an internal/detail namespace, or otherwise clearly marked as
compatibility glue. New scheduler production code should use the lifecycle
surface instead of adding new public phase-specific cache methods.

### G3: Phase-Specific Enums Become Internal Glue

Enums such as `CacheAdmissionKind`, `RequestLocalKVKind`,
`RequestLocalMambaKind`, `CachePublicationKind`, and
`CacheMaterializationKind` are transitional. They encode scheduler event shape,
not cache-family semantics.

The target is:

- scheduler-visible requests are lifecycle-oriented, for example
  `MatchPrefixRequest`, `AdmissionRequest`, `StepCommitRequest`,
  `FinishRequest`, and `StatsRequest`;
- family-specific decisions are driven by `CacheResourceSpec`,
  `RecoveryPlan`, `FamilySlice`, `ResourceDemand`, and `AdmissionVerdict`;
- any remaining prefill/decode/retract/finish switch used to preserve existing
  FSM behavior is internal compatibility glue and does not appear as the public
  facade contract.

### G4: Tree Attachments May Be Transitional

The terminal tree model is generic family slots on `TreeNode`:

```text
TreeNode
  slots[FamilyId] -> TreeAttachment
```

The current PR does not have to rewrite the tree into generic slots if that
would make the change too large. It may keep the existing `TreeNode` fields for
Mamba and `PagedCacheSnapshot`, or move them into a narrow
`TreeNodeAttachments` aggregate.

Regardless of physical representation:

- attach/detach must route through the cache facade/coordinator;
- `Scheduler` must not manipulate Mamba or paged-cache tree attachments
  directly;
- the family registry must describe the attachment semantics even if the
  storage is still the current concrete fields;
- the representation must leave an obvious migration path to generic slots.

### G5: Durable Tests, Not Slice Tests

Tests should protect stable contracts and invariants, not every intermediate
helper. The PR should add or keep focused C++ tests only where they exercise
production-visible behavior:

- family registration and active-family ordering;
- recoverable-prefix selection across KV, Mamba, and V4 paged families;
- admission/eviction ownership and state-cohort atomicity;
- borrowed-prefix import before sliding release/fresh allocation;
- worker compatibility metadata preservation;
- finish/retract cleanup of request-local state;
- stats/diagnostics read boundaries.

Avoid broad end-to-end matrices unless a new invariant cannot be tested more
directly. Temporary helper tests should be removed or rewritten once the helper
is no longer part of the design.

## Target Architecture

```text
Scheduler
  |
  |-- RequestCacheContext[request_id]
  |     |-- request-local page-array entries
  |     `-- request-local checkpoint-pair entries
  |
  `-- CacheCoordinator / HybridPrefixCache
        |
        |-- FamilyRegistry
        |     `-- CacheResourceSpec[family_index]
        |
        |-- active family policy arrays
        |
        |-- RadixTree / TreeNode attachments
        |
        |-- concrete allocators
        |     |-- PageAllocator / LocalKVAllocator
        |     |-- PagedCacheGroupAllocator / PagedCacheGroupTable
        |     `-- MambaChunkAllocator / LocalMambaAllocator
        |
        `-- compatibility flattening for existing forward-op ABI
```

The scheduler owns request ordering, token budget, request-pool allocation,
FSM state transitions, cache-operation routing, and worker operation assembly.
The cache facade owns cache-family matching, admission, commit/publication,
request-local cache state mutation, tree attach/detach, and cache statistics.

This is a logical mutation boundary, not an immediate physical-storage
migration. Scheduler-owned request objects and FSM states remain the owners of
request identity and state transitions. Existing request-local physical storage
may remain in `Request`, FSM event/state objects, or concrete local allocator
objects during this PR. The facade/coordinator must still mediate cache-state
mutations so scheduler code stops reaching around it to mutate KV, Mamba, or
V4 cache internals directly.

Concrete allocators remain concrete. The shared abstraction is not a universal
allocator; it is the coordinator-owned lifecycle order plus per-family specs
and policies.

## Lifecycle Contracts

### MatchPrefix

`MatchPrefix` consumes token pages, match intent, and request/cache context. It
returns a `RecoveryPlan`.

`RecoveryPlan` must include:

- raw token match depth;
- recoverable prefix end across all required families;
- execution resume position;
- per-family slices with borrowed page/slot ids and base offsets;
- optional replay ranges for families that can be rebuilt;
- compatibility fields needed by the current worker ABI.

Rules:

- KV/token-page hits are exact page hits.
- Mamba/GDN recurrent+conv hits require an aligned checkpoint cohort.
- V4 compressed history requires contiguous complete history-family snapshots.
- V4 state/window families gate the usable depth only according to their
  configured recoverability and state policy.
- A token prefix is usable only if every `required_for_recovery` family can
  attach exact state, restore an aligned checkpoint, or provide an explicit
  replay plan.

### Admit

`Admit` consumes the recovery plan, scheduling step shape, and mutable request
cache context. It returns an `AdmissionVerdict` and applies any cache-side
eviction/prune/debit required for the admitted step.

`ResourceDemand` is computed per family. Families with the same
`state_cohort_id` must be admitted atomically.

Rules:

- Borrowed prefix units reduce fresh demand.
- Sliding-window releasable units credit only owned pages, not borrowed ids.
- State-family pressure may prune state-only attachments when history remains
  usable.
- History-family pressure may cascade-prune descendants whose history chain
  would otherwise become invalid.
- Request-local-only families do not attach to the global tree and cannot be
  reported as global hits.
- Admission must not leave request-local tables partially mutated on rejection.

### StepCommit

`StepCommit` consumes the selected step, mutable request context, and recovery
plan. It is the lifecycle boundary for committing already-computed reusable
state and preparing the selected step before the worker receives the forward
operation.

Timing rules:

- it may publish state that was completed by an earlier worker execution;
- it may prepare or acquire request-local suffix pages/slots needed by the
  selected step;
- it may perform final publication for finish/retract when the publishable
  state was already computed;
- it must not publish pages, checkpoints, or paged-cache state for the selected
  step before worker execution has produced them.

Responsibilities:

- publish completed standard KV pages to the device tree;
- publish eligible Mamba/GDN checkpoints at aligned boundaries;
- publish complete V4 compressed-history bundles at history-alignment
  boundaries;
- keep request-local-only state local;
- import borrowed prefix before release/acquire on fresh paged-cache tables;
- acquire request-local suffix pages/slots for the selected step;
- populate existing worker ABI compatibility fields.

The current worker ABI remains unchanged. `StepCommit` is allowed to fill
legacy fields such as:

- `mamba_working_idx`;
- `mamba_checkpoint_dst_idx`;
- `mamba_cow_src_idx`;
- `mamba_branching_seqlen`;
- `paged_cache_block_tables`;
- `paged_cache_block_table_base_offsets`.

### FinishRequest

`FinishRequest` releases request-local cache state for a request that has
finished, aborted, or moved out of active scheduling.

Rules:

- request-local paged-cache tables release owned pages through RAII;
- borrowed prefix ids are dropped without allocator credit;
- request-local Mamba slots are released unless published as tree-owned
  checkpoints;
- shared tree attachments outlive the request and are managed by refcount/LRU;
- the scheduler remains responsible for erasing the request object and routing
  external events.

## Cache Operations And Outside Events

The scheduler remains the owner of cache-operation tracking and routing:

- `cache_op_tracker_` stays scheduler-owned;
- scheduler code assembles and routes `LoadBackOperation`,
  `WriteBackOperation`, and their request ids;
- an unknown cache-operation id in an outside event is a no-op after tracker
  lookup;
- `WriteBackDone` still routes through the tracked request id and advances the
  request/FSM as today;
- the family registry/lifecycle refactor must not restore removed node-touch,
  node-ref, or L3 transfer-tracker behavior in `CacheOpSpec`;
- outside `PrefetchDone` handling remains outside this PR unless a future scope
  explicitly designs scheduler-generated prefetch behavior.

### Stats

`Stats` is read-only and must not mutate cache state.

At minimum it should cover:

- available standard-KV device pages;
- configured paged-cache group ids and per-group capacity/failure counters;
- request-local paged-cache page ids/base offsets for compatibility
  introspection;
- debug memory diagnostics snapshots;
- family-level hit/admission/eviction counters when available.

Public statistics that are still KV-only must say so in the method contract.
Do not silently aggregate unrelated family capacities under an existing KV
method name.

## Registry And Policies

Each active family has one policy implementation selected from its
`CacheResourceSpec`. Policies can be concrete classes, function tables, or
static helpers; the spec does not require virtual dispatch.

Policy responsibilities:

| Phase | Policy responsibility |
|---|---|
| Match | Interpret tree attachment and compute recoverable family slice. |
| Demand | Compute new units, borrowed units, and releasable units. |
| Admit/Evict | Choose family-safe prune/eviction candidates. |
| Commit | Move request-local ownership into tree attachments at boundaries. |
| Finish | Release request-local state and clear compatibility state. |
| Stats | Report family-local counters and snapshots. |

Policy examples:

- `TokenPage`: exact KV pages on device/host, carrier resource for many
  auxiliary attachments.
- `CompressedPage`: V4 compressed-history bundles, complete only at configured
  LCM boundaries.
- `SlidingWindowState`: request-local by default; optionally tree-attached only
  through an explicit recovery policy.
- `CompressionTailState`: request-local tail/replay work by default.
- `RecurrentState` and `ConvState`: separate families but one atomic
  checkpoint cohort for Mamba/GDN-style recovery.

## Current PR Scope

The current PR should implement these if feasible:

- introduce and wire a complete `FamilyRegistry` for the active configured
  families, with construction isolated in the internal `family_registry`
  module;
- make `HybridPrefixCache` or the eventual coordinator expose the lifecycle
  facade as the scheduler-facing API;
- migrate scheduler production paths toward `MatchPrefix`, `Admit`,
  `StepCommit`, `FinishRequest`, and `Stats`;
- move remaining public phase-specific enum APIs into private/internal
  compatibility glue when the scheduler no longer needs to call them directly;
- preserve current concrete allocator boundaries and worker ABI compatibility;
- keep existing `TreeNode` Mamba/paged-cache fields or introduce a narrow
  `TreeNodeAttachments` bridge, but do not require a full generic slot rewrite;
- replace slice-oriented tests with durable invariant tests where practical.

Allowed deferrals:

- generic `TreeNode::slots[FamilyId]` storage;
- worker/Python ABI migration away from compatibility views;
- full replay-range execution for all recoverability modes;
- cold-path transfer descriptor serialization;
- broad benchmark or GPU/runtime validation not available in this local C++
  scheduler test environment.

## Acceptance Criteria

- AC1: The registry covers every current active/configured family listed in the
  active-family inventory and does not treat future vocabulary as live without
  config wiring.
- AC2: Production scheduler match, admission, commit, finish, and stats paths
  call the lifecycle facade wherever the PR claims that boundary moved.
- AC3: Phase-specific prefill/decode/retract enums are private/internal glue
  and are not the public scheduler-facing facade contract.
- AC4: Recoverable-prefix selection gates on every `required_for_recovery`
  family: standard KV, enabled Mamba/GDN checkpoint state, and configured V4
  paged families.
- AC5: Families sharing a `state_cohort_id` admit, publish, evict, and recover
  atomically.
- AC6: Worker-facing forward-op ABI fields and defaults remain unchanged.
- AC7: Scheduler cache-operation tracking, operation assembly, unknown-op no-op
  behavior, and `WriteBackDone` request advancement remain unchanged.
- AC8: The refactor does not re-enable dormant L3 prefetch scheduling or
  reintroduce removed node-touch/L3 transfer bookkeeping.
- AC9: Durable C++ tests cover registry construction, recoverability, cohort
  atomicity, StepCommit timing, compatibility metadata, cache-op routing, and
  read-only stats/diagnostics.

## Invariants

| Id | Invariant |
|---|---|
| I1 | Scheduler code does not branch on model names or cache-family implementation details. |
| I2 | Hot path uses dense family indexes or startup-built active-family arrays, not dynamic family discovery. |
| I3 | A prefix is recoverable only if every required family is exact, checkpoint-restorable, or explicitly replayable. |
| I4 | Families in the same `state_cohort_id` are admitted, published, evicted, and recovered atomically. |
| I5 | Request-local-only families never attach to the global radix tree. |
| I6 | Attach/detach of Mamba and paged-cache tree state routes through the cache facade/coordinator. |
| I7 | Borrowed prefix import happens before sliding release or fresh allocation on a fresh table. |
| I8 | Borrowed pages do not create allocator credit when released from request-local tables. |
| I9 | V4 history snapshots form a contiguous chain at history-alignment boundaries. |
| I10 | State-only prune must not break an otherwise complete history chain. |
| I11 | KV carrier eviction detaches dependent auxiliary attachments before the carrier resource is freed. |
| I12 | Existing forward-op ABI fields keep their current values/defaults until an explicit ABI migration. |
| I13 | Concrete allocators own concrete resources; no universal allocator hides geometry or lifetime rules. |
| I14 | `Stats` and diagnostics are read-only. |
| I15 | Public facade methods are lifecycle-oriented; phase-specific prefill/decode/retract enums remain internal glue. |
| I16 | `StepCommit` never publishes selected-step pages or state before worker execution has computed them. |
| I17 | Scheduler owns `cache_op_tracker_`, operation assembly, outside-event routing, and `WriteBackDone` request advancement. |
| I18 | Unknown cache-operation ids are no-ops, and the refactor does not restore node-touch or L3 transfer-tracker bookkeeping. |
| I19 | Outside `PrefetchDone` remains out of scope unless a future design explicitly wires scheduler-generated prefetch behavior. |

## Durable Test Plan

The minimum durable C++ coverage should be:

1. Family registry construction:
   configured KV, Mamba, and V4 paged groups produce stable family specs and
   dense active-family indexes; policy vocabulary not wired by current config
   stays out of active arrays; invalid group configs still fail at scheduler
   construction.

2. MatchPrefix recoverability:
   a deeper raw token match is capped when Mamba or V4 state is missing, and a
   deeper recoverable prefix is used when all required family state exists.

3. Admission ownership:
   paged-cache admission credits owned sliding releases but not borrowed prefix
   ids; state cohorts admit/reject atomically.

4. StepCommit ordering:
   borrowed-prefix import precedes release/acquire, complete V4 history bundles
   publish only at alignment boundaries, and Mamba checkpoints publish only at
   aligned checkpoint boundaries. State for the selected step is not published
   before worker execution computes it.

5. Finish/retract cleanup:
   request-local KV, paged-cache, and Mamba state is released or published with
   the same RAII ownership behavior as before.

6. Compatibility metadata:
   existing Mamba and V4 forward-op fields are populated from lifecycle results
   and retain current defaults when a family is disabled.

7. Stats/diagnostics:
   available-page and debug-memory snapshots are read through the facade and do
   not mutate cache state.

8. Cache-operation routing:
   unknown operation ids are ignored, `WriteBackDone` advances the tracked
   request as before, and no node-touch/L3 transfer bookkeeping is restored.

If a current helper test only proves a transitional enum/helper, prefer
rewriting it around one of these invariants once the lifecycle facade is wired.

## Non-Goals

- Do not collapse all cache families into a KV-page abstraction.
- Do not introduce a universal allocator.
- Do not require a full generic `TreeNode` slot rewrite in this PR.
- Do not change worker-facing forward-op tensor/metadata ABI in this PR.
- Do not enable dormant L3 prefetch scheduling as part of this refactor.
- Do not add per-token/per-layer virtual dispatch or hot-path string lookup.
- Do not add broad tests whose only purpose is to lock down temporary helper
  names or transitional enum variants.
