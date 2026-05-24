# Scheduler Hybrid Cache Implementation Design

## Core Principle

Extend `HybridPrefixCache` into the scheduler-facing cache lifecycle facade. Do
not add a separate public `CacheCoordinator` for this PR: the current
`HybridPrefixCache` already wraps `KVPrefixCache`, Mamba state, paged-cache
groups, request-local tables, eviction callbacks, and cache diagnostics.

The refactor should make scheduler call sites lifecycle-oriented while keeping
current concrete allocators, `TreeNode` fields, FSM ownership, and worker ABI
metadata intact.

## Type Layout

Public scheduler-facing value types should live next to the facade:

- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache_types.h`
  for lifecycle request/result structs and registry records.
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h`
  for `HybridPrefixCache` facade methods and narrow read-only accessors.
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp`
  for facade/lifecycle orchestration and KV/common compatibility glue.
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/family_registry.{h,cpp}`
  for registry construction, validation, and startup-built dense active-family
  views.
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/mamba_family_ops.{h,cpp}`
  for Mamba-specific match, allocation, publication, metadata, and slot helper
  behavior.
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/paged_cache_family_ops.{h,cpp}`
  for paged-cache config, match augmentation, request-table lifecycle,
  admission/prune, attach/detach, stats, and commit helper behavior.

These files are internal implementation boundaries under the concrete facade.
They are not a broad public `CacheFamily` framework and do not make allocator
semantics universal.

Keep phase-specific enums out of the public contract. If the old helpers are
kept during migration, move their enum request types under `private:` in
`HybridPrefixCache` or an internal `detail` namespace.

Practical public types:

```cpp
enum class CacheFamily {
    TokenPage,
    CompressedPage,
    SlidingWindowState,
    CompressionTailState,
    RecurrentState,
    ConvState,
};

enum class TreeAttachmentKind { ReusableTree, NoneForRequestLocal };
enum class Recoverability { Exact, AlignedCheckpoint, WindowRepairable, RequestLocalOnly };
enum class PublicationKind { CanonicalPrefixIndex, AuxiliaryLocalOnly, RequestLocalOnly };
enum class SplitPolicy { CarrierKV, CheckpointBoundary, SnapshotBoundary, RequestLocalOnly };

struct CacheResourceSpec {
    std::string id;
    std::int32_t family_index;
    CacheFamily family;
    TreeAttachmentKind attachment_kind;
    Recoverability recoverability;
    PublicationKind publication;
    SplitPolicy split_policy;
    std::int32_t rows_per_page{0};
    std::int32_t entry_stride_tokens{0};
    std::int32_t checkpoint_chunk_tokens{0};
    std::optional<std::int32_t> sliding_window_tokens;
    std::string state_cohort_id;
    bool required_for_recovery{false};
};
```

`FamilyRegistry` owns `std::vector<CacheResourceSpec>` plus startup-built dense
arrays for `match`, `admit`, `commit`, `evict`, `finish`, `stats`, and
compatibility flattening. Its construction lives in `family_registry`, not in
the lifecycle orchestration file. It may keep an `id -> family_index` map for
startup validation and diagnostics, but hot paths should iterate the dense
arrays.

`RecoveryPlan` can initially wrap the current `MatchResult` compatibility view
while adding stable family-oriented fields:

- raw token match depth;
- recoverable prefix end;
- resume position;
- per-family `FamilySlice` records with borrowed ids and base offsets;
- optional replay ranges;
- `MatchResult` compatibility fields needed by existing scheduler/FSM code.

`AdmissionRequest`, `AdmissionVerdict`, `StepCommitRequest`,
`StepCommitResult`, `FinishRequest`, `StatsRequest`, and `CacheStatsSnapshot`
should be lifecycle facts, not prefill/decode/retract enum wrappers. Internal
mapping may temporarily convert these structs to the old helper kinds.

## Registry Construction

Build the registry during scheduler cache construction/configuration through
`HybridPrefixCache`, with construction delegated to the internal
`family_registry` module:

1. `Scheduler::Scheduler` constructs `HybridPrefixCache` as today.
2. `HybridPrefixCache` collects the standard KV page geometry, Mamba allocator
   presence/chunk size, and paged-cache group configs plus required-group
   status.
3. `family_registry` builds and validates `CacheResourceSpec` records, dense
   active-family arrays, and derived required history/state group views.
4. `paged_cache_family_ops` owns paged-cache config validation, allocator
   registration, required-group enablement, and rebuild triggers.
5. After scheduler construction/configuration, treat the registry as immutable.

Current-code-friendly family rules:

| Source | Registry behavior |
|---|---|
| Standard KV | Always register one `TokenPage` carrier family. Device/host are storage tiers, not separate families. |
| Mamba/GDN | Register an aligned checkpoint family when `mamba_allocator_ != nullptr`. If recurrent and conv state become separately observable, register both with the same `state_cohort_id`. |
| V4 paged groups | Register one spec per configured `PagedCacheGroupConfig`. Required prefix-cache adjunct groups get tree attachment and recovery policy; transport-only groups are request-local/compatibility families. |
| Future vocabulary | Keep enum values available, but do not place them in active arrays without current config wiring. |

The existing `paged_cache_history_groups_`, `paged_cache_state_groups_`, and
fast lookup sets should become derived registry views rather than separately
maintained sources of truth.

## Facade Mapping

`MatchPrefix` should replace public scheduler calls to `Match`. The first
implementation may call the current `Match` overloads and populate
`RecoveryPlan.compat_match`, then progressively move Mamba and paged-cache
augmentation into per-family policy loops.

`Admit` should keep the name but change to lifecycle request/verdict types.
Scheduler call sites in `scheduler/operations/forward.cpp` should pass resource
demand, step bounds, recovery plan, and protected recovery node facts. Internal
compatibility glue may still map those facts to the old admission cases until
the switch is removed.

`StepCommit` should absorb the roles currently split across:

- `Publish`;
- `Materialize`;
- `PrepareRequestLocalKV`;
- `PrepareRequestLocalMamba`;
- `PrepareForwardOp`.

It must preserve the timing invariant: publish only state already computed by
an earlier worker execution, acquire request-local suffix state for the selected
step, then populate existing forward-op compatibility metadata. It must not
publish pages, checkpoints, or paged-cache snapshots for the selected step
before worker execution completes.

`FinishRequest` remains the request-local cleanup boundary. It should continue
to release paged-cache request tables and request-local state while leaving
tree attachments under refcount/LRU ownership.

`Stats` should group the current read-only helpers:

- `AvailableDevicePages`;
- `PagedCacheGroupIds`;
- `PagedCacheGroupTotalPages`;
- `PagedCacheGroupAvailablePages`;
- `PagedCacheGroupFailedAllocCount`;
- `GetRequestPagedCachePageIds`;
- `GetRequestPagedCacheBaseLogicalPage`;
- `CollectDeviceMemoryDiagnostics`.

Existing public `Scheduler` methods can keep their names and delegate through
`Stats`; do not silently aggregate non-KV family capacity under KV-only method
names.

## Scheduler-Owned Boundaries

Do not move these responsibilities into `HybridPrefixCache`:

- request ordering and token budget in
  `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`;
- request-pool allocation and FSM transitions in
  `tokenspeed-scheduler/csrc/fsm/forward_events.h/.cpp`;
- request object ownership in `tokenspeed-scheduler/csrc/scheduler/scheduler.h`;
- forward/cache operation assembly in
  `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`;
- `cache_op_tracker_` ownership in `scheduler.h` and tracker mutation in
  `scheduler/operations/forward.cpp`;
- outside event routing in `scheduler/outside_event_handler.cpp`;
- unknown cache-operation id no-op behavior and `WriteBackDone` request/FSM
  advancement.

The refactor must not restore scheduler-generated L3 prefetch, node-touch,
node-ref, or L3 transfer-tracker fields in `CacheOpSpec`.

## Acceptance Criteria Map

| AC | Implementation guide |
|---|---|
| AC1 | Add `FamilyRegistry` / `CacheResourceSpec` in `hybrid_prefix_cache_types.h`; build specs from KV, Mamba allocator presence, and `PagedCacheGroupConfig` in `family_registry`; expose registry stats for tests through the facade. |
| AC2 | Migrate production calls in `scheduler/operations/forward.cpp`, `fsm/forward_events.cpp`, and `scheduler/scheduler.cpp` to `MatchPrefix`, lifecycle `Admit`, `StepCommit`, `FinishRequest`, and `Stats`. |
| AC3 | Move `CacheAdmissionKind`, `RequestLocalKVKind`, `RequestLocalMambaKind`, `CachePublicationKind`, `CacheMaterializationKind`, and `CacheOpPrepareKind` behind private/internal compatibility helpers after call sites migrate. |
| AC4 | Encode `required_for_recovery` in specs; make `MatchPrefix` cap `RecoveryPlan.recoverable_prefix_end` across KV, enabled Mamba checkpoint state, and configured V4 required families. |
| AC5 | Use `state_cohort_id` in admission, publication, eviction, and recovery loops; Mamba recurrent/conv and any state cohort must commit or reject as a unit. |
| AC6 | Keep `ForwardOperationBase` fields and defaults unchanged; `StepCommit` only centralizes how Mamba and paged-cache compatibility fields are populated. |
| AC7 | Leave `cache_op_tracker_`, op-id allocation use, operation assembly, unknown-op no-op, and `WriteBackDone` advancement in scheduler files. |
| AC8 | Do not add back removed `CacheOpSpec::last_node` / `nodes` fields or dormant scheduler prefetch/node-touch/L3 transfer behavior. |
| AC9 | Add durable C++ tests for registry, recoverability, cohorts, `StepCommit` timing, ABI metadata, cache-op routing, and read-only stats. |

## Test Strategy

Keep focused registry coverage in the scheduler C++ tests and list any new test
source in `tokenspeed-scheduler/CMakeLists.txt`. Cover KV-only, Mamba-enabled,
V4 transport-only, and V4 prefix-cache-adjunct configs; assert dense indexes
and active arrays are stable.

Update existing invariant tests instead of preserving helper-name tests:

- `test_mamba_cache.cpp`: Mamba recoverability, checkpoint cohort admission,
  and compatibility metadata through `StepCommit`.
- `test_mamba_integration.cpp`: end-to-end Mamba prefill/decode/finish remains
  behaviorally unchanged.
- `test_paged_cache_prefix_match.cpp` and `test_paged_cache_family_split.cpp`:
  V4 recoverability capping and history/state family selection.
- `test_paged_cache_prefix_hit_commit.cpp` and
  `test_paged_cache_attach_loop.cpp`: borrowed-prefix import before
  release/acquire and commit timing.
- `test_paged_cache_eviction.cpp`: state-only prune, carrier eviction detach,
  and cohort-safe cleanup.
- `test_scheduler_memory_diagnostics.cpp`: `Stats` is read-only and keeps
  KV-only diagnostics semantics explicit.
- `test_outside_event_handler.cpp`, `test_write_back.cpp`, and
  `test_retract.cpp`: cache-op routing, unknown-op no-op, and `WriteBackDone`
  advancement remain scheduler-owned.

Tests that directly instantiate phase-specific enum requests should be migrated
to lifecycle requests unless they are explicitly scoped to internal
compatibility glue through `HybridPrefixCacheTestPeer`.

## Implementation Sequence

1. Add lifecycle/registry types and build the startup registry through
   `family_registry`, with `HybridPrefixCache` storing/exposing the result; add
   registry construction tests.
2. Add `MatchPrefix`, lifecycle `Admit`, `StepCommit`, and `Stats` wrappers that
   preserve current behavior by delegating to existing helpers.
3. Migrate scheduler and FSM production call sites to lifecycle methods while
   keeping operation assembly and `cache_op_tracker_` in scheduler code.
4. Move old phase-specific request enums/helpers behind private/internal
   compatibility glue and update helper tests to durable lifecycle tests.
5. Strengthen recoverability/cohort/commit-timing tests, then run the focused
   scheduler C++ test target.
