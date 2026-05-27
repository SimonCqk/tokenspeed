# DeepSeek V4 L2 KV Cache Offloading Design Plan

Status: implementation in progress.
Branch: `feat/v4-l2-kv-offload`.

This document describes the executable design for DeepSeek V4 L2 KV cache
offloading. The goal is to let V4 paged-cache snapshots move from L1 device
pages to L2 host pages when device group pages are exhausted, then load them
back to device pages before a forward pass that reuses the cached prefix.

## Scope

In scope:

- DeepSeek V4 paged-cache groups: SWA KV, compressed KV, compressor state, and
  indexer state.
- L1 device page to L2 host page offload for immutable TreeNode snapshots.
- L2 host page to L1 device page loadback before forward execution.
- Scheduler-side residency, lifetime, admission, eviction, and transfer plan.
- Runtime-side host buffers and page-aware copy for V4 group layouts.
- Tests for scheduler invariants, runtime dispatch, and end-to-end behavior.

Out of scope for the first implementation:

- Generic multi-tier cache policy interfaces.
- Normal KV prefix-cache L2 redesign.
- L3 storage, NVMe, cross-node transfer, compression, or speculative writeback.
- C++ API-wide migrations such as `std::expected` or broad template concepts.
- Writeback cancellation when a pending snapshot is re-hit.

## Current State

Normal KV prefix cache already has a host tier. `KVPrefixCache` accepts both
device and host `PageAllocator`s, `TreeNode` can hold host resources, and the
FSM can materialize host writeback pages and emit KV transfer pairs. That path
is useful prior art, but it is not the missing V4 feature.

DeepSeek V4 currently uses a parallel paged-cache group path:

- `PagedCacheGroupConfig` describes one model-defined group with retention,
  raw-token geometry, family, and device `total_pages`.
- `PagedCacheGroupAllocator` owns one device page pool per group.
- `PagedCacheGroupTable` is request-local and stores device page ids as
  `borrowed_page_ids_ + owned_pages_`.
- `PagedCacheSnapshot` is attached to a `TreeNode` and owns per-group pages.
- Forward metadata receives only device page ids and base logical offsets.

Runtime support is explicitly disabled for V4 L2 today:

- `event_loop.py` rejects `enable_kvstore` for `DeepseekV4TokenToKVPool`.
- `DeepseekV4TokenToKVPool.get_cpu_copy()` and `load_cpu_copy()` raise because
  V4 buffers are page-shaped and need page-aware indexing.

Therefore the implementation must target the V4 paged-cache group path, not the
ordinary KV `TreeNode::host_resource_` path.

## First Principles

1. Scheduler owns addresses and lifetimes.
   It decides which group pages live on device, which live on host, which
   snapshots are complete, and when transfer operations are safe.

2. Runtime owns bytes and layout.
   The C++ scheduler must not know V4 tensor shapes, dtype, layer ratios, or
   indexer buffer layout. It emits group-keyed page transfers; Python runtime
   copies the right buffers for that group.

3. Forward sees only L1 device pages.
   Host page ids must never enter `paged_cache_block_tables` or backend
   metadata. A host hit must be loaded back to device pages before the forward
   op is populated.

4. Offload only published immutable snapshots.
   Request-local `PagedCacheGroupTable` owned pages are still being written by
   the current forward pass and must not be offloaded. They become eligible
   only after commit into a `TreeNode` snapshot.

5. Keep the abstraction narrow.
   V4 group semantics are real: History vs State, full-history vs
   sliding-window, raw-token geometry, and required groups. The design extends
   those existing structures instead of introducing a universal cache-family
   framework.

## Scheduler Data Model

### Group Config

Extend `PagedCacheGroupConfig` with an optional host budget:

```cpp
struct PagedCacheGroupConfig {
    std::string group_id;
    std::int32_t rows_per_page{};
    std::int32_t entry_stride_tokens{};
    std::int32_t total_pages{};       // L1 device pages.
    std::int32_t host_total_pages{};  // L2 host pages; 0 disables L2 for group.
    Retention retention{Retention::FullHistory};
    std::optional<std::int32_t> sliding_window_tokens{};
    PagedCacheGroupFamily family{PagedCacheGroupFamily::History};
};
```

Keep device and host budgets per group. V4 group ids are opaque strings at the
API boundary, but scheduler internals should assign a stable small
`PagedCacheGroupIndex` during configuration to avoid repeated string hashing in
hot transfer paths.

### Group Allocators

Keep the existing device allocator map and add a host allocator map:

```cpp
std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>>
    paged_cache_allocators_;
std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>>
    paged_cache_host_allocators_;
```

`PagedCacheGroupAllocator` can continue wrapping `PageAllocator`. The page id
space is independent per group and per tier; the transfer key identifies the
group and direction.

### Snapshot Residency

Change each group snapshot from single-tier ownership to tiered residency:

```cpp
struct PagedCacheGroupSnapshot {
    OwnedPages device_pages;
    OwnedPages host_pages;
    std::int32_t base_logical_page{0};
    std::int32_t raw_token_cursor{0};
    bool sliding{false};

    bool HasDevice() const { return !device_pages.Empty(); }
    bool HasHost() const { return !host_pages.Empty(); }
};
```

If a staged migration is preferable, first add this as a new type and convert
the existing `pages` field to `device_pages` in a mechanical patch. The semantic
target is one snapshot entry with independent device and host residency.

Snapshot completeness must become tier-aware. A group can be complete on
device, complete on host, or complete on both. Device completeness is required
for immediate forward reuse. Host completeness is enough only to plan loadback.

### Match Result

Keep the current device-ready paged-cache hit as the fast path. Add a host hit
that carries the same per-group logical data but points at host pages:

```cpp
struct PagedCacheHit {
    TreeNode* last_node{nullptr};
    std::int32_t prefix_len_tokens{0};
    std::map<std::string, std::vector<std::int32_t>> per_group_page_ids;
    std::map<std::string, std::int32_t> per_group_base_logical_page;
};

struct PagedCacheMatch {
    PagedCacheHit device;
    PagedCacheHit host;
};
```

The existing `MatchResult::PagedCache` can evolve toward this shape. Avoid
mixing host page ids into the device hit fields.

### Transfer Representation

V4 paged-cache transfers need a group key. `CacheKind::kKV` is not enough
because page id `7` in `v4.swa_kv` and page id `7` in
`v4.c4a.compressed_kv` name different physical buffers.

Use grouped transfer payloads rather than storing a string in every page pair:

```cpp
struct PagedCacheTransferGroup {
    std::string group_id;  // API-facing; internally may be group_index.
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;
};

struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;  // existing KV/Mamba.
    std::vector<PagedCacheTransferGroup> paged_cache_transfers;
    bool is_retract{false};
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;
    std::vector<PagedCacheTransferGroup> paged_cache_transfers;
};
```

Bindings should expose these as `src_pages_by_paged_group` and
`dst_pages_by_paged_group`, parallel to the existing `src_pages_by_kind`.

## Scheduler Flows

### Device Hit

This path must remain unchanged in cost and semantics:

1. `augmentMatchPagedCache()` finds required groups complete on device.
2. `AcquireForRequest()` imports borrowed device page ids into fresh
   `PagedCacheGroupTable`s.
3. `populatePagedCachePagesForOp()` writes device page ids and base offsets into
   the forward op.
4. Runtime runs without L2 transfer.

### Device Pressure and Offload

Trigger offload from the paged-cache group admission path, before pruning a
snapshot or rejecting admission:

1. `checkPagedCacheGroupAdmission()` computes shortfall per group.
2. If a group has device shortfall and host L2 is enabled, try to offload
   eligible snapshot pages for that group.
3. Candidate snapshots come from `paged_cache_snapshot_nodes_`, ordered by the
   same deterministic policy used for snapshot pruning: oldest time first, with
   stable tie breakers.
4. Skip candidates that are active, pinned, already pending writeback, or do not
   have device pages for the pressured group.
5. Allocate host pages from the corresponding host group allocator.
6. Emit one `PagedCacheTransferGroup` per group.
7. Keep device pages visible and unavailable for reuse until writeback ack.
8. On ack, mark host residency visible and release the offloaded device pages.

If host allocation fails, try host snapshot eviction for that group. If host is
still full, fall back to the existing prune or admission-fail behavior.

### Host Hit and Loadback

When device snapshots are incomplete but host snapshots are complete:

1. `augmentMatchPagedCache()` reports a host paged-cache hit, not a device hit.
2. Before forward op creation, scheduler allocates destination device pages for
   every required host group page.
3. Scheduler emits group-keyed H2D loadback transfers.
4. Scheduler attaches destination device pages to the snapshot or request table
   state before populating forward metadata.
5. `populatePagedCachePagesForOp()` uses only the destination device page ids.
6. Runtime fences the forward layer against the loadback completion, using the
   existing layerwise HostExecutor model.

Loadback should keep the host pages resident. The snapshot becomes dual-resident
until a later device demotion releases the L1 copy.

### Snapshot Pinning

Current borrowed device pages are protected indirectly by the request's
`DeviceNodeRef`. Host-only paged-cache hits need equivalent protection. Add a
small snapshot-level guard instead of relying on ordinary KV host resources:

```cpp
class PagedCacheSnapshotRef {
public:
    explicit PagedCacheSnapshotRef(PagedCacheSnapshot* snapshot);
    PagedCacheSnapshotRef(PagedCacheSnapshotRef&&) noexcept;
    PagedCacheSnapshotRef& operator=(PagedCacheSnapshotRef&&) noexcept;
    ~PagedCacheSnapshotRef();
};
```

The guard increments a snapshot reader count and causes offload, loadback, and
host eviction candidate selection to skip that snapshot. This is narrower than
adding a generic refcounted cache-family layer and prevents host pages from
being evicted while an H2D transfer is in flight.

## Runtime Design

### Host Pool

Add a V4-specific host pool, for example `DeepseekV4PagedCacheHostPool`, instead
of using the ordinary `HostKVCache` shape. It mirrors the device V4 group
buffers on CPU pinned memory:

- `v4.swa_kv`: per layer `(host_pages, swa_block_bytes)`, `uint8`.
- `v4.c{ratio}a.compressed_kv`: ratio-specific compressed KV buffers.
- `v4.c4a.compressed_kv`: also copies `indexer_kv_buffer`, because indexer KV
  shares the compressed group page ids.
- `v4.c{ratio}a.compressor_state`: per layer state buffers, `float32`.
- `v4.c4a.indexer_compressor_state`: indexer state buffers, `float32`.

The transfer pool for a paged group should use `page_size() == 1`; transfer
indices are group page ids, not token ids.

### HostExecutor Dispatch

Extend HostExecutor input grouping from `CacheKind` to a group-aware key. The
minimal Python-facing shape is:

```python
writeback_paged_groups: dict[str, tuple[list[int], list[int]]]
loadback_paged_groups: dict[str, tuple[list[int], list[int]]]
```

For each `group_id`, dispatch to the V4 paged-cache host pool. Keep ordinary KV
and Mamba dispatch unchanged.

### Copy Strategy

First implementation:

- Copy per group and per layer.
- View each page as contiguous bytes where possible.
- Use existing transfer kernels only when their contract matches the V4 buffer
  layout.
- Otherwise add a small page-copy kernel under `tokenspeed-kernel`, following
  the dependency boundary: runtime imports only through `tokenspeed-kernel`.

Performance follow-up:

- Bucket layers with identical bytes-per-page and copy them through pointer
  tables in one batched launch.
- Prefer H2D loadback over D2H writeback when both queues are non-empty.
- Pre-register/pin host tensors once at startup.

## Configuration

Add host page budgets per V4 group. The Python scheduler config path should pass
both device and host counts:

- `pool_to_paged_cache_groups()` sets `total_pages` from device counts.
- The same function or a companion helper sets `host_total_pages` from V4 L2
  sizing.
- `event_loop.py` no longer rejects `enable_kvstore` for
  `DeepseekV4TokenToKVPool` once the V4 host pool exists.

Prefer deriving default host group counts from the same V4 group specs used for
device sizing. A CLI override can be added later if operational tuning needs it.

## Invariants

These invariants should be enforced by tests:

1. No host page id enters forward paged-cache metadata.
2. Device pages are not returned to a group allocator before D2H ack.
3. Pending host writeback is not visible as a host hit.
4. Host pages are pinned by a snapshot guard during H2D loadback.
5. Request-local `PagedCacheGroupTable` owned pages are never offloaded.
6. A group snapshot may be device-only, host-only, or dual-resident, but its
   base logical page and raw-token cursor remain identical across tiers.
7. Required-group completeness is evaluated per tier.
8. Host eviction may drop L2 value, but must not corrupt device residency or
   borrowed request tables.

## Performance Notes

- L1 device hits should remain zero-transfer and avoid extra allocations.
- Use stable group indices internally; avoid string work in page-pair loops.
- Reserve transfer vectors from known page counts.
- Offload state-family groups independently; they are often smaller and can
  recover admission pressure without moving full-history compressed KV.
- Keep loadback high priority and writeback opportunistic.
- Record metrics per group: writeback pages, loadback pages, host hit depth,
  device hit depth, offload failures, host evictions, and loadback wait time.
  The first implementation exposes scheduler-side cumulative counters for
  scheduled D2H writeback pages, scheduled H2D loadback pages, L2 host pages
  evicted, and hard H2D materialization failures.

## Implementation Plan

Current implementation status:

- Phase 0-2 scheduler groundwork is implemented: host page budgets, tiered
  paged-cache snapshots, grouped transfer payloads, host allocators/stats, and
  D2H offload with ack-gated device release. When a group host pool is full,
  the scheduler evicts older L2 copies for that group before falling back to
  pruning or admission failure; protected loadback sources are skipped. A D2H
  ack publishes host residency immediately, but it keeps the device copy when
  the snapshot became pinned after writeback scheduling; later pressure can
  demote the dual-resident copy once it is no longer borrowed.
- Phase 3 scheduler loadback is implemented for snapshot-complete host hits:
  host-only or deeper host paged-cache hits allocate destination group device
  pages, emit group-keyed H2D transfers, and populate forward metadata only
  from device page ids. If a later admission step defers the forward pass for
  D2H writeback or capacity reasons, the newly materialized device residency is
  rolled back so uninitialized H2D destinations never become visible hits.
  Active H2D loadback now carries a narrow `PagedCacheSnapshotRef` guard so
  offload, pruning, and host eviction skip the source snapshot until the
  transfer op completes or is abandoned. Failed H2D loadback acknowledgements
  also roll back the materialized device residency, increment the per-group hard
  loadback failure counter, and abort the dependent request so request-local
  borrowed tables cannot retain invalid destination page ids.
- Phase 4 runtime groundwork is implemented for direct V4 group transfers:
  DeepSeek V4 has per-group host transfer pools, group-keyed HostExecutor
  dispatch, and the event loop no longer hard-rejects V4 kvstore. HostExecutor
  now converts completed H2D loadback acks into op-level `PrefetchDone` events,
  after all group/kind fragments for the op finish, so scheduler loadback
  guards and runtime inflight accounting are released deterministically. V4 host
  budgets are derived from each group's usable device pages and then add the
  dummy page required by the scheduler allocator; a zero ratio leaves that group
  with `host_total_pages = 0`, which disables L2 for the group. L3 storage
  remains disabled for V4 in this first implementation.
- Phase 5 observability groundwork is partially implemented: each paged-cache
  group reports scheduled D2H pages, scheduled H2D pages, host-evicted pages,
  and hard loadback failures through the scheduler stats API.

### Phase 0: Type and Binding Skeleton

- Add `host_total_pages` to `PagedCacheGroupConfig`.
- Extend scheduler Python bindings and `pool_to_paged_cache_groups()`.
- Add grouped paged-cache transfer fields to C++ operations and Python flat ops.
- Add tests that round-trip group-keyed transfer payloads.

Acceptance:

- Existing KV and Mamba transfer tests still pass.
- V4 config can carry host budgets without changing behavior.

### Phase 1: Snapshot Tiered Residency

- Rename or migrate `PagedCacheGroupSnapshot::pages` to `device_pages`.
- Add `host_pages` and tier-aware helper methods.
- Add tier-aware completeness checks for device and host.
- Add `PagedCacheSnapshotRef` or equivalent reader guard.

Acceptance:

- Existing V4 prefix-cache tests pass with device-only snapshots.
- New tests cover host-only and dual-resident snapshot bookkeeping.

### Phase 2: Scheduler Offload

- Add host group allocators.
- Add `TryOffloadPagedCacheSnapshot()` called from admission shortfall handling.
- Emit D2H `PagedCacheTransferGroup` payloads.
- Track pending writebacks and apply ack by publishing host residency and
  releasing device pages.

Acceptance:

- Device pressure offloads the oldest eligible snapshot before pruning.
- Pending writeback does not expose a host hit.
- Device pages are released only after ack.

### Phase 3: Scheduler Loadback

- Extend paged-cache matching to report host-recoverable hits.
- Allocate destination device group pages for host hits.
- Emit H2D `PagedCacheTransferGroup` payloads.
- Populate forward metadata only with destination device pages.

Acceptance:

- Host-only paged-cache hit schedules loadback and then forwards with device ids.
- Host pages remain resident after loadback.
- Active host snapshots cannot be evicted during loadback.
- Failed H2D loadback releases destination device pages and leaves the snapshot
  host-resident for a later retry.

### Phase 4: V4 Runtime Host Pool

- Add `DeepseekV4PagedCacheHostPool`.
- Add group-aware HostExecutor dispatch.
- Implement per-group, per-layer D2H/H2D copy.
- Remove the V4 `enable_kvstore` hard block only after the host pool path exists.

Acceptance:

- Unit tests validate host tensor shapes for SWA, compressed KV, compressor
  state, indexer KV, and indexer state.
- A small V4 request can perform one writeback and one loadback without metadata
  corruption.

### Phase 5: Integration and Performance

- Add end-to-end tests with constrained device group pages.
- Add counters and debug logs for per-group offload/loadback.
- Benchmark with and without V4 L2 under long-prefix workloads.
- Optimize copy batching only after correctness is stable.

Acceptance:

- L2 reduces admission failures under constrained L1 capacity.
- L1-hit latency is unchanged within measurement noise.
- No leaked pages after finish, retract, abort, writeback ack, or host eviction.

## Test Plan

C++ scheduler tests:

- `test_v4_paged_cache_l2_snapshot.cpp`
- `test_v4_paged_cache_l2_offload.cpp`
- `test_v4_paged_cache_l2_loadback.cpp`
- `test_v4_paged_cache_l2_pinning.cpp`
- `test_v4_paged_cache_l2_transfer_flatten.cpp`

Python runtime tests:

- V4 host pool shape and page indexing.
- HostExecutor group-keyed writeback/loadback dispatch.
- Event loop no longer rejects V4 kvstore after the pool is configured.
- Forward metadata contains device page ids after loadback.

Stress and integration:

- Small device page budget to force offload.
- Small host page budget to force host eviction and fallback.
- Finish/retract while writeback is in flight.
- Mixed V4 paged-cache plus Mamba L2 transfer.

## Risks and Open Decisions

- Host budget sizing needs a first default. Start with proportional per-group
  counts from device sizing, then tune from benchmarks.
- Transfer kernel reuse must be verified per V4 buffer shape. Do not assume the
  ordinary MLA KV kernel handles compressor state or indexer state.
- Snapshot guard implementation must stay narrow. If it starts duplicating
  `NodeResource` semantics, revisit the ownership model.
- State-family offload granularity can be optimized later. First preserve
  correctness and required-group completeness.

## Non-Goals to Keep the First PR Small

- Do not add a new `HostPageAllocator` for ordinary KV.
- Do not rewrite normal KV L2 scheduling.
- Do not introduce broad `CacheFamily` concepts.
- Do not migrate admission results to `std::expected` in this feature branch.
- Do not route V4 paged-cache host pages through `TreeNode::host_resource_`.
