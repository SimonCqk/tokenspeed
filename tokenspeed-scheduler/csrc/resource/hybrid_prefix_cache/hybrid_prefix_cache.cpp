// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/page_container.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/paged_cache_snapshot.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/operations/forward.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace tokenspeed {
namespace {

constexpr std::string_view kPerfDebugTag = "[HybridCachePerfDebug]";

bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return false;
    const std::string_view flag{value};
    return flag != "0" && flag != "false" && flag != "FALSE" && flag != "off" && flag != "OFF";
}

std::int64_t ElapsedUs(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now()) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

std::int32_t NodeDepthTokens(const TreeNode* node) {
    return node == nullptr ? -1 : static_cast<std::int32_t>(node->DepthInTokens());
}

const char* MatchIntentName(MatchIntent intent) {
    switch (intent) {
        case MatchIntent::PrefixReuse:
            return "prefix_reuse";
        case MatchIntent::StateRecovery:
            return "state_recovery";
    }
    return "unknown";
}

}  // namespace

bool HybridPrefixCache::PerfDebugEnabled() {
    static const bool enabled = EnvFlagEnabled("TS_HYBRID_CACHE_PERF_DEBUG");
    return enabled;
}

RequestLocalCacheState::RequestLocalCacheState(std::unique_ptr<LocalKVAllocator> local_kv_allocator,
                                               std::unique_ptr<LocalMambaAllocator> adjunct_state)
    : local_kv_allocator_{std::move(local_kv_allocator)}, adjunct_state_{std::move(adjunct_state)} {
    if (local_kv_allocator_ == nullptr) {
        throw std::invalid_argument("RequestLocalCacheState requires local KV state");
    }
}

RequestLocalCacheState::~RequestLocalCacheState() = default;
RequestLocalCacheState::RequestLocalCacheState(RequestLocalCacheState&&) noexcept = default;
RequestLocalCacheState& RequestLocalCacheState::operator=(RequestLocalCacheState&&) noexcept = default;

std::vector<std::int32_t> RequestLocalCacheState::OccupiedPages(const TreeNode* device_node) const {
    return PageContainer{device_node, local_kv_allocator_.get()}.Pages();
}

void RequestLocalCacheState::AcquireKV(std::int32_t tokens) {
    local_kv_allocator_->Acquire(tokens);
}

OwnedPages RequestLocalCacheState::TakeFullKVPages() {
    return local_kv_allocator_->TakeFullPages();
}

OwnedPages RequestLocalCacheState::TakeFirstKVPages(std::int32_t n) {
    return local_kv_allocator_->TakeFirst(n);
}

std::vector<std::int32_t> RequestLocalCacheState::LocalKVPages() const {
    return local_kv_allocator_->Pages();
}

std::int32_t RequestLocalCacheState::TailPageAvailableTokens() const {
    return local_kv_allocator_->TailPageAvailableTokens();
}

LocalMambaAllocator* RequestLocalCacheState::AdjunctState() {
    return adjunct_state_.get();
}

const LocalMambaAllocator* RequestLocalCacheState::AdjunctState() const {
    return adjunct_state_.get();
}

void RequestLocalCacheState::SetAdjunctState(std::unique_ptr<LocalMambaAllocator> adjunct_state) {
    adjunct_state_ = std::move(adjunct_state);
}

void RequestLocalCacheState::ResetAdjunctState() {
    adjunct_state_.reset();
}

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, PageAllocator& device_allocator,
                                     MambaChunkAllocator* mamba_allocator, std::int32_t mamba_cache_chunk_size,
                                     MambaHostAllocator* mamba_host_allocator)
    : kv_prefix_cache_{kv_prefix_cache},
      device_allocator_{device_allocator},
      mamba_allocator_{mamba_allocator},
      mamba_host_allocator_{mamba_host_allocator},
      mamba_eviction_manager_{mamba_allocator},
      mamba_cache_chunk_size_{mamba_cache_chunk_size} {
    if (HasMambaAdjunct() || mamba_host_allocator_ != nullptr) {
        EnsureKvEvictionCallbacksInstalled();
    }
    if (PerfDebugEnabled()) {
        spdlog::info("{} init mamba_adjunct={} mamba_l2={} page_size={} mamba_chunk_size={} kv_eviction_callbacks={}",
                     kPerfDebugTag, HasMambaAdjunct(), mamba_host_allocator_ != nullptr, kv_prefix_cache_.PageSize(),
                     mamba_cache_chunk_size_, has_kv_eviction_callbacks_);
    }
}

HybridPrefixCache::~HybridPrefixCache() {
    SetKvEventSink({});
    if (has_kv_eviction_callbacks_) {
        kv_prefix_cache_.GetDeviceManager().SetEvictionCallback({});
        kv_prefix_cache_.GetHostManager().SetEvictionCallback({});
        has_kv_eviction_callbacks_ = false;
    }
}

RecoveryPlan HybridPrefixCache::MatchPrefix(const std::vector<std::span<const std::int32_t>>& token_pages,
                                            MatchIntent intent) {
    const auto start = std::chrono::steady_clock::now();
    MatchResult raw_match = kv_prefix_cache_.Match(token_pages, intent);
    const auto raw_done = std::chrono::steady_clock::now();
    const std::int32_t raw_device_pages = raw_match.device.DepthInPage();
    const std::int32_t raw_host_pages = raw_match.host.DepthInPage();
    const std::int32_t raw_device_tokens = NodeDepthTokens(raw_match.device.last_node);
    const std::int32_t raw_host_tokens = NodeDepthTokens(raw_match.host.last_node);
    RecoveryPlan plan = BuildRecoveryPlan(std::move(raw_match), intent);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} MatchPrefix intent={} token_pages={} kv_match_us={} augment_us={} raw_device_pages={} "
            "raw_host_pages={} raw_device_tokens={} raw_host_tokens={} compat_device_pages={} compat_host_pages={} "
            "compat_device_tokens={} compat_host_tokens={} paged_prefix_tokens={} paged_history_tokens={} "
            "paged_restore_kind={} mamba_cow={} mamba_host={} mamba_branch={} recovery_ok={} protected_depth={}",
            kPerfDebugTag, MatchIntentName(intent), token_pages.size(), ElapsedUs(start, raw_done), ElapsedUs(raw_done),
            raw_device_pages, raw_host_pages, raw_device_tokens, raw_host_tokens,
            plan.compat_match.device.DepthInPage(), plan.compat_match.host.DepthInPage(),
            NodeDepthTokens(plan.compat_match.device.last_node), NodeDepthTokens(plan.compat_match.host.last_node),
            plan.compat_match.paged_cache.prefix_len_tokens, plan.compat_match.paged_cache.history_hit_tokens,
            static_cast<int>(plan.compat_match.paged_cache.restore_kind), plan.compat_match.mamba_cow_src_index,
            plan.compat_match.mamba_host_src_index, plan.compat_match.mamba_branching_seqlen,
            plan.recovery_state_available, NodeDepthTokens(plan.protected_recovery_node));
    }
    return plan;
}

RecoveryPlan HybridPrefixCache::BuildRecoveryPlan(MatchResult raw_match, MatchIntent intent) const {
    RecoveryPlan plan{};
    if (intent == MatchIntent::StateRecovery) {
        plan.compat_match = std::move(raw_match);
        augmentMatch(plan.compat_match);
        augmentMatchPagedCache(plan.compat_match);

        const DecodeFromRetractedRecovery recovery = PrepareDecodeFromRetractedRecovery(plan.compat_match);
        plan.recovery_state_available = recovery.ok;
        plan.protected_recovery_node = recovery.protected_source_node;
        return plan;
    }

    plan.compat_match = std::move(raw_match);
    augmentMatch(plan.compat_match);
    augmentMatchPagedCache(plan.compat_match);
    return plan;
}

HybridPrefixCache::RawHostStorageHashSeed HybridPrefixCache::LookupRawHostStorageHashSeed(
    const std::vector<std::span<const std::int32_t>>& token_pages) {
    MatchResult match = kv_prefix_cache_.Match(token_pages);
    const std::int32_t host_matched_pages = match.host.DepthInPage();
    const auto& page_hashes = match.host.last_node->PageHashes();
    return RawHostStorageHashSeed{
        .host_matched_pages = host_matched_pages,
        .prior_hash_seed = page_hashes.empty() ? std::string{} : page_hashes.back(),
    };
}

cache_op_id HybridPrefixCache::AllocateCacheOpId() {
    return kv_prefix_cache_.AllocateCacheOpId();
}

void HybridPrefixCache::SetKvEventSink(KvEventSink sink) {
    if (!sink) {
        if (has_facade_kv_event_sink_) {
            kv_prefix_cache_.SetKvEventSink({});
            has_facade_kv_event_sink_ = false;
        }
        return;
    }

    kv_prefix_cache_.SetKvEventSink(std::move(sink));
    has_facade_kv_event_sink_ = true;
}

void HybridPrefixCache::EnsureKvEvictionCallbacksInstalled() {
    if (has_kv_eviction_callbacks_) return;
    kv_prefix_cache_.GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { OnKVEvict(node); });
    kv_prefix_cache_.GetHostManager().SetEvictionCallback([this](TreeNode* node) { OnKVHostEvict(node); });
    has_kv_eviction_callbacks_ = true;
}

cache::publish::DevicePrefix::Result HybridPrefixCache::Apply(const cache::publish::DevicePrefix& op) {
    const auto start = std::chrono::steady_clock::now();
    cache::publish::DevicePrefix::Result result{};
    if (op.device_node_ref == nullptr) {
        throw std::invalid_argument("HybridPrefixCache::Apply(DevicePrefix) requires device_node_ref");
    }
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(op.device_node_ref->Node());
    const std::int32_t new_page_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
    const std::int32_t page_size = kv_prefix_cache_.PageSize();
    const std::int32_t last_inserted_len = static_cast<std::int32_t>(op.full_paged_tokens.size()) * page_size;
    LocalMambaAllocator* adjunct_state = op.local_cache.AdjunctState();
    const bool had_checkpoint = adjunct_state != nullptr && adjunct_state->HasCheckpoint();
    const std::int32_t checkpoint_position = had_checkpoint ? adjunct_state->CheckpointPosition() : -1;
    auto should_publish_mamba_checkpoint = [&]() {
        if (!had_checkpoint) {
            return false;
        }
        if (checkpoint_position < 0 || checkpoint_position == last_inserted_len) {
            return true;
        }
        if (!op.chunk_begin.has_value() || page_size <= 0) {
            return false;
        }
        const std::int32_t chunk_begin = *op.chunk_begin;
        if (last_inserted_len <= chunk_begin || last_inserted_len >= checkpoint_position) {
            return false;
        }
        const std::int32_t track_len = last_inserted_len - chunk_begin;
        return AlignMambaCacheSeqlen(track_len) == track_len;
    };
    const bool publish_mamba_checkpoint = should_publish_mamba_checkpoint();
    if (new_page_count <= 0) {
        if (adjunct_state != nullptr && adjunct_state->HasCheckpoint()) {
            (void)adjunct_state->DetachCheckpoint();
        }
        if (PerfDebugEnabled()) {
            spdlog::info(
                "{} ApplyDevicePrefix no_insert full_pages={} prefix_pages={} checkpoint={} checkpoint_pos={} "
                "publish_checkpoint={} elapsed_us={}",
                kPerfDebugTag, op.full_paged_tokens.size(), prefix_pages.size(), had_checkpoint, checkpoint_position,
                publish_mamba_checkpoint, ElapsedUs(start));
        }
        return result;
    }

    OwnedPages pages_to_insert = op.local_cache.TakeFirstKVPages(new_page_count);
    auto insert_result =
        kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(pages_to_insert));

    if (adjunct_state != nullptr && adjunct_state->HasCheckpoint()) {
        std::unique_ptr<MambaSlot> checkpoint = adjunct_state->DetachCheckpoint();
        if (publish_mamba_checkpoint) {
            InsertMamba(insert_result.last_node, std::move(checkpoint));
        }
    }
    op.device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
    result.device_insert_page_count = new_page_count;
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} ApplyDevicePrefix inserted_pages={} full_pages={} prefix_pages={} terminal_depth={} checkpoint={} "
            "checkpoint_pos={} publish_checkpoint={} elapsed_us={}",
            kPerfDebugTag, new_page_count, op.full_paged_tokens.size(), prefix_pages.size(),
            NodeDepthTokens(insert_result.last_node), had_checkpoint, checkpoint_position, publish_mamba_checkpoint,
            ElapsedUs(start));
    }
    return result;
}

cache::publish::FinishedRequest::Result HybridPrefixCache::Apply(const cache::publish::FinishedRequest& op) {
    const auto start = std::chrono::steady_clock::now();
    cache::publish::FinishedRequest::Result result{};
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(&op.current_device_node);
    const std::int32_t alloc_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    if (alloc_count > 0) {
        OwnedPages alloc_pages = op.local_cache.TakeFirstKVPages(alloc_count);
        kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                      op.page_hashes);
    }
    if (alloc_count >= 0 && HasMambaAdjunct()) {
        PublishFinishMambaState(op.full_paged_tokens, op.local_cache);
    }

    result.device_insert_page_count = std::max(0, alloc_count);
    result.match_result = kv_prefix_cache_.Match(op.full_paged_tokens);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} ApplyFinishedRequest alloc_count={} full_pages={} prefix_pages={} result_device_pages={} "
            "result_host_pages={} result_device_tokens={} elapsed_us={}",
            kPerfDebugTag, alloc_count, op.full_paged_tokens.size(), prefix_pages.size(),
            result.match_result.device.DepthInPage(), result.match_result.host.DepthInPage(),
            NodeDepthTokens(result.match_result.device.last_node), ElapsedUs(start));
    }
    return result;
}

cache::publish::RetractPrefixCommit::Result HybridPrefixCache::Apply(const cache::publish::RetractPrefixCommit& op) {
    const auto start = std::chrono::steady_clock::now();
    cache::publish::RetractPrefixCommit::Result result{};
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(&op.current_device_node);
    const std::int32_t alloc_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
    if (alloc_count < 0) {
        throw std::logic_error(
            "HybridPrefixCache::Apply(RetractPrefixCommit): current device prefix exceeds available full token pages");
    }
    OwnedPages pages_to_insert = op.local_cache.TakeFirstKVPages(alloc_count);
    if (pages_to_insert.Size() != alloc_count) {
        throw std::logic_error("HybridPrefixCache::Apply(RetractPrefixCommit): request-local page count mismatch");
    }

    kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(pages_to_insert));
    result.device_insert_page_count = alloc_count;
    result.match_result = kv_prefix_cache_.Match(op.full_paged_tokens, MatchIntent::StateRecovery);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} ApplyRetractPrefixCommit alloc_count={} full_pages={} prefix_pages={} result_device_pages={} "
            "result_host_pages={} result_device_tokens={} elapsed_us={}",
            kPerfDebugTag, alloc_count, op.full_paged_tokens.size(), prefix_pages.size(),
            result.match_result.device.DepthInPage(), result.match_result.host.DepthInPage(),
            NodeDepthTokens(result.match_result.device.last_node), ElapsedUs(start));
    }
    return result;
}

cache::materialize::PrefixOnDevice::Result HybridPrefixCache::Apply(const cache::materialize::PrefixOnDevice& op) {
    const auto start = std::chrono::steady_clock::now();
    cache::materialize::PrefixOnDevice::Result result{};
    const std::vector<TreeNode*> nodes = op.compat_match.NodesWithout<ResourceType::Device>();
    if (op.require_all_pages) {
        result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(nodes);
        if (PerfDebugEnabled()) {
            spdlog::info("{} MaterializePrefixOnDevice require_all=1 nodes={} ok={} elapsed_us={}", kPerfDebugTag,
                         nodes.size(), result.ok, ElapsedUs(start));
        }
        return result;
    }
    (void)kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(nodes);
    if (PerfDebugEnabled()) {
        spdlog::info("{} MaterializePrefixOnDevice require_all=0 nodes={} elapsed_us={}", kPerfDebugTag, nodes.size(),
                     ElapsedUs(start));
    }
    return result;
}

cache::materialize::HostWritebackPages::Result HybridPrefixCache::Apply(
    const cache::materialize::HostWritebackPages& op) {
    const auto start = std::chrono::steady_clock::now();
    cache::materialize::HostWritebackPages::Result result{};
    std::int32_t host_pages_num = 0;
    if (op.ensure_capacity_before_allocate) {
        for (TreeNode* node : op.write_diff) {
            host_pages_num += node->Device().NumPages();
        }
        if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
            result.ok = false;
            if (PerfDebugEnabled()) {
                spdlog::info("{} MaterializeHostWriteback capacity_failed nodes={} host_pages_needed={} elapsed_us={}",
                             kPerfDebugTag, op.write_diff.size(), host_pages_num, ElapsedUs(start));
            }
            return result;
        }
    }
    result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Host>(op.write_diff);
    if (!result.ok) return result;
    result.cache_transfer_pairs = PrepareMambaHostWriteBack(op.write_diff);
    for (const TransferPair& transfer : result.cache_transfer_pairs) {
        if (transfer.kind != CacheKind::kMamba) continue;
        for (TreeNode* node : op.write_diff) {
            if (node != nullptr && node->HasMamba() && node->MambaSlotIndex() == transfer.src) {
                result.pending_state.adjunct_nodes.push_back(node);
                break;
            }
        }
    }
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} MaterializeHostWriteback nodes={} host_pages_needed={} ensure_capacity={} ok={} transfers={} "
            "mamba_pending_nodes={} elapsed_us={}",
            kPerfDebugTag, op.write_diff.size(), host_pages_num, op.ensure_capacity_before_allocate, result.ok,
            result.cache_transfer_pairs.size(), result.pending_state.adjunct_nodes.size(), ElapsedUs(start));
    }
    return result;
}

void HybridPrefixCache::OnKVEvict(TreeNode* node) {
    if (node == nullptr) return;
    if (mamba_allocator_ != nullptr && node->HasMamba()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
    // Passive paged-cache detach on KV LRU drop: returns OwnedPages via RAII;
    // the chain scan sees the gap because `HasPagedCacheSnapshot()` is false.
    // Route through DetachPagedCacheSnapshotFromNode to keep membership set in sync.
    if (node->HasPagedCacheSnapshot()) {
        DetachPagedCacheSnapshotFromNode(node);
    }
    if (PerfDebugEnabled()) {
        spdlog::info("{} OnKVEvict depth={} had_mamba={} had_paged_snapshot={}", kPerfDebugTag, NodeDepthTokens(node),
                     node->HasMamba(), node->HasPagedCacheSnapshot());
    }
}

void HybridPrefixCache::OnKVHostEvict(TreeNode* node) {
    if (node == nullptr || mamba_host_allocator_ == nullptr) return;
    pending_mamba_host_writebacks_.erase(node);
    if (node->HasMambaOnHost()) {
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
        mamba_host_writeback_done_nodes_.erase(node);
    }
}

void HybridPrefixCache::DemoteIdleMambaDeviceCopiesPresentOnHost() {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    const auto start = std::chrono::steady_clock::now();
    std::int32_t demoted = 0;
    std::int32_t stale = 0;
    std::int32_t already_host_only = 0;
    std::int32_t device_ref_pinned = 0;
    std::vector<TreeNode*> nodes(mamba_host_writeback_done_nodes_.begin(), mamba_host_writeback_done_nodes_.end());
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.erase(node);
            ++stale;
            continue;
        }
        if (!node->HasMamba()) {
            mamba_host_writeback_done_nodes_.erase(node);
            ++already_host_only;
            continue;
        }
        if (node->OnDevice() && node->Device().RefCount() != 0) {
            ++device_ref_pinned;
            continue;
        }
        OnKVDeviceDemote(node);
        mamba_host_writeback_done_nodes_.erase(node);
        ++demoted;
    }
    if (demoted > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] demoted device copies after host writeback count={}", demoted);
    }
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} DemoteIdleMamba scan_nodes={} demoted={} stale={} already_host_only={} device_ref_pinned={} "
            "remaining_done_nodes={} elapsed_us={}",
            kPerfDebugTag, nodes.size(), demoted, stale, already_host_only, device_ref_pinned,
            mamba_host_writeback_done_nodes_.size(), ElapsedUs(start));
    }
}

void HybridPrefixCache::OnMambaHostWriteBackDone(TreeNode* last_node) {
    if (last_node == nullptr) return;
    std::vector<TreeNode*> nodes;
    for (TreeNode* node : LeafToRoot(last_node)) {
        if (node == nullptr || !node->OnHost()) break;
        nodes.push_back(node);
    }
    OnMambaHostWriteBackDone(nodes);
}

void HybridPrefixCache::OnMambaHostWriteBackDone(const std::vector<TreeNode*>& nodes) {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    std::int32_t attached = 0;
    std::int32_t completed = 0;
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->OnHost()) continue;
        auto pending = pending_mamba_host_writebacks_.find(node);
        if (pending != pending_mamba_host_writebacks_.end()) {
            node->AttachMambaHost(std::move(pending->second));
            pending_mamba_host_writebacks_.erase(pending);
            mamba_host_nodes_.insert(node);
            ++attached;
        }
        if (node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.insert(node);
            ++completed;
        }
    }
    if (attached > 0 || completed > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] host writeback done attach_count={} completed_nodes={}", attached,
                      completed);
    }
    if (PerfDebugEnabled()) {
        spdlog::info("{} OnMambaHostWriteBackDone input_nodes={} attached={} completed={} pending={} done_nodes={}",
                     kPerfDebugTag, nodes.size(), attached, completed, pending_mamba_host_writebacks_.size(),
                     mamba_host_writeback_done_nodes_.size());
    }
    DemoteIdleMambaDeviceCopiesPresentOnHost();
}

void HybridPrefixCache::OnKVDeviceDemote(TreeNode* node) {
    if (node == nullptr || mamba_allocator_ == nullptr) return;
    if (node->HasMamba() && node->HasMambaOnHost()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
}

void HybridPrefixCache::CompleteHostWriteBack(const PendingHostWritebackState& pending_state, TreeNode* device_node) {
    const auto start = std::chrono::steady_clock::now();
    OnMambaHostWriteBackDone(pending_state.adjunct_nodes);
    if (device_node != nullptr) {
        kv_prefix_cache_.ReleaseDeviceResourcesPresentOnHost(device_node,
                                                             [this](TreeNode* node) { OnKVDeviceDemote(node); });
    }
    DemoteIdleMambaDeviceCopiesPresentOnHost();
    if (PerfDebugEnabled()) {
        spdlog::info("{} CompleteHostWriteBack adjunct_nodes={} device_node_depth={} elapsed_us={}", kPerfDebugTag,
                     pending_state.adjunct_nodes.size(), NodeDepthTokens(device_node), ElapsedUs(start));
    }
}

std::size_t HybridPrefixCache::AvailableDevicePages() const {
    return static_cast<std::size_t>(device_allocator_.AvailablePages());
}

std::vector<std::string> HybridPrefixCache::PagedCacheGroupIds() const {
    std::vector<std::string> ids;
    ids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, _] : paged_cache_allocators_) {
        ids.push_back(gid);
    }
    return ids;
}

std::int32_t HybridPrefixCache::PagedCacheGroupTotalPages(std::string_view group_id) const {
    auto alloc_it = paged_cache_allocators_.find(group_id);
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupTotalPages: group_id not configured");
    }
    return alloc_it->second->TotalPages();
}

std::int32_t HybridPrefixCache::PagedCacheGroupAvailablePages(std::string_view group_id) const {
    auto alloc_it = paged_cache_allocators_.find(group_id);
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return alloc_it->second->AvailablePages();
}

std::int64_t HybridPrefixCache::PagedCacheGroupFailedAllocCount(std::string_view group_id) const {
    auto alloc_it = paged_cache_allocators_.find(group_id);
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return alloc_it->second->FailedAllocCount();
}

std::vector<std::int32_t> HybridPrefixCache::GetRequestPagedCachePageIds(std::string_view request_id,
                                                                         std::string_view group_id) const {
    (void)PagedCacheGroupTotalPages(group_id);
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) return {};
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) return {};
    return group_it->second.PageIds();
}

std::int32_t HybridPrefixCache::GetRequestPagedCacheBaseLogicalPage(std::string_view request_id,
                                                                    std::string_view group_id) const {
    (void)PagedCacheGroupTotalPages(group_id);
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) return 0;
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) return 0;
    return group_it->second.BaseLogicalPage();
}

CacheStatsSnapshot HybridPrefixCache::Stats(const StatsRequest& request) const {
    CacheStatsSnapshot snapshot{
        .available_device_pages = AvailableDevicePages(),
    };

    snapshot.paged_cache_group_ids = PagedCacheGroupIds();

    std::vector<std::string> requested_groups = request.paged_cache_group_ids;
    if (requested_groups.empty()) {
        requested_groups = snapshot.paged_cache_group_ids;
    }
    for (const auto& gid : requested_groups) {
        snapshot.paged_cache_total_pages[gid] = PagedCacheGroupTotalPages(gid);
        snapshot.paged_cache_available_pages[gid] = PagedCacheGroupAvailablePages(gid);
        snapshot.paged_cache_failed_alloc_count[gid] = PagedCacheGroupFailedAllocCount(gid);

        if (request.request_id.has_value()) {
            snapshot.request_paged_cache_page_ids[gid] = GetRequestPagedCachePageIds(*request.request_id, gid);
            snapshot.request_paged_cache_base_logical_page[gid] =
                GetRequestPagedCacheBaseLogicalPage(*request.request_id, gid);
        }
    }

    if (request.include_device_memory_diagnostics) {
        snapshot.device_memory_diagnostics = CacheDeviceMemoryDiagnosticsSnapshot{
            .tree_device_pages = kv_prefix_cache_.CollectAllPages<ResourceType::Device>(),
            .free_device_pages = device_allocator_.AvailablePages(),
            .total_device_pages = device_allocator_.TotalPages() - 1,
        };
    }

    return snapshot;
}

cache::worker::CommitPrefillFirstChunkMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitPrefillFirstChunkMetadata& op) {
    const auto start = std::chrono::steady_clock::now();
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    PopulateMambaMatchCompatibilityFields(op.op_base, op.match_result);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         op.match_result.paged_cache);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} WorkerCommitPrefillFirst request={} first={} target={} match_device_pages={} match_host_pages={} "
            "paged_prefix={} op_pages={} elapsed_us={}",
            kPerfDebugTag, op.op_base.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
            op.match_result.device.DepthInPage(), op.match_result.host.DepthInPage(),
            op.match_result.paged_cache.prefix_len_tokens, op.op_base.occupied_pages.size(), ElapsedUs(start));
    }
    return {};
}

cache::worker::CommitPrefillMetadata::Result HybridPrefixCache::Apply(const cache::worker::CommitPrefillMetadata& op) {
    const auto start = std::chrono::steady_clock::now();
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    if (PerfDebugEnabled()) {
        spdlog::info("{} WorkerCommitPrefill request={} first={} target={} op_pages={} elapsed_us={}", kPerfDebugTag,
                     op.op_base.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                     op.op_base.occupied_pages.size(), ElapsedUs(start));
    }
    return {};
}

cache::worker::CommitDecodeAfterPrefillMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitDecodeAfterPrefillMetadata& op) {
    const auto start = std::chrono::steady_clock::now();
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    if (PerfDebugEnabled()) {
        spdlog::info("{} WorkerCommitDecodeAfterPrefill request={} first={} target={} op_pages={} elapsed_us={}",
                     kPerfDebugTag, op.op_base.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                     op.op_base.occupied_pages.size(), ElapsedUs(start));
    }
    return {};
}

cache::worker::CommitDecodeMetadata::Result HybridPrefixCache::Apply(const cache::worker::CommitDecodeMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    return {};
}

cache::worker::CommitDecodeRecoveryMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitDecodeRecoveryMetadata& op) {
    const auto start = std::chrono::steady_clock::now();
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    PopulateMambaRecoveryCompatibilityFields(op.op_base, op.match_result);
    ReleaseRequest(op.op_base.request_id);
    acquireAndPopulateOp(op.op_base, 0, op.target_raw_tokens_exclusive, op.match_result.paged_cache);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} WorkerCommitDecodeRecovery request={} target={} match_device_pages={} match_host_pages={} "
            "paged_prefix={} op_pages={} elapsed_us={}",
            kPerfDebugTag, op.op_base.request_id, op.target_raw_tokens_exclusive, op.match_result.device.DepthInPage(),
            op.match_result.host.DepthInPage(), op.match_result.paged_cache.prefix_len_tokens,
            op.op_base.occupied_pages.size(), ElapsedUs(start));
    }
    return {};
}

std::vector<TreeNode*> HybridPrefixCache::MambaDeviceLoadbackNodes(const MatchResult& match_result,
                                                                   TreeNode* preferred_source) const {
    std::vector<TreeNode*> nodes;
    if (mamba_host_allocator_ == nullptr || match_result.mamba_host_src_index < 0 ||
        match_result.mamba_cow_src_index >= 0) {
        return nodes;
    }
    TreeNode* host_mamba_node =
        preferred_source != nullptr ? preferred_source : FindLastMambaHostNode(match_result.host.last_node);
    if (host_mamba_node != nullptr && host_mamba_node->HasMambaOnHost() && !host_mamba_node->HasMamba()) {
        nodes.push_back(host_mamba_node);
    }
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} MambaDeviceLoadbackNodes host_src={} cow_src={} preferred_depth={} host_match_depth={} "
            "loadback_nodes={}",
            kPerfDebugTag, match_result.mamba_host_src_index, match_result.mamba_cow_src_index,
            NodeDepthTokens(preferred_source), NodeDepthTokens(match_result.host.last_node), nodes.size());
    }
    return nodes;
}

cache::admit::PrefillFirstChunk::Result HybridPrefixCache::Apply(const cache::admit::PrefillFirstChunk& op,
                                                                 std::map<std::string, std::int32_t>& simulated_free) {
    const auto start = std::chrono::steady_clock::now();
    cache::admit::PrefillFirstChunk::Result result{};
    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(op.match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitPrefillFirst denied=request_kv_capacity request={} device_pages_needed={}",
                         kPerfDebugTag, op.request_id, op.device_pages_needed);
        }
        return result;
    }
    std::vector<TreeNode*> loadback_nodes = MambaDeviceLoadbackNodes(op.match_result);
    std::optional<std::int32_t> mamba_branching_seqlen;
    if (HasMambaAdjunct()) {
        if (op.match_result.mamba_branching_seqlen == -1) {
            const std::int32_t aligned = AlignMambaCacheSeqlen(op.tokens_this_round);
            if (aligned > 0) {
                mamba_branching_seqlen = aligned;
            }
        }
        const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
        if (!EnsureMambaCapacityByEvict(slots_needed)) {
            if (PerfDebugEnabled()) {
                spdlog::info("{} AdmitPrefillFirst denied=mamba_capacity request={} slots_needed={}", kPerfDebugTag,
                             op.request_id, slots_needed);
            }
            return result;
        }
    }
    result.admitted = AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                                 simulated_free, op.match_result.paged_cache);
    if (result.admitted) {
        if (mamba_branching_seqlen.has_value()) {
            op.match_result.mamba_branching_seqlen = *mamba_branching_seqlen;
        }
        result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
        if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
            op.match_result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
        }
    }
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} AdmitPrefillFirst request={} admitted={} device_pages_needed={} tokens_this_round={} first={} "
            "target={} match_device_pages={} match_host_pages={} loadback_nodes={} transfers={} "
            "mamba_branching={} elapsed_us={}",
            kPerfDebugTag, op.request_id, result.admitted, op.device_pages_needed, op.tokens_this_round,
            op.first_raw_position_of_op, op.target_raw_tokens_exclusive, op.match_result.device.DepthInPage(),
            op.match_result.host.DepthInPage(), loadback_nodes.size(), result.cache_transfer_pairs.size(),
            op.match_result.mamba_branching_seqlen, ElapsedUs(start));
    }
    return result;
}

cache::admit::PrefillChunk::Result HybridPrefixCache::Apply(const cache::admit::PrefillChunk& op,
                                                            std::map<std::string, std::int32_t>& simulated_free) {
    const auto start = std::chrono::steady_clock::now();
    cache::admit::PrefillChunk::Result result{};
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitPrefill denied=request_kv_capacity request={} device_pages_needed={}", kPerfDebugTag,
                         op.request_id, op.device_pages_needed);
        }
        return result;
    }
    if (HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitPrefill denied=mamba_capacity request={} slots_needed=1", kPerfDebugTag,
                         op.request_id);
        }
        return result;
    }
    result.admitted =
        AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive, simulated_free);
    if (PerfDebugEnabled()) {
        spdlog::info("{} AdmitPrefill request={} admitted={} device_pages_needed={} first={} target={} elapsed_us={}",
                     kPerfDebugTag, op.request_id, result.admitted, op.device_pages_needed, op.first_raw_position_of_op,
                     op.target_raw_tokens_exclusive, ElapsedUs(start));
    }
    return result;
}

cache::admit::Decode::Result HybridPrefixCache::Apply(const cache::admit::Decode& op,
                                                      std::map<std::string, std::int32_t>& simulated_free) {
    const auto start = std::chrono::steady_clock::now();
    cache::admit::Decode::Result result{};
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitDecode denied=request_kv_capacity request={} device_pages_needed={}", kPerfDebugTag,
                         op.request_id, op.device_pages_needed);
        }
        return result;
    }
    if (op.refresh_local_cache_state && HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitDecode denied=mamba_capacity request={} refresh_local_cache=1", kPerfDebugTag,
                         op.request_id);
        }
        return result;
    }
    result.admitted =
        AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive, simulated_free);
    if (PerfDebugEnabled() && (!result.admitted || op.device_pages_needed > 0 || op.refresh_local_cache_state)) {
        spdlog::info(
            "{} AdmitDecode request={} admitted={} device_pages_needed={} first={} target={} refresh_local_cache={} "
            "elapsed_us={}",
            kPerfDebugTag, op.request_id, result.admitted, op.device_pages_needed, op.first_raw_position_of_op,
            op.target_raw_tokens_exclusive, op.refresh_local_cache_state, ElapsedUs(start));
    }
    return result;
}

cache::admit::DecodeFromRetracted::Result HybridPrefixCache::Apply(
    const cache::admit::DecodeFromRetracted& op, std::map<std::string, std::int32_t>& simulated_free) {
    const auto start = std::chrono::steady_clock::now();
    cache::admit::DecodeFromRetracted::Result result{};
    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(op.match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        if (PerfDebugEnabled()) {
            spdlog::info("{} AdmitDecodeRetracted denied=request_kv_capacity request={} device_pages_needed={}",
                         kPerfDebugTag, op.request_id, op.device_pages_needed);
        }
        return result;
    }
    std::vector<TreeNode*> loadback_nodes = MambaDeviceLoadbackNodes(op.match_result, op.protected_recovery_node);
    if (HasMambaAdjunct()) {
        if (op.protected_recovery_node == nullptr) {
            if (PerfDebugEnabled()) {
                spdlog::info("{} AdmitDecodeRetracted denied=no_protected_mamba_source request={}", kPerfDebugTag,
                             op.request_id);
            }
            return result;
        }
        // Recovery COWs the tree-owned Mamba state into fresh request-local
        // working/checkpoint slots. Protect the source node only for this
        // allocation; retracted Mamba states are otherwise normal evictable
        // tree-owned cache entries.
        const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
        if (!EnsureMambaCapacityByEvict(slots_needed, op.protected_recovery_node)) {
            if (PerfDebugEnabled()) {
                spdlog::info("{} AdmitDecodeRetracted denied=mamba_capacity request={} slots_needed={}", kPerfDebugTag,
                             op.request_id, slots_needed);
            }
            return result;
        }
    }
    result.admitted = AdmitChunkFromRetracted(op.request_id, op.target_raw_tokens_exclusive, simulated_free,
                                              op.match_result.paged_cache);
    if (result.admitted) {
        result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
        if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
            op.match_result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
        }
    }
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} AdmitDecodeRetracted request={} admitted={} device_pages_needed={} target={} match_device_pages={} "
            "match_host_pages={} paged_prefix={} protected_depth={} loadback_nodes={} transfers={} elapsed_us={}",
            kPerfDebugTag, op.request_id, result.admitted, op.device_pages_needed, op.target_raw_tokens_exclusive,
            op.match_result.device.DepthInPage(), op.match_result.host.DepthInPage(),
            op.match_result.paged_cache.prefix_len_tokens, NodeDepthTokens(op.protected_recovery_node),
            loadback_nodes.size(), result.cache_transfer_pairs.size(), ElapsedUs(start));
    }
    return result;
}

cache::admit::Retract::Result HybridPrefixCache::Apply(const cache::admit::Retract& op,
                                                       std::map<std::string, std::int32_t>&) {
    const auto start = std::chrono::steady_clock::now();
    cache::admit::Retract::Result result{};
    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(op.match_result.host.last_node);
    result.admitted = kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(op.host_pages_needed);
    if (PerfDebugEnabled()) {
        spdlog::info(
            "{} AdmitRetract admitted={} host_pages_needed={} match_device_pages={} match_host_pages={} "
            "elapsed_us={}",
            kPerfDebugTag, result.admitted, op.host_pages_needed, op.match_result.device.DepthInPage(),
            op.match_result.host.DepthInPage(), ElapsedUs(start));
    }
    return result;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult&, cache::EmptyResult) {}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result, cache::publish::DevicePrefix::Result op_result) {
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::publish::FinishedRequest::Result&& op_result) {
    result.match_result = std::move(op_result.match_result);
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::publish::RetractPrefixCommit::Result&& op_result) {
    result.match_result = std::move(op_result.match_result);
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::materialize::PrefixOnDevice::Result op_result) {
    result.ok = op_result.ok;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::materialize::HostWritebackPages::Result&& op_result) {
    result.ok = op_result.ok;
    if (!result.ok) return;
    result.cache_transfer_pairs = std::move(op_result.cache_transfer_pairs);
    result.pending_host_writeback = std::move(op_result.pending_state);
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::state::CreateRequestLocalCache::Result&& op_result) {
    result.local_cache = std::move(op_result.local_cache);
}

}  // namespace tokenspeed
