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

#include "cache/kv_cache_coordinator.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {
namespace {

std::int32_t CheckedLcm(std::int32_t lhs, std::int32_t rhs) {
    _assert(lhs > 0 && rhs > 0, "block alignment inputs must be positive");
    const std::int32_t quotient = lhs / std::gcd(lhs, rhs);
    _assert(quotient <= std::numeric_limits<std::int32_t>::max() / rhs, "block alignment exceeds int32");
    return quotient * rhs;
}

}  // namespace

KvCacheCoordinator::KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                                       BlockPool* host_pool)
    : groups_{std::move(groups)}, pool_{pool}, host_pool_{host_pool}, cache_block_tokens_{cache_block_tokens} {
    _assert(cache_block_tokens_ > 0, "coordinator needs positive cache_block_tokens");
    prefix_alignment_tokens_ = cache_block_tokens_;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Id() == static_cast<GroupId>(i), "cache manager group id must equal its group index");
        const std::int32_t group_block_tokens = groups_[i].Manager().CacheBlockTokens();
        _assert(group_block_tokens > 0 && group_block_tokens % cache_block_tokens_ == 0,
                "every group block span must be a positive multiple of the base hash grain");
        _assert(groups_[i].Spec().block_size == group_block_tokens,
                "cache manager block span must match its group spec");
        _assert(groups_[i].Manager().CacheBlocksPerLcmBlock() == groups_[i].Spec().cache_blocks_per_lcm_block,
                "cache manager packing must match its group spec");
        prefix_alignment_tokens_ = CheckedLcm(prefix_alignment_tokens_, group_block_tokens);
        if (groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
}

bool KvCacheCoordinator::HasMambaStateGroup() const {
    return std::ranges::any_of(groups_,
                               [](const CacheGroup& group) { return group.Spec().kind == AttnKind::kMambaState; });
}

std::vector<CacheKey> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes, GroupId group_id,
                                                       std::int32_t first_base_slot) const {
    _assert(group_id < groups_.size(), "cache key group id out of range");
    const std::int32_t fold = groups_[group_id].Manager().CacheBlockTokens() / cache_block_tokens_;
    // Base hashes are cumulative rolling digests, so a group key is the digest
    // at its block-end boundary; no second hash fold is required.
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size() / static_cast<std::size_t>(fold) + 1);
    for (std::int32_t index = FirstProjectedBaseHashOffset(first_base_slot, fold);
         index < static_cast<std::int32_t>(content_hashes.size()); index += fold) {
        keys.push_back(CacheKey{
            .group_id = group_id,
            .content_hash = content_hashes[static_cast<std::size_t>(index)],
        });
    }
    return keys;
}

namespace {

struct ConvergedBoundary {
    std::int32_t common_tokens{0};
    std::int32_t prefix_closed_tokens{0};
};

// Shared match skeleton: one ordered sweep (closed groups first), then re-match any window
// group left above the settled bound -- with 2+ window groups a later group can shrink the
// bound UNDER an earlier one's boundary-dependent match. A re-matched group lands at or
// under the current bound and only a further bound drop can lift it back above, so
// re-matches are finite; the result is the greatest boundary every group supports.
//
// Bounds align down to the raw-token boundary shared by every group grid.
template <typename MatchGroup, typename ExtentTokens>
ConvergedBoundary SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                                    std::int32_t bound_tokens, std::int32_t align_tokens, const MatchGroup& match,
                                    const ExtentTokens& extent) {
    const auto align_down = [align_tokens](std::int32_t tokens) { return tokens - tokens % align_tokens; };
    bound_tokens = align_down(bound_tokens);
    std::int32_t prefix_closed_tokens = 0;
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, align_down(extent(i)));
        if (groups[i].Manager().MatchIsPrefixClosed()) {
            prefix_closed_tokens = bound_tokens;
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Manager().MatchIsPrefixClosed() || extent(i) <= bound_tokens) {
                continue;
            }
            match(i, bound_tokens);
            bound_tokens = std::min(bound_tokens, align_down(extent(i)));
            changed = true;
        }
    }
    return {
        .common_tokens = bound_tokens,
        .prefix_closed_tokens = prefix_closed_tokens,
    };
}

}  // namespace

std::vector<std::vector<CacheKey>> KvCacheCoordinator::buildGroupKeys(
    std::span<const std::string> content_hashes) const {
    std::vector<std::vector<CacheKey>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].Id());
    }
    return group_keys;
}

// The one tier matcher: slots below floor_tokens are assumed valid in a lower
// tier. num_base_blocks is the number of base-grain content hashes; group keys,
// probe bounds, and extents use each manager's local block span.
KvCacheCoordinator::PrefixProbe::Tier KvCacheCoordinator::probeTierWithKeys(
    const BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys, std::span<const std::size_t> match_order,
    std::int32_t num_base_blocks, std::int32_t floor_tokens) const {
    PrefixProbe::Tier out;
    out.per_group.resize(groups_.size());
    if (match_order.empty()) {
        return out;
    }
    const ConvergedBoundary boundary = SweepThenConverge(
        match_order, groups_, num_base_blocks * cache_block_tokens_, prefix_alignment_tokens_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_block_tokens = groups_[i].Manager().CacheBlockTokens();
            out.per_group[i] = groups_[i].Manager().Probe(pool, group_keys[i], floor_tokens / group_block_tokens,
                                                          bound_tokens / group_block_tokens);
        },
        [&](std::size_t i) {
            const std::int32_t group_block_tokens = groups_[i].Manager().CacheBlockTokens();
            return (floor_tokens / group_block_tokens + static_cast<std::int32_t>(out.per_group[i].hits.size())) *
                   group_block_tokens;
        });

    // Truncate closed probes to the converged boundary.
    // Non-closed groups were re-probed against the settled bound and are already at or below it.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t group_block_tokens = groups_[i].Manager().CacheBlockTokens();
        const std::int32_t floor_blocks = floor_tokens / group_block_tokens;
        GroupPrefixProbe& probe = out.per_group[i];
        if ((floor_blocks + static_cast<std::int32_t>(probe.hits.size())) * group_block_tokens >
            boundary.common_tokens) {
            _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged boundary");
            probe.hits.resize(static_cast<std::size_t>(boundary.common_tokens / group_block_tokens - floor_blocks));
        }
    }
    out.num_common_tokens = boundary.common_tokens;
    out.prefix_closed_tokens = boundary.prefix_closed_tokens;
    return out;
}

CoordinatorMatch KvCacheCoordinator::acquireTierWithKeys(BlockPool& pool,
                                                         std::span<const std::vector<CacheKey>> group_keys,
                                                         std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                         std::uint64_t access_epoch) {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_blocks = floor_tokens / groups_[i].Manager().CacheBlockTokens();
        out.per_group[i] = groups_[i].Manager().AcquireMatchedBlocks(pool, group_keys[i], floor_blocks,
                                                                     probe.per_group[i], access_epoch);
    }
    return out;
}

KvCacheCoordinator::PrefixProbe KvCacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / cache_block_tokens_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_cache_blocks = static_cast<std::int32_t>(content_hashes.size());
    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    out.device = probeTierWithKeys(pool_, out.group_keys, match_order_, num_cache_blocks, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys(*host_pool_, out.group_keys, match_order_, num_cache_blocks,
                                     /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::PrefixProbe KvCacheCoordinator::ProbeDecodeDestinationPrefix(
    std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / cache_block_tokens_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_cache_blocks = static_cast<std::int32_t>(content_hashes.size());
    std::vector<std::size_t> history_match_order;
    history_match_order.reserve(match_order_.size());
    for (std::size_t group_index : match_order_) {
        if (groups_[group_index].Spec().kind != AttnKind::kMambaState) {
            history_match_order.push_back(group_index);
        }
    }

    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    const auto probe_tier = [&](const BlockPool& pool, std::int32_t floor_tokens) {
        PrefixProbe::Tier tier =
            probeTierWithKeys(pool, out.group_keys, history_match_order, num_cache_blocks, floor_tokens);
        const std::int64_t covered_tokens =
            static_cast<std::int64_t>(tier.num_common_tokens) - static_cast<std::int64_t>(floor_tokens);
        _assert(covered_tokens >= 0, "decode destination state coverage precedes the tier floor");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].Spec().kind == AttnKind::kMambaState) {
                const std::int32_t group_block_tokens = groups_[i].Manager().CacheBlockTokens();
                const std::int64_t num_holes = covered_tokens / group_block_tokens;
                _assert(num_holes <= static_cast<std::int64_t>(out.group_keys[i].size()),
                        "decode destination state hole count is outside the probed range");
                const std::size_t hole_count = static_cast<std::size_t>(num_holes);
                tier.per_group[i].hits.resize(hole_count);
            }
        }
        return tier;
    };
    out.device = probe_tier(pool_, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probe_tier(*host_pool_, /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::AcquiredPrefix KvCacheCoordinator::acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch) {
    AcquiredPrefix out;
    out.device =
        acquireTierWithKeys(pool_, probe.group_keys, /*floor_tokens=*/0, std::move(probe.device), access_epoch);
    if (host_pool_ != nullptr) {
        out.host = acquireTierWithKeys(*host_pool_, probe.group_keys, out.device.num_common_tokens,
                                       std::move(probe.host), access_epoch);
    }
    return out;
}

std::int32_t KvCacheCoordinator::NumAvailableLcmBlocks() const {
    std::int32_t available = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
        const std::optional<GroupId> group_id = pool_.BoundGroup(parent_id);
        if (!group_id || groups_[*group_id].Manager().ParentIsFullyEvictable(pool_, parent_id)) {
            ++available;
        }
    }
    return available;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                         std::uint64_t access_epoch, std::int32_t first_slot,
                                         CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        cacheFullBlocksForGroup(i, tables[i], content_hashes, first_slot, access_epoch, boundary_kind);
    }
}

void KvCacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                 std::span<const std::string> content_hashes, std::int32_t first_slot,
                                                 std::uint64_t access_epoch, CacheBoundaryKind boundary_kind) {
    const std::int32_t fold = groups_[group_index].Manager().CacheBlockTokens() / cache_block_tokens_;
    std::vector<CacheKey> keys = keysForGroup(content_hashes, groups_[group_index].Id(), first_slot);
    const std::int32_t group_first_slot = FirstProjectedGroupSlot(first_slot, fold);
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    groups_[group_index].Manager().CacheFullBlocks(pool_, table, keys, access_epoch, group_first_slot, boundary_kind,
                                                   host_pool_ != nullptr ? &newly_cached : nullptr);
    for (auto& [key, block_ref] : newly_cached) {
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
            .block_ref = std::move(block_ref),
        });
    }
}

void KvCacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                                      std::uint64_t access_epoch) {
    _assert(demand.first_new_page_slot >= 0 &&
                static_cast<std::size_t>(demand.first_new_page_slot) <= demand.page_hashes.size(),
            "first new page slot is outside the hash history");
    if (static_cast<std::size_t>(demand.first_new_page_slot) == demand.page_hashes.size()) {
        return;
    }

    const KvCacheManager& manager = groups_[group_index].Manager();
    const std::int32_t group_block_tokens = manager.CacheBlockTokens();
    const std::int32_t fold = group_block_tokens / cache_block_tokens_;
    if (manager.MatchIsPrefixClosed()) {
        cacheFullBlocksForGroup(group_index, *demand.table,
                                demand.page_hashes.subspan(static_cast<std::size_t>(demand.first_new_page_slot)),
                                demand.first_new_page_slot, access_epoch, demand.boundary_kind);
        return;
    }
    if (demand.num_computed_tokens < 0 || demand.num_computed_tokens % group_block_tokens != 0) {
        return;
    }

    const std::int32_t boundary_slot = demand.num_computed_tokens / group_block_tokens;
    _assert(static_cast<std::int64_t>(boundary_slot) * fold == static_cast<std::int64_t>(demand.page_hashes.size()),
            "non-closed boundary must end at the hash history tail");
    const std::int32_t lookback = std::min(manager.BoundaryLookbackBlocks(), boundary_slot);
    if (lookback == 0) {
        return;
    }
    const std::int32_t first_group_slot = boundary_slot - lookback;
    const std::int32_t first_base_slot = first_group_slot * fold;
    cacheFullBlocksForGroup(group_index, *demand.table,
                            demand.page_hashes.subspan(static_cast<std::size_t>(first_base_slot)), first_base_slot,
                            access_epoch, demand.boundary_kind);
}

void KvCacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ReclaimExpired(pool_, tables[i], num_computed_tokens);
    }
}

void KvCacheCoordinator::ConsumeAvailable(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ConsumeAvailable(tables[i], num_tokens);
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(tables[i]);
    }
}

bool KvCacheCoordinator::ContainsHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Manager().ContainsCachedBlock(*host_pool_, key);
}

bool KvCacheCoordinator::IsHostCachedBlock(CacheBlockLocation location) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    return std::ranges::any_of(
        groups_, [&](const CacheGroup& group) { return group.Manager().ContainsCachedBlock(*host_pool_, location); });
}

std::int32_t KvCacheCoordinator::NumHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Manager().NumCachedBlocks(*host_pool_);
    }
    return count;
}

std::int32_t KvCacheCoordinator::NumPinnedHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Manager().NumPinnedCachedBlocks(*host_pool_);
    }
    return count;
}

void KvCacheCoordinator::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < groups_.size(), "CacheHostBlock group id out of range");
    groups_[key.group_id].Manager().RegisterCachedBlock(*host_pool_, block_ref, key, ++next_access_epoch_);
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, std::int32_t cache_block_tokens, BlockPool& pool,
                                   BlockPool* host_pool) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    _assert(cache_block_tokens > 0, "cache_block_tokens must be > 0");
    _assert(specs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "number of cache groups exceeds int32 range");
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        KvCacheSpec spec = specs[i];
        const GroupId group_id = static_cast<GroupId>(i);
        _assert(spec.block_size >= 0, "group block_size must be >= 0");
        const std::int32_t group_block_tokens = spec.block_size > 0 ? spec.block_size : cache_block_tokens;
        _assert(group_block_tokens % cache_block_tokens == 0,
                "group block_size must be a multiple of cache_block_tokens");
        _assert(spec.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        spec.block_size = group_block_tokens;
        std::unique_ptr<KvCacheManager> manager;
        if (spec.kind == AttnKind::kFull) {
            manager = std::make_unique<FullAttnManager>(group_block_tokens, spec.cache_blocks_per_lcm_block, group_id);
        } else if (spec.kind == AttnKind::kMambaState) {
            manager =
                std::make_unique<MambaStateManager>(group_block_tokens, spec.cache_blocks_per_lcm_block, group_id);
        } else {
            manager = std::make_unique<SwaManager>(group_block_tokens, spec.cache_blocks_per_lcm_block,
                                                   spec.sliding_window, group_id);
        }
        groups.emplace_back(spec, std::move(manager));
    }
    return KvCacheCoordinator{std::move(groups), cache_block_tokens, pool, host_pool};
}

}  // namespace tokenspeed
