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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow, kMambaState };

// Why a resumable cache boundary was retained. The declaration order is its
// monotonic promotion order.
enum class CacheBoundaryKind { kChunk, kEndpoint, kPromoted };

using CacheNamespaceId = std::uint32_t;
using ContentHash = std::string;

inline constexpr CacheNamespaceId kDefaultCacheNamespaceId = 0;

struct CacheKey {
    CacheNamespaceId namespace_id{kDefaultCacheNamespaceId};
    GroupId group_id{0};
    ContentHash content_hash{};

    bool operator==(const CacheKey&) const noexcept = default;
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        std::size_t hash = std::hash<CacheNamespaceId>{}(key.namespace_id);
        const std::size_t group_hash = std::hash<GroupId>{}(key.group_id);
        hash ^= group_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        const std::size_t content_hash = std::hash<ContentHash>{}(key.content_hash);
        return hash ^ (content_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U));
    }
};

struct KvCacheSpec {
    AttnKind kind;
    // Logical token span of one block in this group. Zero inherits the
    // coordinator's base hash grain for legacy/equal-grain callers.
    std::int32_t block_size{0};
    // Only kSlidingWindow uses this value. Mamba's one-checkpoint lookback is
    // an internal Manager policy rather than a model window.
    std::int32_t sliding_window;
    // Number of this group's logical cache blocks packed into one physical
    // LCM block. It affects placement only, never prefix-match granularity.
    std::int32_t cache_blocks_per_lcm_block;
};

// Per-request logical-page -> physical-page mapping.
class BlockTable {
public:
    std::span<const CacheBlockRef> Blocks() const noexcept { return blocks_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    // Logical index of the first owned block, or NumBlocks() when the table
    // contains only null holes. Maintained by every table mutation so compact
    // row publication does not rescan the reclaimed prefix.
    std::int32_t FirstLiveBlock() const noexcept { return first_live_block_; }
    std::int32_t AvailableTokens() const { return available_tokens_; }

    CacheBlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()), "EvictToNull index out of range");
        CacheBlockRef old = std::exchange(blocks_[static_cast<std::size_t>(index)], {});
        if (old && index == first_live_block_) {
            advanceFirstLiveBlock();
        }
        return old;
    }

private:
    friend class KvCacheManager;

    void replaceBlocks(std::vector<CacheBlockRef> blocks) {
        blocks_ = std::move(blocks);
        first_live_block_ = 0;
        advanceFirstLiveBlock();
    }

    void appendBlock(CacheBlockRef block) {
        const std::int32_t index = NumBlocks();
        const bool had_live_block = first_live_block_ < index;
        const bool appending_live_block = static_cast<bool>(block);
        blocks_.push_back(std::move(block));
        if (!had_live_block) {
            first_live_block_ = appending_live_block ? index : NumBlocks();
        }
    }

    void appendLiveBlocks(std::vector<CacheBlockRef> blocks) {
        const std::int32_t first_new_block = NumBlocks();
        const bool had_live_block = first_live_block_ < first_new_block;
        blocks_.reserve(blocks_.size() + blocks.size());
        for (CacheBlockRef& block : blocks) {
            blocks_.push_back(std::move(block));
        }
        if (!had_live_block && NumBlocks() > first_new_block) {
            first_live_block_ = first_new_block;
        }
    }

    void resizeWithNullBlocks(std::int32_t size) {
        _assert(size >= NumBlocks(), "BlockTable cannot shrink through resizeWithNullBlocks");
        const bool had_live_block = first_live_block_ < NumBlocks();
        blocks_.resize(static_cast<std::size_t>(size));
        if (!had_live_block) {
            first_live_block_ = NumBlocks();
        }
    }

    void setLiveBlock(std::int32_t index, CacheBlockRef block) {
        _assert(0 <= index && index < NumBlocks(), "setLiveBlock index out of range");
        _assert(static_cast<bool>(block), "setLiveBlock requires an owned block");
        _assert(!blocks_[static_cast<std::size_t>(index)], "setLiveBlock requires a null slot");
        blocks_[static_cast<std::size_t>(index)] = std::move(block);
        first_live_block_ = std::min(first_live_block_, index);
    }

    void clearBlocks() {
        blocks_.clear();
        first_live_block_ = 0;
    }

    void advanceFirstLiveBlock() noexcept {
        while (first_live_block_ < NumBlocks() && !blocks_[static_cast<std::size_t>(first_live_block_)]) {
            ++first_live_block_;
        }
    }

    std::vector<CacheBlockRef> blocks_{};
    std::int32_t first_live_block_{0};
    // Unconsumed capacity at the logical tail. This may span multiple blocks
    // when admission preallocates a later decode/MTP step.
    std::int32_t available_tokens_{0};
};

// Per-group input for one admission. page_hashes is the request's cumulative
// completed base-grain hash history; first_new_page_slot is also measured in
// base-grain slots and splits the newly completed suffix. Managers translate
// it to their group-local logical block span. Non-closed groups select the
// trailing pages required to resume num_computed_tokens. The request owns
// table and the storage behind page_hashes.
struct GroupDemand {
    BlockTable* table{nullptr};
    std::int32_t num_tokens{0};
    std::span<const std::string> page_hashes{};
    std::int32_t first_new_page_slot{0};
    CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
    std::int32_t num_computed_tokens{-1};
    std::int32_t reserve_tokens{0};
    // -1 materializes the ordinary dense suffix. A non-negative input is a
    // base-grain slot; the coordinator translates it to each group's logical
    // slot before admission. Earlier slots remain null holes. Decode-side PD
    // uses this for latest-snapshot state groups.
    std::int32_t materialized_suffix_start{-1};
};

// LCM ownership ids for scheduler accounting/debugging. Kernel-facing page
// tables must instead go through KvCacheManager::BlockTablePageIds().
inline std::vector<std::int32_t> BlockTableLcmBlockIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (const CacheBlockRef& block_ref : table.Blocks()) {
        ids.push_back(block_ref ? block_ref->Location().lcm_block_id : 0);
    }
    return ids;
}

struct PrefixMatch {
    std::vector<CacheBlockRef> blocks{};

    std::int32_t NumHitBlocks() const {
        std::int32_t count = 0;
        for (const CacheBlockRef& block_ref : blocks) {
            count += block_ref ? 1 : 0;
        }
        return count;
    }
};

// Non-owning match shape. A nonzero slot is acquired only after the coordinator
// converges every group to the final common boundary.
struct GroupPrefixProbe {
    std::vector<std::uint8_t> hits{};
};

// Pinned source/destination pages for one asynchronous cache transfer.
struct BlockTransfer {
    GroupId group_id{0};
    CacheBlockRef source;
    CacheBlockRef destination;
};

}  // namespace tokenspeed
