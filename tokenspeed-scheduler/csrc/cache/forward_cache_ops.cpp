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

#include "cache/forward_cache_ops.h"

#include <stdexcept>

#include "resource/allocator/paged_cache_group.h"
#include "scheduler/types.h"

namespace tokenspeed {

std::int32_t AlignFlatPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                                   std::int32_t page_size, std::int32_t promotion_boundary_tokens) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(page_size > 0, "page_size must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size == unscheduled) {
        return chunk_size;
    }

    const std::int32_t page_offset = first_pos % page_size;
    if (page_offset != 0) {
        const std::int32_t tokens_to_boundary = page_size - page_offset;
        return token_budget >= tokens_to_boundary ? tokens_to_boundary : 0;
    }
    return chunk_size - chunk_size % page_size;
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    _assert(config.block_size > 0, "cache_block_tokens must be > 0");
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        _assert(group.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        _assert(group.block_size >= 0, "group block_size must be >= 0");
        const std::int32_t block_size = group.block_size > 0 ? group.block_size : config.block_size;
        _assert(block_size % config.block_size == 0, "group block_size must be a multiple of the base hash grain");
        // Legacy/Radix configs may leave block_size unset while carrying a
        // physical paged-cache geometry unrelated to the Flat hash grain.
        // Cross-check only an explicit group-local logical span.
        if (group.block_size > 0 && group.rows_per_page > 0 && group.entry_stride_tokens > 0) {
            const std::int64_t raw_tokens_per_page =
                static_cast<std::int64_t>(group.rows_per_page) * group.entry_stride_tokens;
            _assert(raw_tokens_per_page == block_size,
                    "group block_size must equal rows_per_page * entry_stride_tokens");
        }
        // family=State marks trailing-window prefix reuse and covers both SWA and
        // linear-attention groups; only a State group WITHOUT SlidingWindow
        // retention is a mamba-style state group.
        const bool final_state_manager = group.family == PagedCacheGroupFamily::State &&
                                         group.retention != PagedCacheGroupConfig::Retention::SlidingWindow;
        if (config.enable_flatkv_pd) {
            const PagedCacheTransferPolicy expected =
                final_state_manager ? PagedCacheTransferPolicy::LatestSnapshot : PagedCacheTransferPolicy::FullSuffix;
            if (group.transfer_policy == PagedCacheTransferPolicy::Unspecified) {
                throw std::invalid_argument("FlatKV PD cache group '" + group.group_id +
                                            "' requires an explicit transfer_policy");
            }
            if (group.transfer_policy != expected) {
                throw std::invalid_argument("FlatKV PD cache group '" + group.group_id +
                                            "' transfer_policy does not match its scheduler "
                                            "destination layout");
            }
        }
        if (final_state_manager) {
            specs.push_back(KvCacheSpec{
                .kind = AttnKind::kMambaState,
                .block_size = block_size,
                .sliding_window = 0,
                .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
            });
            continue;
        }
        const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
        if (is_swa && (!group.sliding_window_tokens || *group.sliding_window_tokens <= 0)) {
            throw std::invalid_argument("Flat cache group '" + group.group_id +
                                        "' requires positive sliding_window_tokens");
        }
        specs.push_back(KvCacheSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .block_size = block_size,
            .sliding_window = is_swa ? *group.sliding_window_tokens : 0,
            .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
        });
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

FlatBlockTableRows BuildFlatBlockTableRows(const KvCacheCoordinator& coordinator, const std::vector<BlockTable>& tables,
                                           std::span<const std::string> group_ids, bool compact_flat_block_tables) {
    _assert(tables.size() == group_ids.size(), "BuildFlatBlockTableRows: tables/group_ids size mismatch");
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()),
            "BuildFlatBlockTableRows: tables/coordinator size mismatch");

    FlatBlockTableRows out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        const KvCacheManager& manager = coordinator.GroupManager(static_cast<std::int32_t>(i));
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        const std::size_t base = compact_flat_block_tables ? static_cast<std::size_t>(tables[i].FirstLiveBlock()) : 0;
        _assert(base <= blocks.size(), "BlockTable first-live offset exceeds its logical size");
        std::vector<std::int32_t> page_ids;
        page_ids.reserve(blocks.size() - base);
        for (const CacheBlockRef& block_ref : blocks.subspan(base)) {
            page_ids.push_back(block_ref ? manager.ResolveKernelPageId(block_ref->Location()) : 0);
        }
        out.tables.emplace(group_ids[i], std::move(page_ids));
        if (compact_flat_block_tables) {
            out.base_offsets.emplace(group_ids[i], static_cast<std::int32_t>(base));
        }
    }
    _assert(out.tables.size() == tables.size(), "Flat block tables require the exact configured group key set");
    _assert(!compact_flat_block_tables || out.base_offsets.size() == out.tables.size(),
            "Compact Flat block tables and base offsets require the same group key set");
    return out;
}

}  // namespace tokenspeed
