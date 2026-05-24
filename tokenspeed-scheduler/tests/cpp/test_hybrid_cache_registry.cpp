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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache_types.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"

namespace tokenspeed::test {

namespace {

constexpr std::int32_t kPageSize = 4;
constexpr std::int32_t kMambaChunkSize = 8;

bool Contains(const std::vector<std::int32_t>& values, std::int32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

PagedCacheGroupConfig MakePagedGroup(std::string group_id, PagedCacheGroupFamily family,
                                     PagedCacheGroupConfig::Retention retention,
                                     std::optional<std::int32_t> sliding_window_tokens = std::nullopt) {
    return PagedCacheGroupConfig{
        .group_id = std::move(group_id),
        .rows_per_page = 4,
        .entry_stride_tokens = 2,
        .total_pages = 16,
        .retention = retention,
        .sliding_window_tokens = sliding_window_tokens,
        .family = family,
    };
}

}  // namespace

TEST(HybridPrefixCacheRegistryTest, KVOnlyRegistryHasSingleCarrierFamily) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaChunkSize};

    const FamilyRegistry& registry = hybrid_prefix_cache.Registry();
    ASSERT_EQ(registry.specs.size(), 1u);

    const CacheResourceSpec* kv = registry.FindById("kv.token_page");
    ASSERT_NE(kv, nullptr);
    EXPECT_EQ(kv->family_index, 0);
    EXPECT_EQ(kv->family, CacheFamily::TokenPage);
    EXPECT_EQ(kv->attachment_kind, TreeAttachmentKind::ReusableTree);
    EXPECT_EQ(kv->recoverability, Recoverability::Exact);
    EXPECT_EQ(kv->publication, PublicationKind::CanonicalPrefixIndex);
    EXPECT_EQ(kv->split_policy, SplitPolicy::CarrierKV);
    EXPECT_EQ(kv->rows_per_page, kPageSize);
    EXPECT_TRUE(kv->required_for_recovery);

    EXPECT_EQ(registry.active_match_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_admit_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_commit_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_evict_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_finish_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_stats_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.active_compatibility_family_indices, std::vector<std::int32_t>{0});
    EXPECT_EQ(registry.FindById("mamba.checkpoint"), nullptr);
}

TEST(HybridPrefixCacheRegistryTest, MambaAdjunctRegistersAlignedCheckpointFamily) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    MambaChunkAllocator mamba_allocator{/*num_slots=*/4};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, &mamba_allocator, kMambaChunkSize};

    const FamilyRegistry& registry = hybrid_prefix_cache.Registry();
    ASSERT_EQ(registry.specs.size(), 2u);

    const CacheResourceSpec* mamba = registry.FindById("mamba.checkpoint");
    ASSERT_NE(mamba, nullptr);
    EXPECT_EQ(mamba->family_index, 1);
    EXPECT_EQ(mamba->family, CacheFamily::RecurrentState);
    EXPECT_EQ(mamba->attachment_kind, TreeAttachmentKind::ReusableTree);
    EXPECT_EQ(mamba->recoverability, Recoverability::AlignedCheckpoint);
    EXPECT_EQ(mamba->publication, PublicationKind::AuxiliaryLocalOnly);
    EXPECT_EQ(mamba->split_policy, SplitPolicy::CheckpointBoundary);
    EXPECT_EQ(mamba->checkpoint_chunk_tokens, kMambaChunkSize);
    EXPECT_EQ(mamba->state_cohort_id, "mamba.checkpoint");
    EXPECT_TRUE(mamba->required_for_recovery);

    EXPECT_TRUE(Contains(registry.active_match_family_indices, mamba->family_index));
    EXPECT_TRUE(Contains(registry.active_admit_family_indices, mamba->family_index));
    EXPECT_TRUE(Contains(registry.active_commit_family_indices, mamba->family_index));
    EXPECT_TRUE(Contains(registry.active_evict_family_indices, mamba->family_index));
}

TEST(HybridPrefixCacheRegistryTest, PagedGroupsWithoutAdjunctAreRequestLocalCompatibilityFamilies) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaChunkSize};
    const std::vector<PagedCacheGroupConfig> groups = {
        MakePagedGroup("v4.history", PagedCacheGroupFamily::History, PagedCacheGroupConfig::Retention::FullHistory),
        MakePagedGroup("v4.swa", PagedCacheGroupFamily::State, PagedCacheGroupConfig::Retention::SlidingWindow,
                       /*sliding_window_tokens=*/16),
    };

    hybrid_prefix_cache.ConfigurePagedCacheAdjunct(std::span<const PagedCacheGroupConfig>{groups}, std::nullopt);

    const FamilyRegistry& registry = hybrid_prefix_cache.Registry();
    const CacheResourceSpec* history = registry.FindById("v4.history");
    const CacheResourceSpec* state = registry.FindById("v4.swa");
    ASSERT_NE(history, nullptr);
    ASSERT_NE(state, nullptr);

    EXPECT_EQ(history->family, CacheFamily::CompressedPage);
    EXPECT_EQ(state->family, CacheFamily::SlidingWindowState);
    EXPECT_EQ(history->attachment_kind, TreeAttachmentKind::NoneForRequestLocal);
    EXPECT_EQ(state->attachment_kind, TreeAttachmentKind::NoneForRequestLocal);
    EXPECT_EQ(history->recoverability, Recoverability::RequestLocalOnly);
    EXPECT_EQ(state->recoverability, Recoverability::RequestLocalOnly);
    EXPECT_FALSE(history->required_for_recovery);
    EXPECT_FALSE(state->required_for_recovery);

    EXPECT_FALSE(Contains(registry.active_match_family_indices, history->family_index));
    EXPECT_FALSE(Contains(registry.active_match_family_indices, state->family_index));
    EXPECT_FALSE(Contains(registry.active_commit_family_indices, history->family_index));
    EXPECT_FALSE(Contains(registry.active_commit_family_indices, state->family_index));
    EXPECT_TRUE(Contains(registry.active_admit_family_indices, history->family_index));
    EXPECT_TRUE(Contains(registry.active_admit_family_indices, state->family_index));
    EXPECT_TRUE(Contains(registry.active_compatibility_family_indices, history->family_index));
    EXPECT_TRUE(Contains(registry.active_compatibility_family_indices, state->family_index));
}

TEST(HybridPrefixCacheRegistryTest, RequiredPagedAdjunctRegistersRecoverableHistoryAndStateCohort) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaChunkSize};
    const std::vector<PagedCacheGroupConfig> groups = {
        MakePagedGroup("v4.history", PagedCacheGroupFamily::History, PagedCacheGroupConfig::Retention::FullHistory),
        MakePagedGroup("v4.swa", PagedCacheGroupFamily::State, PagedCacheGroupConfig::Retention::SlidingWindow,
                       /*sliding_window_tokens=*/16),
    };
    const std::vector<std::string> required = {"v4.history", "v4.swa"};

    hybrid_prefix_cache.ConfigurePagedCacheAdjunct(std::span<const PagedCacheGroupConfig>{groups},
                                                   std::span<const std::string>{required});

    const FamilyRegistry& registry = hybrid_prefix_cache.Registry();
    const CacheResourceSpec* history = registry.FindById("v4.history");
    const CacheResourceSpec* state = registry.FindById("v4.swa");
    ASSERT_NE(history, nullptr);
    ASSERT_NE(state, nullptr);

    EXPECT_EQ(history->attachment_kind, TreeAttachmentKind::ReusableTree);
    EXPECT_EQ(state->attachment_kind, TreeAttachmentKind::ReusableTree);
    EXPECT_EQ(history->recoverability, Recoverability::Exact);
    EXPECT_EQ(state->recoverability, Recoverability::Exact);
    EXPECT_EQ(history->publication, PublicationKind::CanonicalPrefixIndex);
    EXPECT_EQ(state->publication, PublicationKind::AuxiliaryLocalOnly);
    EXPECT_EQ(history->split_policy, SplitPolicy::SnapshotBoundary);
    EXPECT_EQ(state->split_policy, SplitPolicy::SnapshotBoundary);
    EXPECT_EQ(history->state_cohort_id, "paged.required");
    EXPECT_EQ(state->state_cohort_id, "paged.required");
    EXPECT_TRUE(history->required_for_recovery);
    EXPECT_TRUE(state->required_for_recovery);
    EXPECT_EQ(state->sliding_window_tokens, 16);

    EXPECT_TRUE(Contains(registry.active_match_family_indices, history->family_index));
    EXPECT_TRUE(Contains(registry.active_match_family_indices, state->family_index));
    EXPECT_TRUE(Contains(registry.active_commit_family_indices, history->family_index));
    EXPECT_TRUE(Contains(registry.active_commit_family_indices, state->family_index));
    EXPECT_TRUE(Contains(registry.active_evict_family_indices, history->family_index));
    EXPECT_TRUE(Contains(registry.active_evict_family_indices, state->family_index));
}

}  // namespace tokenspeed::test
