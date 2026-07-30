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
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/forward_cache_ops.h"
#include "cache/kv_cache_coordinator.h"
#include "flat_cache_test_access.h"
#include "resource/allocator/paged_cache_group.h"
#include "scheduler/page_hasher.h"
#include "scheduler/types.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasLogicalBlockSize = requires(T value) { value.block_size; };

static_assert(HasLogicalBlockSize<KvCacheSpec>);

KvCacheCoordinator MakeTwoGroup(BlockPool& pool) {
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow, .sliding_window = 4, .cache_blocks_per_lcm_block = 1},
    };
    return MakeCoordinator(specs, 2, pool);
}

TEST(ForwardCacheOpsFree, ReturnsAllPagesToPool) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    ASSERT_LT(pool.NumEmptyLcmBlocks(), free_before);

    FreeRequest(coordinator, tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(AlignFlatPrefillChunkTest, StopsAtPromotionBoundary) {
    EXPECT_EQ(AlignFlatPrefillChunk(/*first_pos=*/16, /*unscheduled=*/24, /*token_budget=*/24,
                                    /*page_size=*/4, /*promotion_boundary_tokens=*/32),
              16);
}

TEST(AlignFlatPrefillChunkTest, KeepsFuturePromotionWhenBudgetFallsShort) {
    EXPECT_EQ(AlignFlatPrefillChunk(/*first_pos=*/16, /*unscheduled=*/24, /*token_budget=*/8,
                                    /*page_size=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(AlignFlatPrefillChunkTest, LaterChunkStopsAtPromotionBoundary) {
    EXPECT_EQ(AlignFlatPrefillChunk(/*first_pos=*/24, /*unscheduled=*/16, /*token_budget=*/16,
                                    /*page_size=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(AlignFlatPrefillChunkTest, EndpointBeforePromotionWins) {
    EXPECT_EQ(AlignFlatPrefillChunk(/*first_pos=*/24, /*unscheduled=*/4, /*token_budget=*/16,
                                    /*page_size=*/4, /*promotion_boundary_tokens=*/32),
              4);
}

TEST(AlignFlatPrefillChunkTest, ReachedPromotionUsesOrdinaryPageAlignment) {
    EXPECT_EQ(AlignFlatPrefillChunk(/*first_pos=*/32, /*unscheduled=*/16, /*token_budget=*/10,
                                    /*page_size=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(ForwardCacheOpsPrefill, FirstChunkAcquiresPagesForTokens) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(ForwardCacheOpsPrefill, FirstChunkClaimsHitThenAcquiresOnlyRemainder) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    // W=16: the SWA bounded match needs ceil((16-1)/2) = 8 > 4 contiguous pages,
    // so all 4 prefix pages stay real hits and nothing slides out of window.
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow, .sliding_window = 16, .cache_blocks_per_lcm_block = 1},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);

    // r1: 8 tokens -> 4 pages/group; freed blocks keep their hashes (prefix-hittable).
    std::vector<std::string> hashes8(4);
    for (std::size_t i = 0; i < hashes8.size(); ++i) {
        hashes8[i] = std::string(64, static_cast<char>('a' + i));
    }
    std::vector<BlockTable> r1(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, r1, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, r1, hashes8);
    const std::vector<std::int32_t> r1_full_ids = BlockTableLcmBlockIds(r1[0]);
    const std::vector<std::int32_t> r1_swa_ids = BlockTableLcmBlockIds(r1[1]);
    FreeRequest(coordinator, r1);

    // r2: same 8-token prefix, 12-token prefill target -> 4 NEW tokens.
    {
        const KvCacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(hashes8);
        ASSERT_EQ(prefix.device.num_common_tokens, 8);
        ASSERT_EQ(std::ranges::count(prefix.device.per_group[1].hits, std::uint8_t{1}), 4)
            << "W=16 must keep every SWA prefix page real";
    }

    // Claimed pages carry no available capacity: the next allocation starts a fresh block.
    {
        std::vector<BlockTable> probe(coordinator.NumGroups());
        ASSERT_TRUE(AdmitForTest(coordinator, probe, coordinator.ProbePrefix(hashes8), GroupDemand{}));
        EXPECT_EQ(probe[0].AvailableTokens(), 0);
        EXPECT_EQ(probe[1].AvailableTokens(), 0);
        EXPECT_EQ(coordinator.GroupManager(0).BlocksNeededFor(probe[0], /*num_tokens=*/4), 2);
        EXPECT_EQ(coordinator.GroupManager(1).BlocksNeededFor(probe[1], /*num_tokens=*/4), 2);
        FreeRequest(coordinator, probe);
    }

    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    std::vector<BlockTable> r2(coordinator.NumGroups());
    KvCacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(hashes8);
    ASSERT_TRUE(AdmitForTest(coordinator, r2, std::move(prefix), GroupDemand{.num_tokens = 4}));

    // Per-group table: 4 claimed prefix pages + ceil(4 new / 2) = 2 fresh = 6.
    ASSERT_EQ(r2[0].NumBlocks(), 6);
    ASSERT_EQ(r2[1].NumBlocks(), 6);

    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(r2[0].Blocks()[i]->Location().lcm_block_id, r1_full_ids[i]) << "full slot " << i;
        EXPECT_EQ(r2[1].Blocks()[i]->Location().lcm_block_id, r1_swa_ids[i]) << "swa slot " << i;
    }

    // Match already pinned the 4 prefix pages/group before free_before; claim
    // only transfers those refs. This operation allocates 2 new pages/group.
    EXPECT_EQ(free_before - pool.NumEmptyLcmBlocks(), 4);
}

TEST(ForwardCacheOpsPrefill, ChunkAcquiresAndCachesFullBlocks) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    // Second chunk: 4 more tokens -> +2 pages/group.
    // num_computed = 4 -> skipped = 4-4+1 = 1 -> 1/2 = 0 pages slid out yet.
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 4,
                                 .page_hashes = hashes2,
                                 .num_computed_tokens = 4,
                             }));
    EXPECT_EQ(tables[0].NumBlocks(), 4);
    EXPECT_EQ(tables[1].NumBlocks(), 4);
    for (const CacheBlockRef& block : tables[1].Blocks()) {
        EXPECT_TRUE(block) << "num_computed=4, W=4: no page is fully out of window yet";
    }
}

// Register-before-punch: CacheFullBlocks skips holes, so punched pages' hashes
// must be registered before the slide.
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowAndKeepsPunchedPageHashes) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // Chunk 0: 8 tokens -> 4 pages/group (chunk >> window is fine: the slide
    // happens on the next admission.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    const std::int32_t free_before_chunk = pool.NumEmptyLcmBlocks();

    // Chunk 1: num_computed = 8 -> skipped = 8-4+1 = 5 -> 5/2 = 2 pages fully
    // out of window: SWA slots 0,1 punched, then 2 fresh pages acquired.
    std::vector<std::string> hashes{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'),
                                    std::string(64, 'd')};
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 4,
                                 .page_hashes = hashes,
                                 .num_computed_tokens = 8,
                             }));

    EXPECT_EQ(tables[0].NumBlocks(), 6);
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t i = 2; i < 6; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }

    // Only the two-page resume tail is cached. The older two SWA pages become
    // free before four fresh pages are acquired, for a net cost of two.
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before_chunk - 2);

    EXPECT_FALSE(
        coordinator.GroupManager(1).ContainsCachedBlock(pool, CacheKey{.group_id = 1, .content_hash = hashes[0]}));
    EXPECT_FALSE(
        coordinator.GroupManager(1).ContainsCachedBlock(pool, CacheKey{.group_id = 1, .content_hash = hashes[1]}));
    EXPECT_TRUE(
        coordinator.GroupManager(1).ContainsCachedBlock(pool, CacheKey{.group_id = 1, .content_hash = hashes[2]}));
    EXPECT_TRUE(
        coordinator.GroupManager(1).ContainsCachedBlock(pool, CacheKey{.group_id = 1, .content_hash = hashes[3]}));
}

// The first decode step (query at position P) only reads keys back to P - W + 1.
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowBeforeAcquire) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 12-token prefill -> 6 pages/group, tails full.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    // num_computed = 12 -> skipped = 12-4+1 = 9 -> 9/2 = 4 pages punched
    // (slots 0..3); reserve 1 token -> 1 fresh page/group.
    std::vector<std::string> hashes(6, "");
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .page_hashes = hashes,
                                 .num_computed_tokens = 12,
                             }));

    ASSERT_EQ(tables[1].NumBlocks(), 7);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(tables[1].Blocks()[i]) << "slot " << i;
    }
    for (std::int32_t i = 4; i < 7; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    // Pool: the uncached SWA prefix frees four parents, then one page per group
    // consumes two parents.
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before + 2);
}

TEST(ForwardCacheOpsDecode, StepAcquiresAndSlidesSwaWindow) {
    BlockPool pool(/*num_lcm_blocks=*/64);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // swa window=4, block_size=2
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));  // 3 pages/group

    for (std::int32_t computed = 7; computed <= 13; ++computed) {
        ASSERT_TRUE(AdmitForTest(coordinator, tables,
                                 GroupDemand{
                                     .num_tokens = 1,
                                     .num_computed_tokens = computed,
                                 }));
    }
    // 13 tokens -> ceil(13/2) = 7 pages.
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    std::int32_t full_nulls = 0;
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        if (!block) ++full_nulls;
    }
    EXPECT_EQ(full_nulls, 0);
    std::int32_t swa_active = 0;
    for (const CacheBlockRef& block : tables[1].Blocks()) {
        if (block) ++swa_active;
    }
    EXPECT_LE(swa_active, 3);
}

TEST(ForwardCacheOpsDecode, DecodeStepRegistersFilledPages) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 8 tokens -> 4 full pages; pages 0-1 registered at prefill time.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    std::vector<std::string> hashes(4);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    CacheFullBlocksForTest(coordinator, tables, std::span<const std::string>(hashes).first(2));
    ASSERT_EQ(MatchPrefixForTest(coordinator, hashes).device.num_common_tokens, 4);

    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .page_hashes = hashes,
                                 .first_new_page_slot = 2,
                                 .num_computed_tokens = 8,
                             }));

    // Registration maps slots to this request's physical pages, not copies.
    const CoordinatorMatch hit = MatchPrefixForTest(coordinator, hashes).device;
    EXPECT_EQ(hit.num_common_tokens, 8);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(hit.per_group[0].blocks[i]->Location().lcm_block_id, tables[0].Blocks()[i]->Location().lcm_block_id)
            << "slot " << i;
    }
}

TEST(ForwardCacheOpsDecode, AdmissionWithEmptyHashesOnlySlidesAndAllocates) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));  // 4 pages/group
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .num_computed_tokens = 8,
                             }));

    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
    EXPECT_EQ(tables[0].NumBlocks(), 5);
    EXPECT_EQ(tables[1].NumBlocks(), 5);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t g = 0; g < coordinator.NumGroups(); ++g) {
        EXPECT_EQ(coordinator.GroupManager(g).NumCachedBlocks(pool), 0);
        EXPECT_EQ(tables[static_cast<std::size_t>(g)].AvailableTokens(), 1);
    }
}

TEST(MakeSpecsFromConfigTest, TranslatesPagedCacheGroups) {
    SchedulerConfig config;
    config.block_size = 16;
    PagedCacheGroupConfig full_grp;
    full_grp.group_id = "full";
    full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupConfig swa_grp;
    swa_grp.group_id = "swa";
    swa_grp.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa_grp.sliding_window_tokens = 128;
    config.paged_cache_groups = {full_grp, swa_grp};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[0].block_size, 16);
    EXPECT_EQ(specs[0].sliding_window, 0);
    EXPECT_EQ(specs[0].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[1].kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].block_size, 16);
    EXPECT_EQ(specs[1].sliding_window, 128);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, PreservesHeterogeneousGroupBlockSpansAboveBaseGrain) {
    SchedulerConfig config;
    config.block_size = 4;
    PagedCacheGroupConfig state;
    state.group_id = "state";
    state.rows_per_page = 8;
    state.entry_stride_tokens = 1;
    state.block_size = 8;
    state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    state.sliding_window_tokens = 16;
    state.family = PagedCacheGroupFamily::State;
    PagedCacheGroupConfig history;
    history.group_id = "history";
    history.rows_per_page = 64;
    history.entry_stride_tokens = 4;
    history.block_size = 256;
    config.paged_cache_groups = {state, history};

    const std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);

    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].block_size, 8);
    EXPECT_EQ(specs[0].kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].block_size, 256);
    EXPECT_EQ(specs[1].kind, AttnKind::kFull);
}

TEST(MakeSpecsFromConfigTest, RejectsGroupSpanOffBaseGridOrRawPageGeometry) {
    SchedulerConfig config;
    config.block_size = 4;
    PagedCacheGroupConfig group;
    group.group_id = "bad";
    group.block_size = 10;
    config.paged_cache_groups = {group};
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);

    group.block_size = 8;
    group.rows_per_page = 4;
    group.entry_stride_tokens = 4;
    config.paged_cache_groups = {group};
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);
}

TEST(MakeSpecsFromConfigTest, StateFamilyMapsToMambaStateKind) {
    SchedulerConfig config;
    config.block_size = 4;
    PagedCacheGroupConfig full_grp;
    full_grp.group_id = "full_attention";
    full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupConfig state_grp;
    state_grp.group_id = "linear_attention";
    state_grp.family = PagedCacheGroupFamily::State;
    config.paged_cache_groups = {full_grp, state_grp};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[1].kind, AttnKind::kMambaState);
    EXPECT_EQ(specs[1].sliding_window, 0);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, Qwen35Fp8UsesOneLogicalPAndPerGroupPacking) {
    SchedulerConfig config;
    config.block_size = 128;
    PagedCacheGroupConfig full;
    full.group_id = "full";
    full.cache_blocks_per_lcm_block = 16;
    PagedCacheGroupConfig state0;
    state0.group_id = "state0";
    state0.family = PagedCacheGroupFamily::State;
    PagedCacheGroupConfig state1 = state0;
    state1.group_id = "state1";
    PagedCacheGroupConfig state2 = state0;
    state2.group_id = "state2";
    config.paged_cache_groups = {full, state0, state1, state2};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);

    ASSERT_EQ(specs.size(), 4u);
    EXPECT_EQ(specs[0].cache_blocks_per_lcm_block, 16);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[2].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[3].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, RejectsNonPositiveGlobalP) {
    SchedulerConfig config;
    config.block_size = 0;
    config.paged_cache_groups.resize(1);
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);

    config.block_size = -1;
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);
}

TEST(MakeSpecsFromConfigTest, RejectsNonPositivePerGroupPacking) {
    SchedulerConfig config;
    config.block_size = 128;
    config.paged_cache_groups.resize(1);
    config.paged_cache_groups[0].cache_blocks_per_lcm_block = 0;
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);

    config.paged_cache_groups[0].cache_blocks_per_lcm_block = -1;
    EXPECT_THROW(MakeSpecsFromConfig(config), std::runtime_error);
}

TEST(MakeSpecsFromConfigTest, RejectsMissingSlidingWindowWithGroupId) {
    SchedulerConfig config;
    config.block_size = 128;
    PagedCacheGroupConfig group;
    group.group_id = "missing_window";
    group.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    config.paged_cache_groups = {group};

    try {
        (void)MakeSpecsFromConfig(config);
        FAIL() << "missing sliding_window_tokens was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find(group.group_id), std::string::npos);
    }
}

TEST(MakeSpecsFromConfigTest, RejectsNonPositiveSlidingWindowWithGroupId) {
    SchedulerConfig config;
    config.block_size = 128;
    PagedCacheGroupConfig group;
    group.group_id = "nonpositive_window";
    group.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    config.paged_cache_groups = {group};

    for (const std::int32_t window : {0, -1}) {
        config.paged_cache_groups[0].sliding_window_tokens = window;
        try {
            (void)MakeSpecsFromConfig(config);
            FAIL() << "sliding_window_tokens=" << window << " was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find(group.group_id), std::string::npos);
        }
    }
}

TEST(MakeSpecsFromConfigTest, PagedCacheGroupConfigRejectsNonPositivePacking) {
    PagedCacheGroupConfig group;
    group.group_id = "full";
    group.rows_per_page = 1;
    group.entry_stride_tokens = 1;
    group.total_pages = 2;

    group.cache_blocks_per_lcm_block = 0;
    EXPECT_THROW(group.Validate(), std::invalid_argument);

    group.cache_blocks_per_lcm_block = -1;
    EXPECT_THROW(group.Validate(), std::invalid_argument);
}

TEST(MakeSpecsFromConfigTest, AcceptsGroupLogicalSpanThatIsAMultipleOfBaseGrain) {
    SchedulerConfig config;
    config.block_size = 128;
    config.paged_cache_groups.resize(1);
    config.paged_cache_groups[0].block_size = 1024;
    const std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].block_size, 1024);
}

TEST(ForwardCacheOpsBuildFlatBlockTableRows, CapabilityControlsHeterogeneousLongHoleCompaction) {
    BlockPool pool(/*num_lcm_blocks=*/256);
    const std::vector<KvCacheSpec> specs{
        {.kind = AttnKind::kFull, .block_size = 2, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        {.kind = AttnKind::kSlidingWindow, .block_size = 4, .sliding_window = 4, .cache_blocks_per_lcm_block = 1},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/260));
    coordinator.ReclaimExpired(tables, /*num_computed_tokens=*/260);
    ASSERT_EQ(tables[0].NumBlocks(), 130);
    ASSERT_EQ(tables[1].NumBlocks(), 65);
    ASSERT_FALSE(tables[1].Blocks()[0]);
    ASSERT_FALSE(tables[1].Blocks()[1]);
    ASSERT_EQ(tables[1].FirstLiveBlock(), 64);

    const std::vector<std::string> group_ids{"full", "swa"};
    struct Case {
        bool compact;
        std::int32_t expected_base;
        std::size_t expected_columns;
    };
    for (const Case test_case : {Case{false, 0, 65}, Case{true, 64, 1}}) {
        const FlatBlockTableRows built = BuildFlatBlockTableRows(coordinator, tables, group_ids, test_case.compact);
        SCOPED_TRACE(test_case.compact ? "capability enabled" : "capability disabled");
        EXPECT_EQ(built.tables.size(), group_ids.size());
        EXPECT_EQ(built.tables.at("full"), coordinator.GroupManager(0).BlockTablePageIds(tables[0]));
        ASSERT_EQ(built.tables.at("swa").size(), test_case.expected_columns);
        if (test_case.compact) {
            EXPECT_EQ(built.base_offsets.size(), built.tables.size());
            EXPECT_EQ(built.base_offsets.at("full"), 0);
            EXPECT_EQ(built.base_offsets.at("swa"), test_case.expected_base);
            EXPECT_TRUE(std::ranges::all_of(built.tables.at("swa"), [](std::int32_t page_id) { return page_id > 0; }));
        } else {
            EXPECT_TRUE(built.base_offsets.empty());
            EXPECT_EQ(built.tables.at("swa"), coordinator.GroupManager(1).BlockTablePageIds(tables[1]));
        }
    }
}

TEST(ForwardCacheOpsBuildFlatBlockTableRows, FreshTablesProduceExactEmptyKeys) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    const std::vector<std::string> group_ids{"alpha", "beta"};
    const FlatBlockTableRows built =
        BuildFlatBlockTableRows(coordinator, tables, group_ids, /*compact_flat_block_tables=*/false);

    EXPECT_EQ(built.tables, (std::map<std::string, std::vector<std::int32_t>>{{"alpha", {}}, {"beta", {}}}));
    EXPECT_TRUE(built.base_offsets.empty());
}

TEST(ForwardCacheOpsBuildFlatBlockTableRows, ResolvesEachGroupsPackingRecipe) {
    BlockPool pool(/*num_lcm_blocks=*/16);
    const std::vector<KvCacheSpec> specs{
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    const std::vector<std::string> group_ids{"packed", "single"};
    const FlatBlockTableRows built =
        BuildFlatBlockTableRows(coordinator, tables, group_ids, /*compact_flat_block_tables=*/false);

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_EQ(built.tables.at("packed"),
              (std::vector<std::int32_t>{
                  coordinator.GroupManager(0).ResolveKernelPageId(tables[0].Blocks()[0]->Location()),
                  coordinator.GroupManager(0).ResolveKernelPageId(tables[0].Blocks()[1]->Location()),
              }));
    EXPECT_EQ(built.tables.at("single"),
              (std::vector<std::int32_t>{
                  coordinator.GroupManager(1).ResolveKernelPageId(tables[1].Blocks()[0]->Location()),
                  coordinator.GroupManager(1).ResolveKernelPageId(tables[1].Blocks()[1]->Location()),
              }));
}

}  // namespace
}  // namespace tokenspeed::test
