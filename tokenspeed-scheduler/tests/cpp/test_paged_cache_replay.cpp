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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "integration_test_helper.h"
#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {
namespace {

class PagedCacheReplaySchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig fh{};
        fh.group_id = "fh";
        fh.rows_per_page = 4;
        fh.entry_stride_tokens = 1;
        fh.total_pages = 32;
        fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
        fh.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups.push_back(fh);

        PagedCacheGroupConfig swa{};
        swa.group_id = "swa";
        swa.rows_per_page = 2;
        swa.entry_stride_tokens = 1;
        swa.total_pages = 32;
        swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa.sliding_window_tokens = 8;
        swa.family = PagedCacheGroupFamily::State;
        cfg.paged_cache_groups.push_back(swa);

        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh"};
        spec.replay_window_tokens = 8;
        cfg.prefix_cache_adjunct = spec;
        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

class PagedCacheReplayMixedSchedulerTest : public PagedCacheReplaySchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PagedCacheReplaySchedulerTest::MakeConfig();
        cfg.device_allocator.total_pages = 256;
        cfg.max_scheduled_tokens = 128;
        cfg.enable_mixed_prefill_decode = true;
        for (auto& group : cfg.paged_cache_groups) {
            group.total_pages = 256;
        }
        return cfg;
    }
};

class PagedCacheReplaySeedTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 64;
    static constexpr std::int32_t kDevicePages = 64;
    static constexpr std::int32_t kHistoryRawPerPage = 256;
    static constexpr std::int32_t kSeedTokens = 4;
    static constexpr std::int32_t kSeedWindow = 8;
    static constexpr const char* kHistoryGroup = "history";
    static constexpr const char* kSeedStateGroup = "seed_state";
    static constexpr const char* kIndexerSeedStateGroup = "indexer_seed_state";

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kDevicePages);
        kv_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);

        auto fh_owner = std::make_unique<PagedCacheGroupAllocator>(MakeHistoryGroup(kHistoryGroup));
        auto seed_owner = std::make_unique<PagedCacheGroupAllocator>(MakeSeedGroup(kSeedStateGroup));
        auto indexer_owner = std::make_unique<PagedCacheGroupAllocator>(MakeSeedGroup(kIndexerSeedStateGroup));
        fh_alloc_ = fh_owner.get();
        seed_alloc_ = seed_owner.get();
        indexer_alloc_ = indexer_owner.get();

        hybrid_ = std::make_unique<HybridPrefixCache>(*kv_cache_, /*mamba=*/nullptr,
                                                      /*mamba_chunk_size=*/0);
        hybrid_->RegisterPagedCacheGroup(std::move(fh_owner));
        hybrid_->RegisterPagedCacheGroup(std::move(seed_owner));
        hybrid_->RegisterPagedCacheGroup(std::move(indexer_owner));
        hybrid_->EnablePagedCacheAdjunct({kHistoryGroup, kSeedStateGroup, kIndexerSeedStateGroup},
                                         {{kSeedStateGroup, kSeedWindow}, {kIndexerSeedStateGroup, kSeedWindow}},
                                         /*replay_window_tokens=*/512, kSeedTokens);
        kv_cache_->GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { hybrid_->OnKVEvict(node); });
    }

    TreeNode* InsertDeviceTokens(std::int32_t raw_tokens, token_t token_start = 1) {
        const std::int32_t num_pages = raw_tokens / kPageSize;
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, token_start);
        OwnedPages pages = device_alloc_->Allocate(num_pages);
        auto res = kv_cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages),
                                                           /*page_hashes=*/{}, /*start_node=*/nullptr);
        return res.last_node;
    }

    std::unique_ptr<PagedCacheSnapshot> MakeSnapshot(std::int32_t prefix_len_tokens, bool include_seed = true) {
        auto snap = std::make_unique<PagedCacheSnapshot>();
        snap->prefix_len_tokens = prefix_len_tokens;
        snap->groups.emplace(kHistoryGroup, BuildHistorySnap(prefix_len_tokens));
        if (include_seed) {
            snap->groups.emplace(kSeedStateGroup, BuildSeedSnap(seed_alloc_, prefix_len_tokens));
            snap->groups.emplace(kIndexerSeedStateGroup, BuildSeedSnap(indexer_alloc_, prefix_len_tokens));
        }
        return snap;
    }

    static PagedCacheGroupConfig MakeHistoryGroup(std::string group_id) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = 64;
        cfg.entry_stride_tokens = 4;
        cfg.total_pages = 32;
        cfg.retention = PagedCacheGroupConfig::Retention::FullHistory;
        cfg.family = PagedCacheGroupFamily::History;
        return cfg;
    }

    static PagedCacheGroupConfig MakeSeedGroup(std::string group_id) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = kSeedTokens;
        cfg.entry_stride_tokens = 1;
        cfg.total_pages = 256;
        cfg.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        cfg.sliding_window_tokens = kSeedWindow;
        cfg.family = PagedCacheGroupFamily::State;
        return cfg;
    }

    PagedCacheGroupSnapshot BuildHistorySnap(std::int32_t prefix_len_tokens) {
        PagedCacheGroupTable table{fh_alloc_};
        table.Acquire(kHistoryRawPerPage);
        auto committed = table.CommitHistoryToSnapshot(kHistoryRawPerPage);
        PagedCacheGroupSnapshot snap{};
        snap.pages = std::move(committed.pages);
        snap.base_logical_page = committed.segment_base_logical_page;
        snap.raw_token_cursor = prefix_len_tokens;
        snap.sliding = false;
        return snap;
    }

    static PagedCacheGroupSnapshot BuildSeedSnap(PagedCacheGroupAllocator* alloc, std::int32_t prefix_len_tokens) {
        PagedCacheGroupTable table{alloc};
        table.Acquire(prefix_len_tokens);
        auto committed = table.CheckpointStateSeedToSnapshot(prefix_len_tokens, kSeedTokens);
        PagedCacheGroupSnapshot snap{};
        snap.pages = std::move(committed.pages);
        snap.base_logical_page = committed.segment_base_logical_page;
        snap.raw_token_cursor = prefix_len_tokens;
        snap.sliding = true;
        return snap;
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> kv_cache_;
    PagedCacheGroupAllocator* fh_alloc_{nullptr};
    PagedCacheGroupAllocator* seed_alloc_{nullptr};
    PagedCacheGroupAllocator* indexer_alloc_{nullptr};
    std::unique_ptr<HybridPrefixCache> hybrid_;
};

const PagedCacheGroupConfig& FindGroupConfig(const SchedulerConfig& cfg, const std::string& gid) {
    for (const auto& group : cfg.paged_cache_groups) {
        if (group.group_id == gid) return group;
    }
    throw std::logic_error("test group config missing: " + gid);
}

void ExpectPagedGroupCoversRange(const FlatForwardOperation& fwd, const SchedulerConfig& cfg, const std::string& gid,
                                 std::size_t row, std::int32_t first_token, std::int32_t token_count) {
    const auto& group_cfg = FindGroupConfig(cfg, gid);
    const std::int32_t raw_per_page = group_cfg.RawTokensPerPage();
    ASSERT_GT(raw_per_page, 0);

    auto table_it = fwd.paged_cache_block_tables.find(gid);
    ASSERT_NE(table_it, fwd.paged_cache_block_tables.end()) << "group=" << gid;
    ASSERT_LT(row, table_it->second.size()) << "group=" << gid;
    const auto& pages = table_it->second[row];

    std::int32_t base_logical_page = 0;
    auto base_map_it = fwd.paged_cache_block_table_base_offsets.find(gid);
    if (base_map_it != fwd.paged_cache_block_table_base_offsets.end()) {
        ASSERT_LT(row, base_map_it->second.size()) << "group=" << gid;
        base_logical_page = base_map_it->second[row];
    }

    for (std::int32_t pos = first_token; pos < first_token + token_count; ++pos) {
        const std::int32_t logical_page = pos / raw_per_page;
        const std::int32_t table_page = logical_page - base_logical_page;
        ASSERT_GE(table_page, 0) << "group=" << gid << " row=" << row << " pos=" << pos;
        ASSERT_LT(table_page, static_cast<std::int32_t>(pages.size()))
            << "group=" << gid << " row=" << row << " pos=" << pos;
        const std::int32_t physical_page = pages[static_cast<std::size_t>(table_page)];
        EXPECT_GT(physical_page, 0) << "group=" << gid << " row=" << row << " pos=" << pos;
        EXPECT_LT(physical_page, group_cfg.total_pages) << "group=" << gid << " row=" << row << " pos=" << pos;
    }
}

}  // namespace

TEST_F(PagedCacheReplaySchedulerTest, FirstChunkStartsAtReplayAnchor) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/16, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {99});
    PlanOnce();
    SendFinish("r1");
    PlanOnce();

    Submit(MakeRequestSpec("r2", /*num_pages=*/16, /*start=*/1));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->extend_prefix_lens.size(), 1u);
    ASSERT_EQ(fwd->input_lengths.size(), 1u);
    EXPECT_EQ(fwd->extend_prefix_lens[0], 20);
    EXPECT_EQ(fwd->input_lengths[0], 12);

    auto fh_it = fwd->paged_cache_block_tables.find("fh");
    ASSERT_NE(fh_it, fwd->paged_cache_block_tables.end());
    ASSERT_FALSE(fh_it->second.empty());
    EXPECT_EQ(fh_it->second[0].size(), 8u);

    auto swa_it = fwd->paged_cache_block_tables.find("swa");
    ASSERT_NE(swa_it, fwd->paged_cache_block_tables.end());
    ASSERT_FALSE(swa_it->second.empty());
    EXPECT_EQ(swa_it->second[0].size(), 10u);
}

TEST(PagedCacheReplayAdmissionTest, FreshNonSeedSlidingGroupAdmitsLongReplayFromWindowBase) {
    auto device_alloc = std::make_unique<PageAllocator>(64, 128);
    auto kv_cache = std::make_unique<KVPrefixCache>(device_alloc.get(), /*host_allocator=*/nullptr);

    PagedCacheGroupConfig history{};
    history.group_id = "fh";
    history.rows_per_page = 64;
    history.entry_stride_tokens = 4;
    history.total_pages = 32;
    history.retention = PagedCacheGroupConfig::Retention::FullHistory;
    history.family = PagedCacheGroupFamily::History;

    PagedCacheGroupConfig swa{};
    swa.group_id = "swa";
    swa.rows_per_page = 64;
    swa.entry_stride_tokens = 1;
    swa.total_pages = 5;
    swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa.sliding_window_tokens = 128;
    swa.family = PagedCacheGroupFamily::State;

    auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history);
    auto swa_owner = std::make_unique<PagedCacheGroupAllocator>(swa);
    HybridPrefixCache hybrid(*kv_cache, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
    hybrid.RegisterPagedCacheGroup(std::move(history_owner));
    hybrid.RegisterPagedCacheGroup(std::move(swa_owner));

    MatchResult::PagedCache hit{};
    hit.last_node = kv_cache->GetRadixTree().Root();
    hit.prefix_len_tokens = 3968;
    hit.history_hit_tokens = 4096;
    std::vector<std::int32_t> history_pages(16);
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(history_pages.size()); ++i) {
        history_pages[static_cast<std::size_t>(i)] = i + 1;
    }
    hit.per_group_page_ids.emplace("fh", std::move(history_pages));
    hit.per_group_base_logical_page.emplace("fh", 0);

    auto simulated_free = hybrid.InitialSimulatedFree();
    ASSERT_EQ(simulated_free.at("swa"), 4);
    ASSERT_TRUE(hybrid.AdmitChunk("r", /*first_raw_position_of_op=*/3968,
                                  /*target_raw_tokens_exclusive=*/4096, simulated_free, hit));
    EXPECT_EQ(simulated_free.at("swa"), 0);

    hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/3968,
                             /*target_raw_tokens_exclusive=*/4096, hit);
    EXPECT_EQ(hybrid.GetRequestPagedCacheBaseLogicalPage("r", "swa"), 60);
    EXPECT_EQ(hybrid.GetRequestPagedCachePageIds("r", "swa").size(), 4u);
}

TEST(PagedCacheReplayAdmissionTest, ExistingTransportStateGroupUsesSlidingWindowCredit) {
    auto device_alloc = std::make_unique<PageAllocator>(64, 128);
    auto kv_cache = std::make_unique<KVPrefixCache>(device_alloc.get(), /*host_allocator=*/nullptr);

    PagedCacheGroupConfig history{};
    history.group_id = "fh";
    history.rows_per_page = 8;
    history.entry_stride_tokens = 4;
    history.total_pages = 32;
    history.retention = PagedCacheGroupConfig::Retention::FullHistory;
    history.family = PagedCacheGroupFamily::History;

    PagedCacheGroupConfig swa{};
    swa.group_id = "swa";
    swa.rows_per_page = 4;
    swa.entry_stride_tokens = 1;
    swa.total_pages = 10;
    swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa.sliding_window_tokens = 16;
    swa.family = PagedCacheGroupFamily::State;

    auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history);
    auto swa_owner = std::make_unique<PagedCacheGroupAllocator>(swa);
    HybridPrefixCache hybrid(*kv_cache, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
    hybrid.RegisterPagedCacheGroup(std::move(history_owner));
    hybrid.RegisterPagedCacheGroup(std::move(swa_owner));
    hybrid.EnablePagedCacheAdjunct({"fh"}, {}, /*replay_window_tokens=*/32, /*replay_seed_tokens=*/4);

    hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/0, /*target_raw_tokens_exclusive=*/32);

    auto simulated_free = hybrid.InitialSimulatedFree();
    ASSERT_EQ(simulated_free.at("swa"), 1);
    EXPECT_FALSE(hybrid.AdmitChunk("r", /*first_raw_position_of_op=*/32,
                                   /*target_raw_tokens_exclusive=*/64, simulated_free));
    EXPECT_THROW(hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/32,
                                          /*target_raw_tokens_exclusive=*/64),
                 std::runtime_error);
}

TEST_F(PagedCacheReplayMixedSchedulerTest, MixedPrefillDecodePagedTablesCoverScheduledTokens) {
    std::vector<std::string> decode_ids;
    for (int i = 0; i < 5; ++i) {
        decode_ids.push_back("decode_" + std::to_string(i));
        Submit(MakeRequestSpec(decode_ids.back(), /*num_pages=*/4, static_cast<token_t>(1000 + i * 100)));
    }
    PlanOnce();
    for (const auto& id : decode_ids) {
        SendForwardDone(id, {900});
    }
    PlanOnce();
    for (const auto& id : decode_ids) {
        SendForwardDone(id, {901});
    }

    std::unordered_map<std::string, std::int32_t> decode_first_pos;
    for (const auto& id : decode_ids) {
        decode_first_pos.emplace(id, scheduler_->GetRequestTokenSize(id));
    }

    Submit({
        MakeRequestSpec("prefill_0", /*num_pages=*/16, /*start=*/1),
        MakeRequestSpec("prefill_1", /*num_pages=*/16, /*start=*/100),
        MakeRequestSpec("prefill_2", /*num_pages=*/16, /*start=*/200),
    });

    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 8u);
    ASSERT_EQ(fwd->extend_prefix_lens.size(), 3u);

    for (std::size_t row = 0; row < fwd->request_ids.size(); ++row) {
        std::int32_t first_token = 0;
        if (row < fwd->extend_prefix_lens.size()) {
            first_token = fwd->extend_prefix_lens[row];
        } else {
            auto it = decode_first_pos.find(fwd->request_ids[row]);
            ASSERT_NE(it, decode_first_pos.end()) << "request_id=" << fwd->request_ids[row];
            first_token = it->second;
        }
        ASSERT_LT(row, fwd->input_lengths.size());
        ExpectPagedGroupCoversRange(*fwd, Config(), "fh", row, first_token, fwd->input_lengths[row]);
        ExpectPagedGroupCoversRange(*fwd, Config(), "swa", row, first_token, fwd->input_lengths[row]);
    }
}

TEST_F(PagedCacheReplaySeedTest, ReplayImportsSeedHaloAtAnchorAndCapsHistoryExposure) {
    TreeNode* terminal = InsertDeviceTokens(768);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* n512 = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    TreeNode* n768 = kv_cache_->GetRadixTree().SplitAt(terminal, 768);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n512, nullptr);
    ASSERT_NE(n768, nullptr);

    hybrid_->AttachPagedCacheSnapshotToNode(n256, MakeSnapshot(256));
    hybrid_->AttachPagedCacheSnapshotToNode(n512, MakeSnapshot(512, /*include_seed=*/false));
    hybrid_->AttachPagedCacheSnapshotToNode(n768, MakeSnapshot(768, /*include_seed=*/false));

    auto match = hybrid_->Match(MakeAlignedTokens(768 / kPageSize, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.history_hit_tokens, 768);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
    EXPECT_EQ(match.paged_cache.last_node, n256);

    ASSERT_EQ(match.paged_cache.per_group_page_ids.at(kHistoryGroup).size(), 1u);
    ASSERT_EQ(match.paged_cache.per_group_page_ids.at(kSeedStateGroup).size(), 1u);
    ASSERT_EQ(match.paged_cache.per_group_page_ids.at(kIndexerSeedStateGroup).size(), 1u);
    EXPECT_EQ(match.paged_cache.per_group_base_logical_page.at(kSeedStateGroup), 63);
    EXPECT_EQ(match.paged_cache.per_group_base_logical_page.at(kIndexerSeedStateGroup), 63);

    EXPECT_FALSE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));
    hybrid_->AcquireForRequest("r", match.paged_cache.prefix_len_tokens, 768, match.paged_cache);
    EXPECT_EQ(hybrid_->GetRequestPagedCacheBaseLogicalPage("r", kSeedStateGroup), 63);
    EXPECT_EQ(hybrid_->GetRequestPagedCacheBaseLogicalPage("r", kIndexerSeedStateGroup), 63);

    ASSERT_NO_THROW(hybrid_->CommitChunk("r", n768));
    ASSERT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_EQ(hybrid_->GetRequestPagedCacheBaseLogicalPage("r", kSeedStateGroup), 191);
    EXPECT_EQ(hybrid_->GetRequestPagedCacheBaseLogicalPage("r", kIndexerSeedStateGroup), 191);
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("r", kSeedStateGroup),
              n768->GetPagedCacheSnapshot()->groups.at(kSeedStateGroup).pages.Ids());
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("r", kIndexerSeedStateGroup),
              n768->GetPagedCacheSnapshot()->groups.at(kIndexerSeedStateGroup).pages.Ids());
}

TEST_F(PagedCacheReplaySeedTest, ReplayDoesNotAdvancePastRequestedAnchorForDeeperSeed) {
    TreeNode* terminal = InsertDeviceTokens(768);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* n512 = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    TreeNode* n768 = kv_cache_->GetRadixTree().SplitAt(terminal, 768);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n512, nullptr);
    ASSERT_NE(n768, nullptr);

    hybrid_->AttachPagedCacheSnapshotToNode(n256, MakeSnapshot(256, /*include_seed=*/false));
    hybrid_->AttachPagedCacheSnapshotToNode(n512, MakeSnapshot(512));
    hybrid_->AttachPagedCacheSnapshotToNode(n768, MakeSnapshot(768, /*include_seed=*/false));

    auto match = hybrid_->Match(MakeAlignedTokens(768 / kPageSize, kPageSize, /*start=*/1));
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 0);
    EXPECT_TRUE(match.paged_cache.per_group_page_ids.empty());
}

}  // namespace tokenspeed::test
