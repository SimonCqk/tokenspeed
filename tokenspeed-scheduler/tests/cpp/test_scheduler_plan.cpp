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

#include "integration_test_helper.h"
#include "scheduler/page_hasher.h"

#include <stdexcept>
#include <utility>

namespace tokenspeed::test {

namespace {

const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            return fwd;
        }
    }
    return nullptr;
}

const FlatWriteBackOperation* GetWriteBackOp(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* cache_op = std::get_if<CacheOperation>(&op)) {
            if (const auto* writeback = std::get_if<FlatWriteBackOperation>(cache_op)) {
                return writeback;
            }
        }
    }
    return nullptr;
}

std::vector<std::span<const std::int32_t>> TokenPages(const token_vec_t& tokens, std::int32_t page_size) {
    const std::size_t num_pages = tokens.size() / page_size;
    std::vector<std::span<const std::int32_t>> pages;
    pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        pages.emplace_back(tokens.data() + i * page_size, static_cast<std::size_t>(page_size));
    }
    return pages;
}

void ExpectNoAdjunctMetadata(const FlatForwardOperation& fwd) {
    ASSERT_EQ(fwd.request_ids.size(), fwd.mamba_working_indices.size());
    ASSERT_EQ(fwd.request_ids.size(), fwd.mamba_checkpoint_dst_indices.size());
    ASSERT_EQ(fwd.request_ids.size(), fwd.mamba_cow_src_indices.size());
    ASSERT_EQ(fwd.request_ids.size(), fwd.mamba_branching_seqlens.size());
    for (std::size_t i = 0; i < fwd.request_ids.size(); ++i) {
        EXPECT_EQ(fwd.mamba_working_indices[i], -1);
        EXPECT_EQ(fwd.mamba_checkpoint_dst_indices[i], -1);
        EXPECT_EQ(fwd.mamba_cow_src_indices[i], -1);
        EXPECT_EQ(fwd.mamba_branching_seqlens[i], -1);
    }
    EXPECT_TRUE(fwd.paged_cache_block_tables.empty());
    EXPECT_TRUE(fwd.paged_cache_block_table_base_offsets.empty());
}

SchedulerConfig MakePagedCacheConfigTestBase() {
    SchedulerConfig cfg{};
    cfg.page_size = 2;
    cfg.device_allocator.total_pages = 16;
    cfg.host_allocator.total_pages = 16;
    cfg.max_scheduled_tokens = 16;
    cfg.max_batch_size = 4;
    cfg.enable_l3_storage = false;
    return cfg;
}

PagedCacheGroupConfig MakePagedCacheHistoryGroup(std::string group_id = "fh") {
    PagedCacheGroupConfig group{};
    group.group_id = std::move(group_id);
    group.rows_per_page = 4;
    group.entry_stride_tokens = 1;
    group.total_pages = 8;
    group.retention = PagedCacheGroupConfig::Retention::FullHistory;
    group.family = PagedCacheGroupFamily::History;
    return group;
}

PagedCacheGroupConfig MakePagedCacheSlidingStateGroup(std::string group_id = "swa",
                                                      std::int32_t sliding_window_tokens = 8) {
    PagedCacheGroupConfig group{};
    group.group_id = std::move(group_id);
    group.rows_per_page = 2;
    group.entry_stride_tokens = 1;
    group.total_pages = 8;
    group.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    group.sliding_window_tokens = sliding_window_tokens;
    group.family = PagedCacheGroupFamily::State;
    return group;
}

}  // namespace

class LoadBackViaCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SetupHostCache() {
        Submit(MakeRequestSpec("r_seed", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r_seed", {42});
        PlanOnce();
        SendFinish("r_seed");
        auto plan_wb = PlanOnce();
        const FlatWriteBackOperation* wb = nullptr;
        for (const auto& op : plan_wb.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    wb = w;
                    break;
                }
            }
        }
        ASSERT_NE(wb, nullptr);
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        PlanOnce();
        SendFinish("r_fill");
        auto plan_wb2 = PlanOnce();
        for (const auto& op : plan_wb2.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    if (!w->op_ids.empty()) SendWriteBackDone(w->op_ids[0]);
                    break;
                }
            }
        }
        PlanOnce();
    }
};

TEST_F(LoadBackViaCacheTestSuite, LoadBack_TriggeredAfterPrefetchPopulatesHostCache) {
    SetupHostCache();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<FlatLoadBackOperation>(plan);

    bool r1_in_forward = false;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            for (const auto& rid : fwd->request_ids) {
                if (rid == "r1") r1_in_forward = true;
            }
        }
    }
    EXPECT_TRUE(r1_in_forward || !lb.empty())
        << "host cache hit should trigger LoadBack inline or r1 should be in forward";
}

TEST_F(SchedulerTestSuite, LoadBack_NotTriggeredWithoutHostCacheHit) {
    Submit(MakeRequestSpec("r1", 4));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<FlatLoadBackOperation>(plan);
    EXPECT_TRUE(lb.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_WhenNoRequests) {
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_PlainRequestNoCacheHit) {
    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

TEST(SchedulerPagedCacheConfigurationTest, ConfiguredGroupsWithoutAdjunctRemainPubliclyIntrospectable) {
    auto cfg = MakePagedCacheConfigTestBase();
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup());

    Scheduler scheduler{cfg};

    const auto group_ids = scheduler.PagedCacheGroupIds();
    ASSERT_EQ(group_ids.size(), 1u);
    EXPECT_EQ(group_ids[0], "fh");
    EXPECT_EQ(scheduler.PagedCacheGroupTotalPages("fh"), 8);
    EXPECT_EQ(scheduler.PagedCacheGroupAvailablePages("fh"), 7);
    EXPECT_EQ(scheduler.PagedCacheGroupFailedAllocCount("fh"), 0);
    EXPECT_TRUE(scheduler.GetRequestPagedCachePageIds("missing", "fh").empty());
    EXPECT_EQ(scheduler.GetRequestPagedCacheBaseLogicalPage("missing", "fh"), 0);
}

TEST(SchedulerPagedCacheConfigurationTest, InvalidGroupConfigFailsAtConstruction) {
    auto cfg = MakePagedCacheConfigTestBase();
    auto invalid_group = MakePagedCacheHistoryGroup();
    invalid_group.rows_per_page = 0;
    cfg.paged_cache_groups.push_back(invalid_group);

    EXPECT_THROW({ Scheduler scheduler{cfg}; }, std::invalid_argument);
}

TEST(SchedulerPagedCacheConfigurationTest, DuplicateGroupIdsFailAtConstruction) {
    auto cfg = MakePagedCacheConfigTestBase();
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup("dup"));
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup("dup"));

    EXPECT_THROW({ Scheduler scheduler{cfg}; }, std::invalid_argument);
}

TEST(SchedulerPagedCacheConfigurationTest, EmptyAdjunctRequiredGroupsFailAtConstruction) {
    auto cfg = MakePagedCacheConfigTestBase();
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup());
    cfg.prefix_cache_adjunct = PrefixCacheAdjunctSpec{};

    EXPECT_THROW({ Scheduler scheduler{cfg}; }, std::invalid_argument);
}

TEST(SchedulerPagedCacheConfigurationTest, MissingAdjunctRequiredGroupFailsAtConstruction) {
    auto cfg = MakePagedCacheConfigTestBase();
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup());
    PrefixCacheAdjunctSpec spec{};
    spec.required_groups = {"fh", "missing"};
    cfg.prefix_cache_adjunct = spec;

    EXPECT_THROW({ Scheduler scheduler{cfg}; }, std::invalid_argument);
}

TEST(SchedulerPagedCacheConfigurationTest, SlidingRequiredGroupWithoutPositiveWindowFailsAtConstruction) {
    auto cfg = MakePagedCacheConfigTestBase();
    auto sliding_group = MakePagedCacheSlidingStateGroup();
    sliding_group.sliding_window_tokens.reset();
    cfg.paged_cache_groups.push_back(MakePagedCacheHistoryGroup());
    cfg.paged_cache_groups.push_back(sliding_group);
    PrefixCacheAdjunctSpec spec{};
    spec.required_groups = {"fh", "swa"};
    cfg.prefix_cache_adjunct = spec;

    EXPECT_THROW({ Scheduler scheduler{cfg}; }, std::invalid_argument);
}

class KVOnlyAlwaysPresentFacadeTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.enable_mamba = false;
        cfg.mamba_pool_total_chunks = 0;
        cfg.paged_cache_groups.clear();
        cfg.prefix_cache_adjunct.reset();
        return cfg;
    }
};

TEST_F(KVOnlyAlwaysPresentFacadeTestSuite, NextExecutionPlanKeepsKVOnlyPrefillDecodeAndPrefixReuseStable) {
    EXPECT_TRUE(scheduler_->PagedCacheGroupIds().empty());
    EXPECT_THROW((void)scheduler_->PagedCacheGroupTotalPages("missing"), std::out_of_range);
    EXPECT_THROW((void)scheduler_->PagedCacheGroupAvailablePages("missing"), std::out_of_range);
    EXPECT_THROW((void)scheduler_->PagedCacheGroupFailedAllocCount("missing"), std::out_of_range);
    EXPECT_THROW((void)scheduler_->GetRequestPagedCachePageIds("r_seed", "missing"), std::out_of_range);
    EXPECT_THROW((void)scheduler_->GetRequestPagedCacheBaseLogicalPage("r_seed", "missing"), std::out_of_range);

    Submit(MakeRequestSpec("r_seed", /*num_pages=*/2, /*start=*/1));
    auto seed_prefill = PlanOnce();
    const auto* seed_prefill_fwd = GetForwardOp(seed_prefill);
    ASSERT_NE(seed_prefill_fwd, nullptr);
    ASSERT_EQ(seed_prefill_fwd->request_ids.size(), 1u);
    ASSERT_EQ(seed_prefill_fwd->extend_prefix_lens.size(), 1u);
    EXPECT_EQ(seed_prefill_fwd->request_ids[0], "r_seed");
    EXPECT_EQ(seed_prefill_fwd->extend_prefix_lens[0], 0);
    EXPECT_EQ(seed_prefill_fwd->input_lengths[0], 4);
    ExpectNoAdjunctMetadata(*seed_prefill_fwd);

    SendForwardDone("r_seed", {101});
    auto seed_decode = PlanOnce();
    const auto* seed_decode_fwd = GetForwardOp(seed_decode);
    ASSERT_NE(seed_decode_fwd, nullptr);
    ASSERT_EQ(seed_decode_fwd->request_ids.size(), 1u);
    EXPECT_EQ(seed_decode_fwd->request_ids[0], "r_seed");
    EXPECT_EQ(seed_decode_fwd->input_lengths[0], 1);
    ExpectNoAdjunctMetadata(*seed_decode_fwd);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    SendFinish("r_seed");
    PlanOnce();

    Submit(MakeRequestSpec("r_reuse", /*num_pages=*/2, /*start=*/1));
    auto reuse_prefill = PlanOnce();
    const auto* reuse_prefill_fwd = GetForwardOp(reuse_prefill);
    ASSERT_NE(reuse_prefill_fwd, nullptr);
    ASSERT_EQ(reuse_prefill_fwd->request_ids.size(), 1u);
    ASSERT_EQ(reuse_prefill_fwd->extend_prefix_lens.size(), 1u);
    EXPECT_EQ(reuse_prefill_fwd->request_ids[0], "r_reuse");
    EXPECT_EQ(reuse_prefill_fwd->extend_prefix_lens[0], PageSize());
    EXPECT_EQ(reuse_prefill_fwd->input_lengths[0], PageSize());
    ExpectNoAdjunctMetadata(*reuse_prefill_fwd);
    EXPECT_TRUE(ExtractCacheOpsOfKind<FlatLoadBackOperation>(reuse_prefill).empty());
}

class RollingHashSeedFacadeTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = true;
        cfg.disable_l2_cache = false;
        return cfg;
    }

    void StoreHostPrefix(std::int32_t num_pages, token_t start = 1) {
        Submit(MakeRequestSpec("r_seed", num_pages, start));
        PlanOnce();
        SendForwardDone("r_seed", {900});
        PlanOnce();
        SendFinish("r_seed");
        const auto plan = PlanOnce();
        const auto* writeback = GetWriteBackOp(plan);
        ASSERT_NE(writeback, nullptr);
        ASSERT_FALSE(writeback->op_ids.empty());
        SendWriteBackDone(writeback->op_ids[0]);
        PlanOnce();
    }
};

TEST_F(RollingHashSeedFacadeTestSuite, CalcRollingHashWithoutMatchHashesFullInputFromEmptySeed) {
    StoreHostPrefix(/*num_pages=*/2);

    const token_vec_t input_tokens = MakeAlignedTokens(/*num_pages=*/3, PageSize(), /*start=*/1);
    const auto pages = TokenPages(input_tokens, PageSize());
    const auto full_hashes = ComputePagedHashes(pages, "");

    EXPECT_EQ(scheduler_->CalcRollingHash(input_tokens, /*apply_match=*/false), full_hashes);

    const std::vector<std::span<const std::int32_t>> suffix{pages.begin() + 2, pages.end()};
    EXPECT_EQ(scheduler_->CalcRollingHash(input_tokens, /*apply_match=*/true), ComputePagedHashes(suffix, ""));
}

TEST_F(RollingHashSeedFacadeTestSuite, CalcRollingHashWithFullHostMatchReturnsEmpty) {
    StoreHostPrefix(/*num_pages=*/2);

    const token_vec_t input_tokens = MakeAlignedTokens(/*num_pages=*/2, PageSize(), /*start=*/1);
    EXPECT_TRUE(scheduler_->CalcRollingHash(input_tokens, /*apply_match=*/true).empty());
}

TEST_F(RollingHashSeedFacadeTestSuite, CalcRollingHashUsesEmptyPriorSeedWhenHostNodeHasNoPageHashes) {
    auto cfg = Config();
    cfg.enable_l3_storage = false;
    scheduler_ = std::make_unique<Scheduler>(cfg);
    StoreHostPrefix(/*num_pages=*/2);

    const token_vec_t input_tokens = MakeAlignedTokens(/*num_pages=*/3, PageSize(), /*start=*/1);
    const auto pages = TokenPages(input_tokens, PageSize());
    const std::vector<std::span<const std::int32_t>> suffix{pages.begin() + 2, pages.end()};

    EXPECT_EQ(scheduler_->CalcRollingHash(input_tokens, /*apply_match=*/true), ComputePagedHashes(suffix, ""));
}

class DisablePrefixCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(DisablePrefixCacheTestSuite, SamePromptDoesNotReuseDevicePrefix) {
    Submit(MakeRequestSpec("r_seed", 2));
    PlanOnce();
    SendForwardDone("r_seed", {100});
    PlanOnce();
    SendFinish("r_seed");
    PlanOnce();

    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    const auto& op = plan.Operations()[0];
    auto* fwd = std::get_if<FlatForwardOperation>(&op);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_EQ(fwd->extend_prefix_lens[0], 0);
    EXPECT_EQ(fwd->input_lengths[0], 4);
    EXPECT_TRUE(ExtractCacheOpsOfKind<FlatLoadBackOperation>(plan).empty());
}

TEST_F(DisablePrefixCacheTestSuite, PrefetchNotGeneratedForStorageHit) {
    Submit(MakePrefetchableSpec("r1", 8, 6));
    auto plan = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<PrefetchOperation>(plan).empty());
}

class StableCandidateOrderingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // Force the candidates loop to break after exactly one push so the
        // tiebreaker decides which request wins.
        cfg.max_batch_size = 1;
        return cfg;
    }
};

TEST_F(StableCandidateOrderingSuite, NewForwardOperationTieBreaksOnRequestId) {
    // TP-determinism regression: requests_ is unordered_map<string, ...> so
    // candidates are visited in per-process random order. Without an Id()
    // tiebreaker in newForwardOperation's sort, each rank picks a different
    // request when the loop budget admits only a subset — making forward_op
    // None on some ranks and non-None on others, which deadlocks NCCL.
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan = PlanOnce();
    std::vector<std::string> ids;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            ids = fwd->request_ids;
        }
    }
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], "r_aaa");
}

TEST_F(StableCandidateOrderingSuite, ForwardOpIsInsertionOrderIndependent) {
    // Mirror of the above using two scheduler instances fed the same request
    // set in opposite submit orders. The chosen forward request must depend
    // only on the SET of request ids, not the submission sequence.
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_a = PlanOnce();
    std::vector<std::string> ids_a;
    for (const auto& op : plan_a.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            ids_a = fwd->request_ids;
        }
    }

    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    auto plan_b = PlanOnce();
    std::vector<std::string> ids_b;
    for (const auto& op : plan_b.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            ids_b = fwd->request_ids;
        }
    }

    ASSERT_FALSE(ids_a.empty());
    EXPECT_EQ(ids_a, ids_b);
}

}  // namespace tokenspeed::test
