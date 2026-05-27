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

// Coverage: end-to-end borrowed-prefix re-import on a fully-cached prefill.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

// page=2, LCM=4 raw tokens (2 KV pages per segment); 12-token prompt spans 3 segments.
class PagedCacheAttachLoopTest : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kLcmRawTokens = 4;

    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 16;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig fh{};
        fh.group_id = "fh";
        fh.rows_per_page = 4;
        fh.entry_stride_tokens = 1;
        fh.total_pages = 32;
        fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
        cfg.paged_cache_groups.push_back(fh);

        PagedCacheGroupConfig swa{};
        swa.group_id = "swa";
        swa.rows_per_page = 2;
        swa.entry_stride_tokens = 1;
        swa.total_pages = 32;
        swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa.sliding_window_tokens = 8;
        cfg.paged_cache_groups.push_back(swa);

        // Enable prefix-cache adjunct (LCM and sliding window derived from groups).
        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh", "swa"};
        cfg.prefix_cache_adjunct = spec;

        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }

    static const FlatWriteBackOperation* GetWriteBackOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cache_op)) return wb;
            }
        }
        return nullptr;
    }

    static const FlatLoadBackOperation* GetLoadBackOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* lb = std::get_if<FlatLoadBackOperation>(cache_op)) return lb;
            }
        }
        return nullptr;
    }
};

class PagedCacheOffloadIntegrationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 16;
        cfg.max_batch_size = 4;
        cfg.disable_l2_cache = true;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig fh{};
        fh.group_id = "fh";
        fh.rows_per_page = 4;
        fh.entry_stride_tokens = 1;
        // PageAllocator keeps page 0 reserved, leaving 4 usable device pages.
        // A 12-token prompt publishes three 4-token snapshots; one extra page
        // lets the decode transition complete before the pressure request
        // forces group offload.
        fh.total_pages = 5;
        fh.host_total_pages = 8;
        fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
        cfg.paged_cache_groups.push_back(fh);

        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh"};
        cfg.prefix_cache_adjunct = spec;
        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }

    static const FlatWriteBackOperation* GetWriteBackOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cache_op)) return wb;
            }
        }
        return nullptr;
    }

    static const FlatLoadBackOperation* GetLoadBackOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* lb = std::get_if<FlatLoadBackOperation>(cache_op)) return lb;
            }
        }
        return nullptr;
    }
};

}  // namespace

// R1 primes 12 tokens; R2's same prefix must skip commit, re-import borrowed
// pages, and populate FlatForwardOperation.paged_cache_block_tables.
TEST_F(PagedCacheAttachLoopTest, FullyCachedPrefillBorrowedPrefixReimported) {
    // R1 primes the cache with 12 tokens.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {99});
    PlanOnce();
    SendFinish("r1");
    PlanOnce();

    // R2 uses the same prefix and should import all 3 LCM segments.
    // borrowed.
    Submit(MakeRequestSpec("r2", /*num_pages=*/6, /*start=*/1));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);

    // (a) prefix-hit covers at least one LCM segment.
    EXPECT_GE(fwd->extend_prefix_lens[0], kLcmRawTokens);

    // (b) per-group tables already contain borrowed pages.
    auto fh_ids = scheduler_->GetRequestPagedCachePageIds("r2", "fh");
    EXPECT_GE(fh_ids.size(), 1u) << "borrowed fh prefix must be imported";

    // (c) paged_cache_block_tables populated for the executor.
    EXPECT_FALSE(fwd->paged_cache_block_tables.empty());
    auto fh_it = fwd->paged_cache_block_tables.find("fh");
    ASSERT_NE(fh_it, fwd->paged_cache_block_tables.end());
    EXPECT_FALSE(fh_it->second.empty());
    EXPECT_FALSE(fh_it->second[0].empty()) << "fh block table row must not be empty for cached prefill";
}

TEST_F(PagedCacheOffloadIntegrationTest, DevicePressureEmitsPagedCacheWriteBackBeforeAdmissionRetry) {
    Submit(MakeRequestSpec("prime", /*num_pages=*/6, /*start=*/1));
    PlanOnce();
    SendForwardDone("prime", {99});
    PlanOnce();
    SendFinish("prime");
    PlanOnce();

    ASSERT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 1);
    ASSERT_EQ(scheduler_->PagedCacheGroupHostAvailablePages("fh"), 7);

    Submit(MakeRequestSpec("pressure", /*num_pages=*/6, /*start=*/1000));
    ExecutionPlan offload_plan = PlanOnce();

    const FlatForwardOperation* offload_forward = GetForwardOp(offload_plan);
    ASSERT_NE(offload_forward, nullptr);
    EXPECT_TRUE(offload_forward->request_ids.empty())
        << "the pressure request must wait until D2H offload ack releases group device pages";

    const FlatWriteBackOperation* writeback = GetWriteBackOp(offload_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);
    auto src_it = writeback->src_pages_by_paged_group.find("fh");
    auto dst_it = writeback->dst_pages_by_paged_group.find("fh");
    ASSERT_NE(src_it, writeback->src_pages_by_paged_group.end());
    ASSERT_NE(dst_it, writeback->dst_pages_by_paged_group.end());
    ASSERT_EQ(src_it->second.size(), 1u);
    ASSERT_EQ(dst_it->second.size(), 1u);
    EXPECT_EQ(src_it->second[0].size(), 2u);
    EXPECT_EQ(dst_it->second[0].size(), 2u);
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 1)
        << "device pages remain owned by snapshots until writeback ack";
    EXPECT_EQ(scheduler_->PagedCacheGroupHostAvailablePages("fh"), 5);

    SendWriteBackDone(writeback->op_ids[0]);
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 3);
    EXPECT_EQ(scheduler_->PagedCacheGroupHostAvailablePages("fh"), 5);

    ExecutionPlan retry_plan = PlanOnce();
    const FlatForwardOperation* retry_forward = GetForwardOp(retry_plan);
    ASSERT_NE(retry_forward, nullptr);
    EXPECT_NE(std::find(retry_forward->request_ids.begin(), retry_forward->request_ids.end(), "pressure"),
              retry_forward->request_ids.end());
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 0);
}

TEST_F(PagedCacheOffloadIntegrationTest, DevicePressureDoesNotOffloadRequestLocalPages) {
    Submit(MakeRequestSpec("active", /*num_pages=*/9, /*start=*/1));
    ExecutionPlan active_plan = PlanOnce();
    const FlatForwardOperation* active_forward = GetForwardOp(active_plan);
    ASSERT_NE(active_forward, nullptr);
    EXPECT_NE(std::find(active_forward->request_ids.begin(), active_forward->request_ids.end(), "active"),
              active_forward->request_ids.end());
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 0);
    EXPECT_EQ(scheduler_->PagedCacheGroupHostAvailablePages("fh"), 7);
    EXPECT_EQ(scheduler_->GetRequestPagedCachePageIds("active", "fh").size(), 4u);

    Submit(MakeRequestSpec("pressure", /*num_pages=*/8, /*start=*/1000));
    ExecutionPlan pressure_plan = PlanOnce();
    const FlatWriteBackOperation* writeback = GetWriteBackOp(pressure_plan);
    EXPECT_EQ(writeback, nullptr) << "request-local owned paged-cache pages are not published snapshots yet";

    const FlatForwardOperation* pressure_forward = GetForwardOp(pressure_plan);
    ASSERT_NE(pressure_forward, nullptr);
    EXPECT_TRUE(pressure_forward->request_ids.empty())
        << "both requests must wait instead of offloading uncommitted request-local pages";
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 0);
    EXPECT_EQ(scheduler_->PagedCacheGroupHostAvailablePages("fh"), 7);
    EXPECT_EQ(scheduler_->PagedCacheGroupHostWriteBackPagesScheduledTotal("fh"), 0);
}

TEST_F(PagedCacheOffloadIntegrationTest, HostPagedCacheHitEmitsLoadBackBeforeForwardReuse) {
    Submit(MakeRequestSpec("prime", /*num_pages=*/6, /*start=*/1));
    PlanOnce();
    SendForwardDone("prime", {99});
    PlanOnce();
    SendFinish("prime");
    PlanOnce();

    Submit(MakeRequestSpec("zz_pressure", /*num_pages=*/8, /*start=*/1000));
    ExecutionPlan offload_plan = PlanOnce();
    const FlatWriteBackOperation* writeback = GetWriteBackOp(offload_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);
    SendWriteBackDone(writeback->op_ids[0]);
    SendAbort("zz_pressure");
    PlanOnce();

    Submit(MakeRequestSpec("aa_reuse", /*num_pages=*/6, /*start=*/1));
    ExecutionPlan reuse_plan = PlanOnce();

    const FlatForwardOperation* forward = GetForwardOp(reuse_plan);
    ASSERT_NE(forward, nullptr);
    auto req_it = std::find(forward->request_ids.begin(), forward->request_ids.end(), "aa_reuse");
    ASSERT_NE(req_it, forward->request_ids.end());
    const auto row = static_cast<std::size_t>(std::distance(forward->request_ids.begin(), req_it));
    ASSERT_LT(row, forward->paged_cache_block_tables.at("fh").size());

    const FlatLoadBackOperation* loadback = GetLoadBackOp(reuse_plan);
    ASSERT_NE(loadback, nullptr);
    ASSERT_EQ(loadback->op_ids.size(), 1u);
    auto src_it = loadback->src_pages_by_paged_group.find("fh");
    auto dst_it = loadback->dst_pages_by_paged_group.find("fh");
    ASSERT_NE(src_it, loadback->src_pages_by_paged_group.end());
    ASSERT_NE(dst_it, loadback->dst_pages_by_paged_group.end());
    ASSERT_EQ(src_it->second.size(), 1u);
    ASSERT_EQ(dst_it->second.size(), 1u);
    EXPECT_EQ(src_it->second[0].size(), 2u);
    EXPECT_EQ(dst_it->second[0].size(), 2u);

    const std::vector<std::int32_t> request_pages = scheduler_->GetRequestPagedCachePageIds("aa_reuse", "fh");
    EXPECT_EQ(forward->paged_cache_block_tables.at("fh")[row], request_pages);
    for (std::int32_t loaded_device_page : dst_it->second[0]) {
        EXPECT_NE(std::find(request_pages.begin(), request_pages.end(), loaded_device_page), request_pages.end())
            << "forward metadata must use the H2D destination device pages for the loaded prefix";
    }

    SendPrefetchDone(loadback->op_ids[0]);
    SendForwardDone("aa_reuse", {100});
    ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode_forward = GetForwardOp(decode_plan);
    ASSERT_NE(decode_forward, nullptr);
    EXPECT_NE(std::find(decode_forward->request_ids.begin(), decode_forward->request_ids.end(), "aa_reuse"),
              decode_forward->request_ids.end())
        << "loadback completion is an op-level ack and must not disturb the request FSM";
}

TEST_F(PagedCacheOffloadIntegrationTest, HostPagedCacheLoadBackFailureRollsBackDeviceResidency) {
    Submit(MakeRequestSpec("prime", /*num_pages=*/6, /*start=*/1));
    PlanOnce();
    SendForwardDone("prime", {99});
    PlanOnce();
    SendFinish("prime");
    PlanOnce();

    Submit(MakeRequestSpec("zz_pressure", /*num_pages=*/8, /*start=*/1000));
    ExecutionPlan offload_plan = PlanOnce();
    const FlatWriteBackOperation* writeback = GetWriteBackOp(offload_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);
    SendWriteBackDone(writeback->op_ids[0]);
    SendAbort("zz_pressure");
    PlanOnce();

    Submit(MakeRequestSpec("aa_reuse", /*num_pages=*/6, /*start=*/1));
    ExecutionPlan reuse_plan = PlanOnce();
    const FlatForwardOperation* forward = GetForwardOp(reuse_plan);
    ASSERT_NE(forward, nullptr);
    EXPECT_NE(std::find(forward->request_ids.begin(), forward->request_ids.end(), "aa_reuse"),
              forward->request_ids.end());

    const FlatLoadBackOperation* loadback = GetLoadBackOp(reuse_plan);
    ASSERT_NE(loadback, nullptr);
    ASSERT_EQ(loadback->op_ids.size(), 1u);
    const std::int32_t available_after_materialize = scheduler_->PagedCacheGroupAvailablePages("fh");

    SendPrefetchDone(loadback->op_ids[0], /*success=*/false);
    EXPECT_GT(scheduler_->PagedCacheGroupAvailablePages("fh"), available_after_materialize)
        << "failed H2D loadback must release the uninitialized destination device pages";
    EXPECT_EQ(scheduler_->PagedCacheGroupDeviceLoadBackFailedCount("fh"), 1);

    PlanOnce();
    EXPECT_TRUE(scheduler_->GetRequestPagedCachePageIds("aa_reuse", "fh").empty())
        << "the request depending on the failed loadback must be aborted and cleaned up";

    Submit(MakeRequestSpec("bb_reuse", /*num_pages=*/6, /*start=*/1));
    ExecutionPlan retry_plan = PlanOnce();
    const FlatLoadBackOperation* retry_loadback = GetLoadBackOp(retry_plan);
    ASSERT_NE(retry_loadback, nullptr)
        << "failed loadback rollback must leave the snapshot host-only, so a later reuse schedules H2D again";
}

}  // namespace tokenspeed::test
