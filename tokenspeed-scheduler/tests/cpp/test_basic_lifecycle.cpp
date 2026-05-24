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

#include <cstddef>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class BasicLifecycleTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_l3_storage = false;
        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

TEST_F(BasicLifecycleTestSuite, Submit_CreatesWaitingRequest) {
    Submit(MakeRequestSpec("r1", 2));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(BasicLifecycleTestSuite, PlanOnce_SchedulesPrefill) {
    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_FALSE(fwd->request_ids.empty());
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_GT(fwd->input_lengths[0], 0);
}

TEST_F(BasicLifecycleTestSuite, ForwardDone_TransitionsToPrefillDone) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {42});
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);
}

TEST_F(BasicLifecycleTestSuite, SecondPlan_TransitionsToDecoding) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

TEST_F(BasicLifecycleTestSuite, FullLifecycle_SubmitThroughFinish) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    SendFinish("r1");
    auto plan = PlanOnce();
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(BasicLifecycleTestSuite, EmptyPlan_NoRequests) {
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    EXPECT_TRUE(fwd->request_ids.empty());
}

TEST_F(BasicLifecycleTestSuite, MultipleSubmits_AllGetScheduled) {
    Submit(MakeRequestSpec("r1", 2, 1));
    Submit(MakeRequestSpec("r2", 2, 50));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    EXPECT_EQ(fwd->request_ids.size(), 2u);
}

TEST_F(BasicLifecycleTestSuite, TokenSizeGrowsWithDecodeOutput) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 5);

    SendForwardDone("r1", {43});
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 6);
}

TEST_F(BasicLifecycleTestSuite, GetRequestTokenSize_UnknownRequest) {
    EXPECT_EQ(scheduler_->GetRequestTokenSize("nonexistent"), -1);
}

TEST_F(BasicLifecycleTestSuite, AvailableKvPages_DecreasesAfterPrefill) {
    auto before = scheduler_->AvailableKvPages();
    EXPECT_EQ(before, static_cast<std::size_t>(Config().device_allocator.total_pages - 1));
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    auto after = scheduler_->AvailableKvPages();
    EXPECT_LT(after, before);
    EXPECT_EQ(before - after, scheduler_->ActiveKvPages());
}

TEST_F(BasicLifecycleTestSuite, ActiveKvPagesCountsOnlyForwardStates) {
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);

    Submit(MakeRequestSpec("r1", 1));
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u) << "submitted requests must not contribute";

    PlanOnce();
    EXPECT_EQ(scheduler_->ActiveKvPages(), 2u) << "active forward state contributes occupied prefix/local pages";

    SendForwardDone("r1", {42});
    EXPECT_EQ(scheduler_->ActiveKvPages(), 2u);

    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 2u) << "Decoding remains active";

    SendFinish("r1");
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u) << "Draining/finished requests must not contribute";

    const auto writeback_plan = PlanOnce();
    EXPECT_FALSE(ExtractCacheOpsOfKind<FlatWriteBackOperation>(writeback_plan).empty());
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u) << "WritingBack requests must not contribute";
}

TEST_F(BasicLifecycleTestSuite, ActiveKvPagesDeduplicatesSharedPrefixPages) {
    Submit(MakeRequestSpec("r_seed", 2, 1));
    PlanOnce();
    SendForwardDone("r_seed", {101});
    PlanOnce();
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);
    ASSERT_EQ(scheduler_->ActiveKvPages(), 3u);

    Submit(MakeRequestSpec("r_reuse", 2, 1));
    EXPECT_EQ(scheduler_->ActiveKvPages(), 3u) << "submitted requests must not add pages";

    auto reuse_plan = PlanOnce();
    auto* reuse_fwd = GetForwardOp(reuse_plan);
    ASSERT_NE(reuse_fwd, nullptr);
    ASSERT_EQ(reuse_fwd->request_ids.size(), 1u);
    EXPECT_EQ(reuse_fwd->request_ids[0], "r_reuse");
    ASSERT_EQ(reuse_fwd->extend_prefix_lens.size(), 1u);
    EXPECT_EQ(reuse_fwd->extend_prefix_lens[0], PageSize());

    // r_seed and r_reuse both observe the same prefix-cache page for the same
    // prompt. ActiveKvPages counts that shared page id once; summing per-request
    // occupied-page snapshots would report 6 pages instead of the 5 unique page
    // ids visible through the public statistic.
    EXPECT_EQ(scheduler_->ActiveKvPages(), 5u);
}

}  // namespace tokenspeed::test
