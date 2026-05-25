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

#include <cstdint>
#include <vector>

#include "fsm/forward_events.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/page_allocator.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/request.h"
#include "scheduler/request_cache_context.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {

namespace {

constexpr std::int32_t kPageSize = 2;
constexpr std::int32_t kMambaCacheChunkSize = 4;

Request MakeRequest(const std::string& request_id, std::int32_t num_pages) {
    return Request{RequestSpec{.request_id = request_id, .tokens = MakeAlignedTokens(num_pages, kPageSize)}, kPageSize,
                   Role::kFused};
}

void ApplyFirstChunkToPrefillDone(Request& request, ReqPoolAllocator& req_pool_allocator,
                                  HybridPrefixCache& hybrid_prefix_cache) {
    auto match = hybrid_prefix_cache.MatchPrefix(request.GetFullPagedTokens(/*except_last=*/true)).compat_match;
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        request.PrefillSize(), /*decode_input_tokens=*/0, &req_pool_allocator, match, Role::kFused,
        /*disable_l2_cache=*/false, std::vector<TreeNode*>{}, std::vector<TransferPair>{}, hybrid_prefix_cache});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
}

}  // namespace

TEST(RequestCacheContextTest, ForwardViewExposesPostPrefillFlatteningInputs) {
    PageAllocator device_allocator(kPageSize, /*total_pages=*/16);
    PageAllocator host_allocator(kPageSize, /*total_pages=*/0);
    KVPrefixCache prefix_cache(&device_allocator, &host_allocator);
    MambaChunkAllocator mamba_allocator(/*num_slots=*/4);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_allocator, &mamba_allocator, kMambaCacheChunkSize);
    ReqPoolAllocator req_pool_allocator(/*size=*/2);
    Request request = MakeRequest("r_prefill", /*num_pages=*/2);

    RequestCacheContext pre_apply_context(request);
    EXPECT_EQ(pre_apply_context.OccupiedPageCountSnapshot(), 0);

    auto match = hybrid_prefix_cache.MatchPrefix(request.GetFullPagedTokens(/*except_last=*/true)).compat_match;
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        request.PrefillSize(), /*decode_input_tokens=*/1, &req_pool_allocator, match, Role::kFused,
        /*disable_l2_cache=*/false, std::vector<TreeNode*>{}, std::vector<TransferPair>{}, hybrid_prefix_cache});

    RequestCacheContext context(request);
    std::vector<std::int32_t> occupied_pages = context.OccupiedPagesSnapshot();

    EXPECT_EQ(context.OccupiedPageCountSnapshot(), static_cast<std::int32_t>(occupied_pages.size()));
    EXPECT_EQ(occupied_pages.size(), 3u);
    EXPECT_GT(context.RequestPoolIndex(), 0);
    ASSERT_NE(context.LocalMambaAllocatorView(), nullptr);

    RequestCacheMutation mutation(request);
    EXPECT_NE(mutation.MutableTerminalDeviceNode(), nullptr);

    ForwardOperationBase op{};
    hybrid_prefix_cache.StepCommit({
        .worker_metadata =
            WorkerCompatibilityCommitRequest{
                .op_base = &op,
                .local_mamba_allocator_view = context.LocalMambaAllocatorView(),
            },
    });
    EXPECT_NE(op.mamba_working_idx, -1);
    EXPECT_NE(op.mamba_checkpoint_dst_idx, -1);
}

TEST(RequestCacheContextTest, ForwardViewExposesPostDecodeFromRetractedRecoveryInputs) {
    PageAllocator device_allocator(kPageSize, /*total_pages=*/16);
    PageAllocator host_allocator(kPageSize, /*total_pages=*/0);
    KVPrefixCache prefix_cache(&device_allocator, &host_allocator);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaCacheChunkSize);
    ReqPoolAllocator req_pool_allocator(/*size=*/2);
    Request request = MakeRequest("r_recovered", /*num_pages=*/1);

    auto prefill_match = hybrid_prefix_cache.MatchPrefix(request.GetFullPagedTokens(/*except_last=*/true)).compat_match;
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        request.PrefillSize(), /*decode_input_tokens=*/0, &req_pool_allocator, prefill_match, Role::kFused,
        /*disable_l2_cache=*/false, std::vector<TreeNode*>{}, std::vector<TransferPair>{}, hybrid_prefix_cache});

    auto retract_match = hybrid_prefix_cache.MatchPrefix(request.GetFullPagedTokens(/*except_last=*/true)).compat_match;
    request.Apply(fsm::ScheduleRetractEvent{retract_match, hybrid_prefix_cache});
    request.Apply(fsm::WriteBackDoneEvent{});
    ASSERT_TRUE(request.Is<fsm::Retracted>());

    auto recovery_match =
        hybrid_prefix_cache.MatchPrefix(request.GetFullPagedTokens(/*except_last=*/true), MatchIntent::StateRecovery)
            .compat_match;
    request.Apply(fsm::ScheduleDecodeFromRetractedEvent{/*decode_input_tokens=*/1, &req_pool_allocator, recovery_match,
                                                        std::vector<TreeNode*>{}, std::vector<TransferPair>{},
                                                        hybrid_prefix_cache});
    ASSERT_TRUE(request.Is<fsm::Decoding>());

    RequestCacheContext context(request);
    std::vector<std::int32_t> occupied_pages = context.OccupiedPagesSnapshot();
    DecodeOperation recovered_op{{
        .request_id = request.Id(),
        .request_pool_index = context.RequestPoolIndex(),
        .input_length = 1,
        .occupied_pages = occupied_pages,
        .begin = 0,
        .size = context.OccupiedPageCountSnapshot(),
        .prefill_length = request.PrefillSize(),
    }};

    EXPECT_EQ(context.OccupiedPageCountSnapshot(), static_cast<std::int32_t>(occupied_pages.size()));
    EXPECT_GT(context.RequestPoolIndex(), 0);
    EXPECT_EQ(context.LocalMambaAllocatorView(), nullptr);
    RequestCacheMutation mutation(request);
    EXPECT_NE(mutation.MutableTerminalDeviceNode(), nullptr);
    EXPECT_EQ(recovered_op.begin, 0);
    EXPECT_EQ(recovered_op.size, static_cast<std::int32_t>(recovered_op.occupied_pages.size()));
}

TEST(RequestCacheContextTest, MutationBridgeDelegatesFrontToBackOwnershipTransfer) {
    PageAllocator device_allocator(kPageSize, /*total_pages=*/16);
    PageAllocator host_allocator(kPageSize, /*total_pages=*/0);
    KVPrefixCache prefix_cache(&device_allocator, &host_allocator);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaCacheChunkSize);
    ReqPoolAllocator req_pool_allocator(/*size=*/2);
    Request request = MakeRequest("r_transfer", /*num_pages=*/3);
    ApplyFirstChunkToPrefillDone(request, req_pool_allocator, hybrid_prefix_cache);

    std::vector<std::int32_t> original_local_pages = request.GetLocalAllocatorPages();
    ASSERT_EQ(original_local_pages.size(), 3u);

    RequestCacheMutation mutation(request);
    OwnedPages zero_pages = mutation.TakeFirstLocalKVPages(/*alloc_count=*/0);
    EXPECT_TRUE(zero_pages.Empty());
    EXPECT_EQ(request.GetLocalAllocatorPages(), original_local_pages);

    OwnedPages taken_pages = mutation.TakeFirstLocalKVPages(/*alloc_count=*/2);
    const std::vector<std::int32_t> expected_taken(original_local_pages.begin(), original_local_pages.begin() + 2);
    const std::vector<std::int32_t> expected_remaining(original_local_pages.begin() + 2, original_local_pages.end());

    EXPECT_EQ(taken_pages.Ids(), expected_taken);
    EXPECT_EQ(request.GetLocalAllocatorPages(), expected_remaining);
}

}  // namespace tokenspeed::test
