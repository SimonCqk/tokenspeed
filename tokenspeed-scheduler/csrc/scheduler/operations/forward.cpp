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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/request.h"
#include "scheduler/request_cache_context.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

namespace tokenspeed {

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free, std::vector<WriteBackOperation>& writeback_ops) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    RecoveryPlan recovery_plan = hybrid_prefix_cache_.MatchPrefix(request->GetFullPagedTokens(true));
    MatchResult match_result = recovery_plan.compat_match;
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    if (disable_l2_cache) {
        unscheduled = request->PrefillSize() - device_matched * config_.page_size;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.page_size * (host_matched - device_matched);
        }
        unscheduled = request->PrefillSize() - std::max(device_matched, host_matched) * config_.page_size;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    const std::string request_id = request->Id();
    PagedCacheDeviceLoadBackResult paged_loadback =
        hybrid_prefix_cache_.PreparePagedCacheDeviceLoadBack(match_result, simulated_free);
    if (!paged_loadback.writeback_transfers.empty()) {
        if (auto op = newPagedCacheWriteBackOperation(std::move(paged_loadback.writeback_transfers),
                                                      std::move(paged_loadback.writeback_nodes_by_group))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!paged_loadback.ok) {
        return {};
    }
    AdmissionVerdict admission = hybrid_prefix_cache_.Apply(
        cache::admit::PrefillFirstChunk{
            .request_id = request_id,
            .match_result = match_result,
            .device_pages_needed = device_pages_needed,
            .tokens_this_round = tokens_this_round,
            .first_raw_position_of_op = first_pos,
            .target_raw_tokens_exclusive = target,
        },
        simulated_free);
    if (!admission.paged_cache_writeback_transfers.empty()) {
        hybrid_prefix_cache_.CancelPagedCacheDeviceLoadBack(paged_loadback.loadback_nodes_by_group, simulated_free);
        if (auto op = newPagedCacheWriteBackOperation(std::move(admission))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!admission.admitted) {
        hybrid_prefix_cache_.CancelPagedCacheDeviceLoadBack(paged_loadback.loadback_nodes_by_group, simulated_free);
        return {};
    }
    if (admission.mamba_branching_seqlen.has_value()) {
        match_result.mamba_branching_seqlen = *admission.mamba_branching_seqlen;
    }
    if (admission.mamba_cow_src_index.has_value()) {
        match_result.mamba_cow_src_index = *admission.mamba_cow_src_index;
    }

    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &req_pool_allocator_,
        match_result,
        config_.role,
        disable_l2_cache,
        std::move(loadback_diff),
        std::move(admission.cache_transfer_pairs),
        std::move(paged_loadback.loadback_transfers),
        std::move(paged_loadback.loadback_nodes_by_group),
        std::move(paged_loadback.loadback_snapshot_refs),
        hybrid_prefix_cache_,
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free, std::vector<WriteBackOperation>& writeback_ops) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

    std::int32_t pages_needed = (tokens_this_round + config_.page_size - 1) / config_.page_size;

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    const std::string request_id = request->Id();
    AdmissionVerdict admission = hybrid_prefix_cache_.Apply(
        cache::admit::PrefillChunk{
            .request_id = request_id,
            .device_pages_needed = pages_needed,
            .first_raw_position_of_op = first_pos,
            .target_raw_tokens_exclusive = target,
        },
        simulated_free);
    if (!admission.paged_cache_writeback_transfers.empty()) {
        if (auto op = newPagedCacheWriteBackOperation(std::move(admission))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!admission.admitted) {
        return {};
    }

    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_};
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free,
                                                                  std::vector<WriteBackOperation>& writeback_ops) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.page_size - 1) / config_.page_size;

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target = first_pos + config_.decode_input_tokens;
    const std::string request_id = request->Id();
    AdmissionVerdict admission = hybrid_prefix_cache_.Apply(
        cache::admit::Decode{
            .request_id = request_id,
            .device_pages_needed = pages_needed,
            .first_raw_position_of_op = first_pos,
            .target_raw_tokens_exclusive = target,
            .refresh_mamba_checkpoint = request->Is<fsm::PrefillDone>(),
        },
        simulated_free);
    if (!admission.paged_cache_writeback_transfers.empty()) {
        if (auto op = newPagedCacheWriteBackOperation(std::move(admission))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!admission.admitted) {
        return {};
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens, hybrid_prefix_cache_};
}

std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free,
    std::vector<WriteBackOperation>& writeback_ops) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    RecoveryPlan recovery_plan =
        hybrid_prefix_cache_.MatchPrefix(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    MatchResult match_result = recovery_plan.compat_match;
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();
    if (!recovery_plan.recovery_state_available) {
        spdlog::warn("[Scheduler] Retracted request {} lost required cache recovery state, aborting request",
                     request->Id());
        request->Apply(fsm::AbortEvent{});
        return {};
    }
    TreeNode* mamba_recovery_node = recovery_plan.protected_recovery_node;

    const std::int32_t device_matched2 = match_result.device.DepthInPage();
    const std::int32_t host_matched2 = match_result.host.DepthInPage();
    // Pages needed: LoadBack nodes (host→device) + pages for decode step itself.
    std::int32_t num_tokens = 0;
    if (host_matched2 > device_matched2) {
        num_tokens += (config_.page_size * (host_matched2 - device_matched2)) + config_.decode_input_tokens;
    } else {
        num_tokens += config_.decode_input_tokens;
    }
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    const std::int32_t target = request->TokenSize();
    const std::string request_id = request->Id();
    PagedCacheDeviceLoadBackResult paged_loadback =
        hybrid_prefix_cache_.PreparePagedCacheDeviceLoadBack(match_result, simulated_free);
    if (!paged_loadback.writeback_transfers.empty()) {
        if (auto op = newPagedCacheWriteBackOperation(std::move(paged_loadback.writeback_transfers),
                                                      std::move(paged_loadback.writeback_nodes_by_group))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!paged_loadback.ok) {
        return {};
    }
    AdmissionVerdict admission = hybrid_prefix_cache_.Apply(
        cache::admit::DecodeFromRetracted{
            .request_id = request_id,
            .match_result = match_result,
            .device_pages_needed = device_pages_needed,
            .target_raw_tokens_exclusive = target,
            .protected_recovery_node = mamba_recovery_node,
        },
        simulated_free);
    if (!admission.paged_cache_writeback_transfers.empty()) {
        hybrid_prefix_cache_.CancelPagedCacheDeviceLoadBack(paged_loadback.loadback_nodes_by_group, simulated_free);
        if (auto op = newPagedCacheWriteBackOperation(std::move(admission))) {
            writeback_ops.push_back(std::move(*op));
        }
        return {};
    }
    if (!admission.admitted) {
        hybrid_prefix_cache_.CancelPagedCacheDeviceLoadBack(paged_loadback.loadback_nodes_by_group, simulated_free);
        return {};
    }
    if (admission.mamba_cow_src_index.has_value()) {
        match_result.mamba_cow_src_index = *admission.mamba_cow_src_index;
    }

    return fsm::ScheduleDecodeFromRetractedEvent{config_.decode_input_tokens,
                                                 &req_pool_allocator_,
                                                 std::move(match_result),
                                                 std::move(loadback_diff),
                                                 std::move(admission.cache_transfer_pairs),
                                                 std::move(paged_loadback.loadback_transfers),
                                                 std::move(paged_loadback.loadback_nodes_by_group),
                                                 std::move(paged_loadback.loadback_snapshot_refs),
                                                 hybrid_prefix_cache_};
}

std::optional<fsm::ScheduleRetractEvent> Scheduler::scheduleRetract(Request* request) {
    auto full_paged_tokens = request->GetFullPagedTokens(true);
    RequestCacheContext cache_context(*request);
    std::int32_t total_available = cache_context.OccupiedPageCountSnapshot();

    // Overlap scheduling: ExtendResult may grow the token container before the
    // next Acquire runs. Clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    RequestCacheMutation cache_mutation(*request);
    TreeNode* terminal_device_node = cache_mutation.MutableTerminalDeviceNode();
    const std::int32_t alloc_count = hybrid_prefix_cache_
                                         .StepCommit(cache::publish::RetractPrefixPlan{
                                             .full_paged_tokens = full_paged_tokens,
                                             .current_device_node = *terminal_device_node,
                                         })
                                         .device_insert_page_count;
    MatchResult match_result = hybrid_prefix_cache_
                                   .StepCommit(cache::publish::RetractPrefixCommit{
                                       .full_paged_tokens = full_paged_tokens,
                                       .current_device_node = *terminal_device_node,
                                       .pages_to_insert = cache_mutation.TakeFirstLocalKVPages(alloc_count),
                                   })
                                   .match_result;

    const std::int32_t device_matched3 = match_result.device.DepthInPage();
    const std::int32_t host_matched3 = match_result.host.DepthInPage();
    std::int32_t host_pages_needed = 0;
    if (device_matched3 > host_matched3) {
        host_pages_needed = device_matched3 - host_matched3;
    }

    std::map<std::string, std::int32_t> simulated_free;
    if (!hybrid_prefix_cache_
             .Apply(
                 cache::admit::Retract{
                     .match_result = match_result,
                     .host_pages_needed = host_pages_needed,
                 },
                 simulated_free)
             .admitted) {
        return {};
    }
    return fsm::ScheduleRetractEvent{match_result, hybrid_prefix_cache_};
}

LoadBackOperation GenerateLoadBackOp(const std::vector<TreeNode*>& diff, std::vector<TransferPair> extra_transfers,
                                     std::vector<PagedCacheTransferGroup> paged_cache_transfers, cache_op_id op_id) {
    std::vector<TransferPair> transfers;

    for (TreeNode* node : diff) {
        const auto& host_pages = node->Host().Pages();
        const auto& device_pages = node->Device().Pages();
        for (std::size_t i = 0; i < host_pages.size(); ++i) {
            transfers.push_back(TransferPair{CacheKind::kKV, host_pages[i], device_pages[i]});
        }
    }
    transfers.insert(transfers.end(), std::make_move_iterator(extra_transfers.begin()),
                     std::make_move_iterator(extra_transfers.end()));
    return LoadBackOperation{op_id, std::move(transfers), std::move(paged_cache_transfers)};
}

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    // Event applier builds the (device_page, host_page) pairs.
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(fsm::WriteBackDoneEvent{&kv_prefix_cache_, &hybrid_prefix_cache_});
        return std::nullopt;
    }
    // Register op_id so WriteBackDone can route back.
    cache_op_id op_id = hybrid_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    spec.writeback_nodes = request->GetWriteBackNodes<fsm::Retracting>();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end()),
                              true};
}

std::optional<WriteBackOperation> Scheduler::newRetractOperation(Request* retract_request) {
    if (auto event = scheduleRetract(retract_request)) {
        if (auto op = applyEventAndGenerateOp(retract_request, std::move(*event))) {
            return std::move(*op);
        }
    } else {
        spdlog::warn("[Scheduler] Retract failed for request {}: host capacity exhausted, aborting request",
                     retract_request->Id());
        retract_request->Apply(fsm::AbortEvent{});
    }
    return std::nullopt;
}

std::optional<WriteBackOperation> Scheduler::newPagedCacheWriteBackOperation(AdmissionVerdict&& admission) {
    if (admission.paged_cache_writeback_transfers.empty()) {
        return std::nullopt;
    }
    return newPagedCacheWriteBackOperation(std::move(admission.paged_cache_writeback_transfers),
                                           std::move(admission.paged_cache_writeback_nodes_by_group));
}

std::optional<WriteBackOperation> Scheduler::newPagedCacheWriteBackOperation(
    std::vector<PagedCacheTransferGroup> transfers, std::map<std::string, std::vector<TreeNode*>> nodes_by_group) {
    if (transfers.empty()) {
        return std::nullopt;
    }
    cache_op_id op_id = hybrid_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.paged_cache_writeback_nodes_by_group = std::move(nodes_by_group);
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{op_id, {}, std::move(transfers)};
}

// Apply event: state transfer + resource allocation
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event event) {
    RequestCacheContext pre_apply_cache(*request);
    std::int32_t begin = pre_apply_cache.OccupiedPageCountSnapshot();
    request->Apply(event);
    RequestCacheContext cache_context(*request);
    std::vector<std::int32_t> all_pages = cache_context.OccupiedPagesSnapshot();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto info = request->GetPrefillInfo();
    auto op = PrefillOperation{{
        .request_id = request->Id(),
        .request_pool_index = cache_context.RequestPoolIndex(),
        .input_length = info.extend_len,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.input_ids = std::vector<std::int32_t>(info.input_ids.begin(), info.input_ids.end());
    op.shifted_input_ids = std::move(info.shifted_input_ids);
    op.extend_prefix_len = info.already_scheduled_len;

    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event) {
    auto match = event.GetMatchResult();
    auto op = applyPrefillEvent(request, std::move(event));
    RequestCacheContext cache_context(*request);
    RequestCacheMutation cache_mutation(*request);
    (void)hybrid_prefix_cache_.StepCommit(cache::worker::CommitPrefillFirstChunkMetadata{
        .op_base = op,
        .first_raw_position_of_op = op.extend_prefix_len,
        .target_raw_tokens_exclusive = op.extend_prefix_len + op.input_length,
        .tree_prefix_to_commit = *cache_mutation.MutableTerminalDeviceNode(),
        .match_result = match,
        .local_mamba_allocator = cache_context.LocalMambaAllocatorView(),
    });
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, std::move(event));
    RequestCacheContext cache_context(*request);
    RequestCacheMutation cache_mutation(*request);
    (void)hybrid_prefix_cache_.StepCommit(cache::worker::CommitPrefillMetadata{
        .op_base = op,
        .first_raw_position_of_op = op.extend_prefix_len,
        .target_raw_tokens_exclusive = op.extend_prefix_len + op.input_length,
        .tree_prefix_to_commit = *cache_mutation.MutableTerminalDeviceNode(),
        .local_mamba_allocator = cache_context.LocalMambaAllocatorView(),
    });
    return op;
}

template <typename Event>
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent> ||
             std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>)
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens) {
    RequestCacheContext pre_apply_cache(*request);
    std::int32_t begin = pre_apply_cache.OccupiedPageCountSnapshot();
    request->Apply(std::move(event));
    RequestCacheContext cache_context(*request);
    std::vector<std::int32_t> all_pages = cache_context.OccupiedPagesSnapshot();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto op = DecodeOperation{{
        .request_id = request->Id(),
        .request_pool_index = cache_context.RequestPoolIndex(),
        .input_length = decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const std::int32_t first_pos = request->TokenSize();
    const bool came_from_prefill_done = request->Is<fsm::PrefillDone>();

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens);
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
    RequestCacheMutation cache_mutation(*request);
    RequestCacheContext cache_context(*request);
    if (came_from_prefill_done) {
        (void)hybrid_prefix_cache_.StepCommit(cache::worker::CommitDecodeAfterPrefillMetadata{
            .op_base = op,
            .first_raw_position_of_op = first_pos,
            .target_raw_tokens_exclusive = first_pos + op.input_length,
            .tree_prefix_to_commit = *cache_mutation.MutableTerminalDeviceNode(),
            .local_mamba_allocator = cache_context.LocalMambaAllocatorView(),
        });
    } else {
        (void)hybrid_prefix_cache_.StepCommit(cache::worker::CommitDecodeMetadata{
            .op_base = op,
            .first_raw_position_of_op = first_pos,
            .target_raw_tokens_exclusive = first_pos + op.input_length,
            .local_mamba_allocator = cache_context.LocalMambaAllocatorView(),
        });
    }
    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event) {
    auto match = event.GetMatchResult();
    request->Apply(std::move(event));
    if (!request->Is<fsm::Decoding>()) {
        throw std::logic_error(
            "Scheduler::applyEventAndGenerateOp: expected state=Decoding after loadback recovery; got state=" +
            request->StateName());
    }
    RequestCacheContext cache_context(*request);
    std::vector<std::int32_t> all_pages = cache_context.OccupiedPagesSnapshot();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size());
    DecodeOperation op{{
        .request_id = request->Id(),
        .request_pool_index = cache_context.RequestPoolIndex(),
        .input_length = config_.decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = 0,
        .size = sz,
    }};
    op.decode_input_id = request->GetLastToken();
    op.hist_token_len = request->TokenSize() - 1;

    (void)hybrid_prefix_cache_.StepCommit(cache::worker::CommitDecodeRecoveryMetadata{
        .op_base = op,
        .target_raw_tokens_exclusive = request->TokenSize(),
        .match_result = match,
        .local_mamba_allocator = cache_context.LocalMambaAllocatorView(),
    });
    return op;
}

Scheduler::ForwardScheduleResult Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 1;
        if (req->Is<fsm::Submitted>()) return 2;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) {
            // Decode-first if mixed-batch is enabled; prefill-first otherwise.
            return config_.enable_mixed_prefill_decode ? 0 : 3;
        }
        if (req->Is<fsm::Retracted>()) return 4;
        return 9;
    };
    // TP-determinism: tie-break on Request::Id() so the relative order within a
    // priority class is identical across ranks. requests_ is an unordered_map
    // keyed by string id; libstdc++ randomizes string hashing per process, so
    // without the tiebreaker each rank visits candidates in a different order
    // and — when token_budget / page / mamba-slot constraints are tight — picks
    // a different subset to schedule. That made forward_op None on some ranks
    // and non-None on others, deadlocking the next NCCL collective.
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a), pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    std::vector<ForwardOperation> ops;
    std::vector<WriteBackOperation> writeback_ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](auto op, bool uses_pool_slot = false) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        ops.push_back(std::move(op));
    };
    std::vector<LoadBackOperation> loadback_ops;
    auto simulated_free = hybrid_prefix_cache_.InitialSimulatedFree();
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserver_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserver_num_tokens, simulated_free, writeback_ops)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            } else if (!writeback_ops.empty()) {
                break;
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            if (auto ev = schedulePrefillFirstChunk(request, token_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free, writeback_ops)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TransferPair> cache_transfer_pairs = ev->GetCacheTransferPairs();
                std::vector<PagedCacheTransferGroup> paged_cache_transfer_groups = ev->GetPagedCacheTransferGroups();
                auto paged_cache_loadback_nodes_by_group = ev->TakePagedCacheLoadBackNodesByGroup();
                std::vector<PagedCacheSnapshotRef> paged_cache_snapshot_refs = ev->TakePagedCacheSnapshotRefs();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)), true);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty() || !cache_transfer_pairs.empty() || !paged_cache_transfer_groups.empty()) {
                    cache_op_id op_id = hybrid_prefix_cache_.AllocateCacheOpId();
                    if (!paged_cache_snapshot_refs.empty() || !paged_cache_loadback_nodes_by_group.empty()) {
                        CacheOpSpec spec;
                        spec.request_id = request->Id();
                        spec.paged_cache_loadback_nodes_by_group = std::move(paged_cache_loadback_nodes_by_group);
                        spec.paged_cache_loadback_snapshot_refs = std::move(paged_cache_snapshot_refs);
                        cache_op_tracker_[op_id] = std::move(spec);
                    }
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, std::move(cache_transfer_pairs),
                                                              std::move(paged_cache_transfer_groups), op_id));
                }
            } else if (!writeback_ops.empty()) {
                break;
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // If mixed-batch is disabled, skip ALL decode if any prefill was scheduled this round.
            // If mixed-batch is enabled, the priority sort puts decodes first, so this
            // branch is reached before any prefill push.
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecode(request, simulated_free, writeback_ops)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            } else if (!writeback_ops.empty()) {
                break;
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free, writeback_ops)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TransferPair> cache_transfer_pairs = ev->GetCacheTransferPairs();
                std::vector<PagedCacheTransferGroup> paged_cache_transfer_groups = ev->GetPagedCacheTransferGroups();
                auto paged_cache_loadback_nodes_by_group = ev->TakePagedCacheLoadBackNodesByGroup();
                std::vector<PagedCacheSnapshotRef> paged_cache_snapshot_refs = ev->TakePagedCacheSnapshotRefs();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)));
                if (!loadback_diff.empty() || !cache_transfer_pairs.empty() || !paged_cache_transfer_groups.empty()) {
                    cache_op_id op_id = hybrid_prefix_cache_.AllocateCacheOpId();
                    if (!paged_cache_snapshot_refs.empty() || !paged_cache_loadback_nodes_by_group.empty()) {
                        CacheOpSpec spec;
                        spec.request_id = request->Id();
                        spec.paged_cache_loadback_nodes_by_group = std::move(paged_cache_loadback_nodes_by_group);
                        spec.paged_cache_loadback_snapshot_refs = std::move(paged_cache_snapshot_refs);
                        cache_op_tracker_[op_id] = std::move(spec);
                    }
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, std::move(cache_transfer_pairs),
                                                              std::move(paged_cache_transfer_groups), op_id));
                }
            } else if (!writeback_ops.empty()) {
                break;
            }
        }
    }

    // If all active decode requests failed, device memory is exhausted: retract the longest one.
    if (ops.empty() && !candidates.empty()) {
        std::vector<Request*> retract_candidates;
        for (Request* req : candidates) {
            if ((req->Is<fsm::Decoding>() || (req->Is<fsm::PrefillDone>() && config_.role != Role::kD)) &&
                config_.role != Role::kP) {
                retract_candidates.push_back(req);
            }
        }
        if (!retract_candidates.empty()) {
            Request* victim =
                *std::max_element(retract_candidates.begin(), retract_candidates.end(),
                                  [](const Request* a, const Request* b) { return a->TokenSize() < b->TokenSize(); });
            std::vector<WriteBackOperation> wb_ops;
            if (auto op = newRetractOperation(victim)) {
                wb_ops.push_back(std::move(*op));
            }
            return ForwardScheduleResult{.writeback_ops = std::move(wb_ops)};
        }
    }

    return ForwardScheduleResult{.forward_ops = std::move(ops),
                                 .loadback_ops = std::move(loadback_ops),
                                 .writeback_ops = std::move(writeback_ops)};
}

}  // namespace tokenspeed
