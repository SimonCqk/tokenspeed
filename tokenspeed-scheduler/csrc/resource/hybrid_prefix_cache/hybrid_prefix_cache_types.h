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

#include <concepts>
#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

namespace tokenspeed {

class ForwardOperationBase;
class HybridPrefixCache;
class HybridPrefixCacheTestPeer;
class LocalKVAllocator;
class LocalMambaAllocator;
class TreeNode;

class RequestLocalCacheState {
public:
    RequestLocalCacheState(std::unique_ptr<LocalKVAllocator> local_kv_allocator,
                           std::unique_ptr<LocalMambaAllocator> adjunct_state = nullptr);
    ~RequestLocalCacheState();

    RequestLocalCacheState(RequestLocalCacheState&&) noexcept;
    RequestLocalCacheState& operator=(RequestLocalCacheState&&) noexcept;
    RequestLocalCacheState(const RequestLocalCacheState&) = delete;
    RequestLocalCacheState& operator=(const RequestLocalCacheState&) = delete;

    std::vector<std::int32_t> OccupiedPages(const TreeNode* device_node) const;
    void AcquireKV(std::int32_t tokens);
    OwnedPages TakeFullKVPages();
    OwnedPages TakeFirstKVPages(std::int32_t n);
    std::vector<std::int32_t> LocalKVPages() const;
    std::int32_t TailPageAvailableTokens() const;

private:
    friend class HybridPrefixCache;
    friend class HybridPrefixCacheTestPeer;

    LocalMambaAllocator* AdjunctState();
    const LocalMambaAllocator* AdjunctState() const;
    void SetAdjunctState(std::unique_ptr<LocalMambaAllocator> adjunct_state);
    void ResetAdjunctState();

    std::unique_ptr<LocalKVAllocator> local_kv_allocator_;
    std::unique_ptr<LocalMambaAllocator> adjunct_state_;
};

struct PendingHostWritebackState {
    std::vector<TreeNode*> adjunct_nodes{};
};

struct RecoveryPlan {
    bool recovery_state_available{true};
    TreeNode* protected_recovery_node{nullptr};
    MatchResult compat_match{};
};

struct AdmissionVerdict {
    bool admitted{false};
    std::optional<std::int32_t> mamba_branching_seqlen{};
    std::optional<std::int32_t> mamba_cow_src_index{};
    std::vector<TransferPair> cache_transfer_pairs{};
};

namespace cache {

namespace admit {

struct PrefillFirstChunk {
    const std::string& request_id;
    const MatchResult& match_result;
    std::int32_t device_pages_needed{0};
    std::int32_t tokens_this_round{0};
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};

    using Result = AdmissionVerdict;
};

struct PrefillChunk {
    const std::string& request_id;
    std::int32_t device_pages_needed{0};
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};

    using Result = AdmissionVerdict;
};

struct Decode {
    const std::string& request_id;
    std::int32_t device_pages_needed{0};
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    bool refresh_local_cache_state{false};

    using Result = AdmissionVerdict;
};

struct DecodeFromRetracted {
    const std::string& request_id;
    const MatchResult& match_result;
    std::int32_t device_pages_needed{0};
    std::int32_t target_raw_tokens_exclusive{0};
    TreeNode* protected_recovery_node{nullptr};

    using Result = AdmissionVerdict;
};

struct Retract {
    const MatchResult& match_result;
    std::int32_t host_pages_needed{0};

    using Result = AdmissionVerdict;
};

}  // namespace admit

struct EmptyResult {};

namespace publish {

struct DevicePrefix {
    const std::vector<std::span<const std::int32_t>>& full_paged_tokens;
    std::unique_ptr<DeviceNodeRef>& device_node_ref;
    RequestLocalCacheState& local_cache;
    std::optional<std::int32_t> chunk_begin{};

    struct Result {
        std::int32_t device_insert_page_count{0};
    };
};

struct FinishedRequest {
    const std::vector<std::span<const std::int32_t>>& full_paged_tokens;
    const TreeNode& current_device_node;
    RequestLocalCacheState& local_cache;
    const std::vector<std::string>& page_hashes;

    struct Result {
        MatchResult match_result{};
        std::int32_t device_insert_page_count{0};
    };
};

struct RetractPrefixPlan {
    const std::vector<std::span<const std::int32_t>>& full_paged_tokens;
    const TreeNode& current_device_node;

    struct Result {
        std::int32_t device_insert_page_count{0};
    };
};

struct RetractPrefixCommit {
    const std::vector<std::span<const std::int32_t>>& full_paged_tokens;
    const TreeNode& current_device_node;
    OwnedPages pages_to_insert{};

    struct Result {
        MatchResult match_result{};
        std::int32_t device_insert_page_count{0};
    };
};

}  // namespace publish

namespace materialize {

struct PrefixOnDevice {
    const MatchResult& compat_match;
    bool require_all_pages{false};

    struct Result {
        bool ok{true};
    };
};

struct HostWritebackPages {
    const std::vector<TreeNode*>& write_diff;
    bool ensure_capacity_before_allocate{false};

    struct Result {
        bool ok{true};
        std::vector<TransferPair> cache_transfer_pairs{};
        PendingHostWritebackState pending_state{};
    };
};

}  // namespace materialize

namespace state {

struct CreateRequestLocalCache {
    std::int32_t initial_tokens{0};
    std::int32_t acquire_tokens{0};
    std::optional<std::int32_t> checkpoint_raw_position{};
    bool require_adjunct_state{false};

    struct Result {
        std::unique_ptr<RequestLocalCacheState> local_cache{};
    };
};

struct AcquireRequestLocalCache {
    RequestLocalCacheState& local_cache;
    std::int32_t tokens{0};
    std::optional<std::int32_t> checkpoint_raw_position{};
    bool replace_adjunct_state{false};
    bool require_adjunct_state{false};

    using Result = EmptyResult;
};

struct PublishTreeOwnedRequestState {
    TreeNode& terminal;
    RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

}  // namespace state

namespace worker {

struct CommitPrefillFirstChunkMetadata {
    ForwardOperationBase& op_base;
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    TreeNode& tree_prefix_to_commit;
    const MatchResult& match_result;
    const RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

struct CommitPrefillMetadata {
    ForwardOperationBase& op_base;
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    TreeNode& tree_prefix_to_commit;
    const RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

struct CommitDecodeAfterPrefillMetadata {
    ForwardOperationBase& op_base;
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    TreeNode& tree_prefix_to_commit;
    const RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

struct CommitDecodeMetadata {
    ForwardOperationBase& op_base;
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    const RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

struct CommitDecodeRecoveryMetadata {
    ForwardOperationBase& op_base;
    std::int32_t target_raw_tokens_exclusive{0};
    const MatchResult& match_result;
    const RequestLocalCacheState& local_cache;

    using Result = EmptyResult;
};

}  // namespace worker

template <typename T>
struct IsStepOp : std::false_type {};

template <>
struct IsStepOp<publish::DevicePrefix> : std::true_type {};
template <>
struct IsStepOp<publish::FinishedRequest> : std::true_type {};
template <>
struct IsStepOp<publish::RetractPrefixPlan> : std::true_type {};
template <>
struct IsStepOp<publish::RetractPrefixCommit> : std::true_type {};
template <>
struct IsStepOp<materialize::PrefixOnDevice> : std::true_type {};
template <>
struct IsStepOp<materialize::HostWritebackPages> : std::true_type {};
template <>
struct IsStepOp<state::CreateRequestLocalCache> : std::true_type {};
template <>
struct IsStepOp<state::AcquireRequestLocalCache> : std::true_type {};
template <>
struct IsStepOp<state::PublishTreeOwnedRequestState> : std::true_type {};
template <>
struct IsStepOp<worker::CommitPrefillFirstChunkMetadata> : std::true_type {};
template <>
struct IsStepOp<worker::CommitPrefillMetadata> : std::true_type {};
template <>
struct IsStepOp<worker::CommitDecodeAfterPrefillMetadata> : std::true_type {};
template <>
struct IsStepOp<worker::CommitDecodeMetadata> : std::true_type {};
template <>
struct IsStepOp<worker::CommitDecodeRecoveryMetadata> : std::true_type {};

template <typename Op>
concept StepOp = IsStepOp<std::remove_cvref_t<Op>>::value;

}  // namespace cache

struct StepCommitResult {
    bool ok{true};
    MatchResult match_result{};
    std::int32_t device_insert_page_count{0};
    std::unique_ptr<RequestLocalCacheState> local_cache{};
    std::vector<TransferPair> cache_transfer_pairs{};
    PendingHostWritebackState pending_host_writeback{};
};

struct CacheDeviceMemoryDiagnosticsSnapshot {
    std::unordered_map<std::int32_t, int> tree_device_pages{};
    std::int32_t free_device_pages{0};
    // Usable device pages; page id 0 remains reserved/invalid.
    std::int32_t total_device_pages{0};
};

struct StatsRequest {
    std::optional<std::string> request_id{};
    std::vector<std::string> paged_cache_group_ids{};
    bool include_device_memory_diagnostics{false};
};

struct CacheStatsSnapshot {
    std::size_t available_device_pages{0};
    std::vector<std::string> paged_cache_group_ids{};
    std::map<std::string, std::int32_t> paged_cache_total_pages{};
    std::map<std::string, std::int32_t> paged_cache_available_pages{};
    std::map<std::string, std::int64_t> paged_cache_failed_alloc_count{};
    std::map<std::string, std::vector<std::int32_t>> request_paged_cache_page_ids{};
    std::map<std::string, std::int32_t> request_paged_cache_base_logical_page{};
    std::optional<CacheDeviceMemoryDiagnosticsSnapshot> device_memory_diagnostics{};
};

}  // namespace tokenspeed
