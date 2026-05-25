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

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tokenspeed {

void FamilyRegistry::Clear() {
    specs.clear();
    id_to_index_.clear();
    active_match_family_indices.clear();
    active_admit_family_indices.clear();
    active_commit_family_indices.clear();
    active_evict_family_indices.clear();
    active_finish_family_indices.clear();
    active_stats_family_indices.clear();
    active_compatibility_family_indices.clear();
}

const CacheResourceSpec* FamilyRegistry::FindById(const std::string& id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return nullptr;
    return &specs.at(static_cast<std::size_t>(it->second));
}

const CacheResourceSpec& FamilyRegistry::At(std::int32_t family_index) const {
    if (family_index < 0 || family_index >= static_cast<std::int32_t>(specs.size())) {
        throw std::out_of_range("FamilyRegistry::At: family_index out of range");
    }
    return specs.at(static_cast<std::size_t>(family_index));
}

std::int32_t FamilyRegistry::Register(CacheResourceSpec spec, bool active_match, bool active_admit, bool active_commit,
                                      bool active_evict, bool active_finish, bool active_stats,
                                      bool active_compatibility) {
    if (spec.id.empty()) {
        throw std::invalid_argument("FamilyRegistry::Register: id must be non-empty");
    }
    if (id_to_index_.find(spec.id) != id_to_index_.end()) {
        throw std::invalid_argument("FamilyRegistry::Register: duplicate cache family id: " + spec.id);
    }

    const std::int32_t index = static_cast<std::int32_t>(specs.size());
    spec.family_index = index;
    id_to_index_.emplace(spec.id, index);
    specs.push_back(std::move(spec));
    if (active_match) active_match_family_indices.push_back(index);
    if (active_admit) active_admit_family_indices.push_back(index);
    if (active_commit) active_commit_family_indices.push_back(index);
    if (active_evict) active_evict_family_indices.push_back(index);
    if (active_finish) active_finish_family_indices.push_back(index);
    if (active_stats) active_stats_family_indices.push_back(index);
    if (active_compatibility) active_compatibility_family_indices.push_back(index);
    return index;
}

namespace {

struct PagedCacheFamilyRegistryInput {
    std::string group_id;
    PagedCacheGroupConfig config;
    bool required{false};
};

struct FamilyRegistryBuildInput {
    std::int32_t kv_page_size{0};
    bool has_mamba_adjunct{false};
    std::int32_t mamba_cache_chunk_size{0};
    std::span<const PagedCacheFamilyRegistryInput> paged_cache_groups{};
};

struct FamilyRegistryBuildResult {
    FamilyRegistry registry{};
    std::vector<std::string> paged_cache_history_groups{};
    std::vector<std::string> paged_cache_state_groups{};
    std::unordered_set<std::string> paged_cache_history_group_set{};
    std::unordered_set<std::string> paged_cache_state_group_set{};
};

FamilyRegistryBuildResult BuildFamilyRegistry(const FamilyRegistryBuildInput& input) {
    FamilyRegistryBuildResult result{};

    result.registry.Register(
        CacheResourceSpec{
            .id = "kv.token_page",
            .family = CacheFamily::TokenPage,
            .attachment_kind = TreeAttachmentKind::ReusableTree,
            .recoverability = Recoverability::Exact,
            .publication = PublicationKind::CanonicalPrefixIndex,
            .split_policy = SplitPolicy::CarrierKV,
            .rows_per_page = input.kv_page_size,
            .entry_stride_tokens = 1,
            .required_for_recovery = true,
        },
        /*active_match=*/true, /*active_admit=*/true, /*active_commit=*/true,
        /*active_evict=*/true, /*active_finish=*/true, /*active_stats=*/true,
        /*active_compatibility=*/true);

    if (input.has_mamba_adjunct) {
        result.registry.Register(
            CacheResourceSpec{
                .id = "mamba.checkpoint",
                .family = CacheFamily::RecurrentState,
                .attachment_kind = TreeAttachmentKind::ReusableTree,
                .recoverability = Recoverability::AlignedCheckpoint,
                .publication = PublicationKind::AuxiliaryLocalOnly,
                .split_policy = SplitPolicy::CheckpointBoundary,
                .checkpoint_chunk_tokens = input.mamba_cache_chunk_size,
                .state_cohort_id = "mamba.checkpoint",
                .required_for_recovery = true,
            },
            /*active_match=*/true, /*active_admit=*/true, /*active_commit=*/true,
            /*active_evict=*/true, /*active_finish=*/true, /*active_stats=*/true,
            /*active_compatibility=*/true);
    }

    for (const PagedCacheFamilyRegistryInput& group : input.paged_cache_groups) {
        const auto& cfg = group.config;
        CacheFamily family = CacheFamily::CompressedPage;
        if (cfg.family == PagedCacheGroupFamily::State) {
            family = cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow
                         ? CacheFamily::SlidingWindowState
                         : CacheFamily::CompressionTailState;
        }

        if (group.required) {
            if (cfg.family == PagedCacheGroupFamily::History) {
                result.paged_cache_history_groups.push_back(group.group_id);
            } else {
                result.paged_cache_state_groups.push_back(group.group_id);
            }
        }

        CacheResourceSpec spec{
            .id = group.group_id,
            .family = family,
            .attachment_kind =
                group.required ? TreeAttachmentKind::ReusableTree : TreeAttachmentKind::NoneForRequestLocal,
            .recoverability = group.required ? Recoverability::Exact : Recoverability::RequestLocalOnly,
            .publication =
                group.required && cfg.family == PagedCacheGroupFamily::History
                    ? PublicationKind::CanonicalPrefixIndex
                    : (group.required ? PublicationKind::AuxiliaryLocalOnly : PublicationKind::RequestLocalOnly),
            .split_policy = group.required ? SplitPolicy::SnapshotBoundary : SplitPolicy::RequestLocalOnly,
            .rows_per_page = cfg.rows_per_page,
            .entry_stride_tokens = cfg.entry_stride_tokens,
            .sliding_window_tokens = cfg.sliding_window_tokens,
            .state_cohort_id = group.required ? "paged.required" : std::string{},
            .required_for_recovery = group.required,
        };

        result.registry.Register(std::move(spec), /*active_match=*/group.required, /*active_admit=*/true,
                                 /*active_commit=*/group.required, /*active_evict=*/group.required,
                                 /*active_finish=*/true, /*active_stats=*/true,
                                 /*active_compatibility=*/true);
    }

    result.paged_cache_history_group_set = std::unordered_set<std::string>(result.paged_cache_history_groups.begin(),
                                                                           result.paged_cache_history_groups.end());
    result.paged_cache_state_group_set =
        std::unordered_set<std::string>(result.paged_cache_state_groups.begin(), result.paged_cache_state_groups.end());
    return result;
}

}  // namespace

void HybridPrefixCache::RebuildFamilyRegistry() {
    const std::unordered_set<std::string> required_group_set(paged_cache_required_groups_.begin(),
                                                             paged_cache_required_groups_.end());
    std::vector<PagedCacheFamilyRegistryInput> paged_groups;
    paged_groups.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        if (allocator == nullptr) continue;
        paged_groups.push_back({
            .group_id = gid,
            .config = allocator->Config(),
            .required = required_group_set.find(gid) != required_group_set.end(),
        });
    }

    auto result = BuildFamilyRegistry({
        .kv_page_size = kv_prefix_cache_.PageSize(),
        .has_mamba_adjunct = HasMambaAdjunct(),
        .mamba_cache_chunk_size = mamba_cache_chunk_size_,
        .paged_cache_groups = std::span<const PagedCacheFamilyRegistryInput>{paged_groups},
    });

    family_registry_ = std::move(result.registry);
    paged_cache_history_groups_ = std::move(result.paged_cache_history_groups);
    paged_cache_state_groups_ = std::move(result.paged_cache_state_groups);
    paged_cache_history_group_set_ = std::move(result.paged_cache_history_group_set);
    paged_cache_state_group_set_ = std::move(result.paged_cache_state_group_set);
}

}  // namespace tokenspeed
