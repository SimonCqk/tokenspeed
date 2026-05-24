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

#include <cstdint>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache_types.h"

namespace tokenspeed::hybrid_prefix_cache::detail {

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

FamilyRegistryBuildResult BuildFamilyRegistry(const FamilyRegistryBuildInput& input);

}  // namespace tokenspeed::hybrid_prefix_cache::detail
