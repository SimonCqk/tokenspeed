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
#include <vector>
#include <cstdint>
#include <memory>

#include "unit_test_helper.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {

namespace {

std::vector<std::span<const std::int32_t>> TokenPages(const token_vec_t& tokens, std::int32_t page_size) {
    const std::size_t num_pages = tokens.size() / page_size;
    std::vector<std::span<const std::int32_t>> pages;
    pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        pages.emplace_back(tokens.data() + i * page_size, static_cast<std::size_t>(page_size));
    }
    return pages;
}

}  // namespace

// page_size=2, device_total=2, host_total=16.
// Step 1: Insert<Device>([1,2], pages=[0]) → node([1,2]) has device=[0].
// Step 2: Match([1,2]) → device.matched=1; Insert<Host> → host=[1].
// Step 3: EnsureCapacityByEvict<Device>(2): evict node([1,2]) device page → device frees up.
// Step 4: Match([1,2]) → device.matched=0, host.matched=1.
TEST(PrefixCacheHostMatchDiag, HostCacheMatchAfterDeviceEvict) {
    PageAllocator device_alloc{2, 3};  // page_size=2, total=3 (2 usable, page 0 reserved)
    PageAllocator host_alloc{2, 16};
    KVPrefixCache cache{&device_alloc, &host_alloc, false};

    OwnedPages dev_owned = device_alloc.Allocate(1);
    ASSERT_EQ(dev_owned.Size(), 1);
    ASSERT_EQ(device_alloc.AvailablePages(), 1);

    token_vec_t tokens12 = {1, 2};
    std::vector<std::span<const std::int32_t>> pages12 = {std::span<const std::int32_t>{tokens12.data(), 2}};
    cache.Insert<ResourceType::Device>(pages12, {}, std::move(dev_owned));

    MatchResult match12 = cache.Match(tokens12);
    ASSERT_EQ(match12.device.DepthInPage(), 1) << "expected device hit after Insert<Device>";
    ASSERT_EQ(match12.host.DepthInPage(), 0) << "no host hit yet";

    // Simulate alloc_host_node via Insert<Host>.
    OwnedPages host_owned = host_alloc.Allocate(1);
    ASSERT_EQ(host_owned.Size(), 1);
    cache.Insert<ResourceType::Host>(pages12, {}, std::move(host_owned));

    MatchResult match12b = cache.Match(tokens12);
    ASSERT_EQ(match12b.device.DepthInPage(), 1) << "device should still match";
    ASSERT_EQ(match12b.host.DepthInPage(), 1) << "host should match after Insert<Host>";

    // Evict node([1,2])'s device page by requesting capacity for 2 pages.
    bool ok = cache.EnsureCapacityByEvict<ResourceType::Device>(2);
    EXPECT_TRUE(ok) << "should have capacity after evicting device page";
    EXPECT_GE(device_alloc.AvailablePages(), 2) << "evicting device page should free enough pages";

    MatchResult match12c = cache.Match(tokens12);
    EXPECT_EQ(match12c.device.DepthInPage(), 0) << "device should be evicted";
    EXPECT_EQ(match12c.host.DepthInPage(), 1) << "host should still match";

    auto diff = match12c.NodesWithout<ResourceType::Device>();
    EXPECT_EQ(diff.size(), 1u) << "NodesWithout<Device>() should return [node([1,2])]";
}

// Same as above but host is attached via direct AttachResource (simulating alloc_host_node behavior).
TEST(PrefixCacheHostMatchDiag, HostCacheMatchAfterDirectAttachResource) {
    PageAllocator device_alloc{2, 2};
    PageAllocator host_alloc{2, 16};
    KVPrefixCache cache{&device_alloc, &host_alloc, false};

    OwnedPages dev_owned = device_alloc.Allocate(1);
    token_vec_t tokens12 = {1, 2};
    std::vector<std::span<const std::int32_t>> pages12 = {std::span<const std::int32_t>{tokens12.data(), 2}};
    cache.Insert<ResourceType::Device>(pages12, {}, std::move(dev_owned));

    MatchResult match12 = cache.Match(tokens12);
    ASSERT_EQ(match12.device.DepthInPage(), 1);
    ASSERT_EQ(match12.host.DepthInPage(), 0);

    // Get the device node via NodesWithout<Host>, then manually attach host resource.
    auto nodes_without_host = match12.NodesWithout<ResourceType::Host>();
    ASSERT_EQ(nodes_without_host.size(), 1u) << "NodesWithout<Host> should have 1 node";
    TreeNode* node = nodes_without_host[0];

    // Manually attach host resource (like alloc_nodes<Host> does).
    OwnedPages host_owned = host_alloc.Allocate(1);
    node->AttachResource(std::make_unique<NodeResource<ResourceType::Host>>(std::move(host_owned)));

    // Verify OnHost().
    EXPECT_TRUE(node->OnHost()) << "node should be OnHost after AttachResource";

    // Match should now return host.matched=1.
    MatchResult match12b = cache.Match(tokens12);
    EXPECT_EQ(match12b.device.DepthInPage(), 1);
    EXPECT_EQ(match12b.host.DepthInPage(), 1) << "host should match after direct AttachResource";

    // Evict device.
    cache.EnsureCapacityByEvict<ResourceType::Device>(2);

    MatchResult match12c = cache.Match(tokens12);
    EXPECT_EQ(match12c.device.DepthInPage(), 0) << "device evicted";
    EXPECT_EQ(match12c.host.DepthInPage(), 1) << "host still matches";
}

TEST(PrefixCacheHostMatchDiag, RawHostStorageHashSeedIgnoresAugmentedMambaMatchCapping) {
    static constexpr std::int32_t kPageSize = 2;
    PageAllocator device_alloc{kPageSize, 16};
    PageAllocator host_alloc{kPageSize, 16};
    KVPrefixCache cache{&device_alloc, &host_alloc, false};
    MambaChunkAllocator mamba_alloc{2};
    HybridPrefixCache hybrid_cache{cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4};

    token_vec_t stored_tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    auto stored_pages = TokenPages(stored_tokens, kPageSize);
    const auto stored_hashes = ComputePagedHashes(stored_pages, "");

    cache.Insert<ResourceType::Device>(stored_pages, {}, device_alloc.Allocate(2), stored_hashes);
    cache.Insert<ResourceType::Host>(stored_pages, {}, host_alloc.Allocate(2), stored_hashes);

    const MatchResult augmented_match = hybrid_cache.MatchPrefix(stored_pages).compat_match;
    ASSERT_EQ(augmented_match.host.DepthInPage(), 0)
        << "regular HybridPrefixCache::Match is Mamba-recovery capped when no Mamba slot exists";

    token_vec_t lookup_tokens = MakeAlignedTokens(/*num_pages=*/3, kPageSize, /*start=*/1);
    auto lookup_pages = TokenPages(lookup_tokens, kPageSize);
    const auto seed = hybrid_cache.LookupRawHostStorageHashSeed(lookup_pages);

    EXPECT_EQ(seed.host_matched_pages, 2);
    EXPECT_EQ(seed.prior_hash_seed, stored_hashes.back());
}

TEST(PrefixCacheHostMatchDiag, RawHostStorageHashSeedReturnsEmptyPriorWhenHostNodeHasNoPageHashes) {
    static constexpr std::int32_t kPageSize = 2;
    PageAllocator device_alloc{kPageSize, 16};
    PageAllocator host_alloc{kPageSize, 16};
    KVPrefixCache cache{&device_alloc, &host_alloc, false};
    HybridPrefixCache hybrid_cache{cache, device_alloc, nullptr, /*mamba_cache_chunk_size=*/0};

    token_vec_t stored_tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    auto stored_pages = TokenPages(stored_tokens, kPageSize);
    cache.Insert<ResourceType::Host>(stored_pages, {}, host_alloc.Allocate(2));

    token_vec_t lookup_tokens = MakeAlignedTokens(/*num_pages=*/3, kPageSize, /*start=*/1);
    auto lookup_pages = TokenPages(lookup_tokens, kPageSize);
    const auto seed = hybrid_cache.LookupRawHostStorageHashSeed(lookup_pages);

    EXPECT_EQ(seed.host_matched_pages, 2);
    EXPECT_TRUE(seed.prior_hash_seed.empty());
}

}  // namespace tokenspeed::test
