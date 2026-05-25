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

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/types.h"
#include "scheduler/device_memory_diagnostics.h"

namespace tokenspeed::test {

namespace {

constexpr std::int32_t kPageSize = 2;
constexpr std::int32_t kMambaCacheChunkSize = 4;

std::vector<std::span<const std::int32_t>> TokenPages(const token_vec_t& tokens) {
    std::vector<std::span<const std::int32_t>> pages;
    pages.reserve(tokens.size() / kPageSize);
    for (std::size_t i = 0; i < tokens.size(); i += kPageSize) {
        pages.emplace_back(tokens.data() + i, kPageSize);
    }
    return pages;
}

}  // namespace

TEST(HybridPrefixCacheDeviceStatsTest, AvailableDevicePagesMirrorsAllocatorAcrossAllocateAndRelease) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaCacheChunkSize};

    EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(), static_cast<std::size_t>(device_allocator.AvailablePages()));
    EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(), 7u);

    {
        OwnedPages pages = device_allocator.Allocate(/*num_pages=*/2);
        ASSERT_EQ(pages.Size(), 2);
        EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(),
                  static_cast<std::size_t>(device_allocator.AvailablePages()));
        EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(), 5u);
    }

    EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(), static_cast<std::size_t>(device_allocator.AvailablePages()));
    EXPECT_EQ(hybrid_prefix_cache.AvailableDevicePages(), 7u);
}

TEST(HybridPrefixCacheDeviceStatsTest, DiagnosticsSnapshotReportsTreeAndAllocatorPages) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaCacheChunkSize};

    auto initial_snapshot = hybrid_prefix_cache.CollectDeviceMemoryDiagnostics();
    EXPECT_TRUE(initial_snapshot.tree_device_pages.empty());
    EXPECT_EQ(initial_snapshot.free_device_pages, 7);
    EXPECT_EQ(initial_snapshot.total_device_pages, 7);

    OwnedPages pages = device_allocator.Allocate(/*num_pages=*/2);
    std::vector<std::int32_t> inserted_page_ids = pages.Ids();
    const token_vec_t tokens = {1, 2, 3, 4};
    prefix_cache.Insert<ResourceType::Device>(TokenPages(tokens), {}, std::move(pages));

    auto snapshot = hybrid_prefix_cache.CollectDeviceMemoryDiagnostics();
    ASSERT_EQ(snapshot.tree_device_pages.size(), 2u);
    for (std::int32_t page_id : inserted_page_ids) {
        EXPECT_EQ(snapshot.tree_device_pages.at(page_id), 1);
    }
    EXPECT_EQ(snapshot.free_device_pages, 5);
    EXPECT_EQ(snapshot.total_device_pages, 7);
    EXPECT_TRUE(ValidateDeviceMemoryDiagnostics(/*request_pages=*/{}, snapshot));
}

}  // namespace tokenspeed::test
