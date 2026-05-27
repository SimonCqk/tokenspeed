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

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/paged_cache_group.h"

namespace tokenspeed {

// Per-group snapshot held by a TreeNode. RAII returns pages to their tier allocator.
struct PagedCacheGroupSnapshot {
    OwnedPages device_pages;
    OwnedPages host_pages;
    std::int32_t base_logical_page{0};
    std::int32_t raw_token_cursor{0};
    bool sliding{false};

    bool HasDevice() const { return !device_pages.Empty(); }
    bool HasHost() const { return !host_pages.Empty(); }
    const std::vector<std::int32_t>& DevicePageIds() const { return device_pages.Ids(); }
    const std::vector<std::int32_t>& HostPageIds() const { return host_pages.Ids(); }
};

// Snapshot for a TreeNode at a history-alignment-aligned raw-token boundary;
// completeness is tracked per family.
struct PagedCacheSnapshot {
    std::int32_t prefix_len_tokens{0};
    std::map<std::string, PagedCacheGroupSnapshot> groups;
    // Filled by HybridPrefixCache::AttachPagedCacheSnapshotToNode based on
    // which group ids landed in `groups` vs required-per-family lists.
    std::set<PagedCacheGroupFamily> complete_families;
    std::set<PagedCacheGroupFamily> host_complete_families;
    std::int32_t reader_count{0};

    bool IsCompleteFor(PagedCacheGroupFamily f) const { return complete_families.find(f) != complete_families.end(); }
    bool IsCompleteOnHostFor(PagedCacheGroupFamily f) const {
        return host_complete_families.find(f) != host_complete_families.end();
    }
    bool IsPinned() const { return reader_count > 0; }
    void Lock() { ++reader_count; }
    void Unlock() {
        if (reader_count > 0) {
            --reader_count;
        }
    }
};

class PagedCacheSnapshotRef {
public:
    explicit PagedCacheSnapshotRef(PagedCacheSnapshot* snapshot = nullptr) : snapshot_(snapshot) { Lock(); }
    ~PagedCacheSnapshotRef() { Unlock(); }

    PagedCacheSnapshotRef(const PagedCacheSnapshotRef&) = delete;
    PagedCacheSnapshotRef& operator=(const PagedCacheSnapshotRef&) = delete;

    PagedCacheSnapshotRef(PagedCacheSnapshotRef&& other) noexcept
        : snapshot_(std::exchange(other.snapshot_, nullptr)) {}

    PagedCacheSnapshotRef& operator=(PagedCacheSnapshotRef&& other) noexcept {
        if (this != &other) {
            Unlock();
            snapshot_ = std::exchange(other.snapshot_, nullptr);
        }
        return *this;
    }

private:
    void Lock() {
        if (snapshot_ != nullptr) {
            snapshot_->Lock();
        }
    }

    void Unlock() {
        if (snapshot_ != nullptr) {
            snapshot_->Unlock();
            snapshot_ = nullptr;
        }
    }

    PagedCacheSnapshot* snapshot_{nullptr};
};

}  // namespace tokenspeed
