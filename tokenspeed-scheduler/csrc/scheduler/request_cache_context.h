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
#include <vector>

#include "scheduler/request.h"

namespace tokenspeed {

class RequestLocalCacheState;
class TreeNode;

// Non-owning read view over request-local cache state used for worker-facing
// metadata, statistics, and diagnostics. It does not own allocators, node refs,
// req-pool indices, cache-operation ids, or tree/table state.
class RequestCacheContext {
public:
    explicit RequestCacheContext(Request& request) : request_(request) {}

    RequestCacheContext(const RequestCacheContext&) = delete;
    RequestCacheContext& operator=(const RequestCacheContext&) = delete;

    std::vector<std::int32_t> OccupiedPagesSnapshot() const { return request_.GetOccupiedPages(); }

    std::int32_t OccupiedPageCountSnapshot() const { return static_cast<std::int32_t>(OccupiedPagesSnapshot().size()); }

    // Debug-memory diagnostics read only the request-local KV allocator pages
    // through this request cache-state boundary. Shared radix-tree pages remain
    // owned and reported by HybridPrefixCache diagnostics.
    std::vector<std::int32_t> LocalKVPagesSnapshot() const { return request_.GetLocalAllocatorPages(); }

    std::int32_t RequestPoolIndex() const { return request_.GetReqPoolIndex(); }

    const RequestLocalCacheState& LocalCache() const { return *request_.GetLocalCache(); }

private:
    Request& request_;
};

// Explicit mutable bridge for cache lifecycle operations that need a mutable
// tree observer or request-local page ownership transfer. Keeping this separate
// from RequestCacheContext prevents read-only flattening paths from growing
// hidden mutation authority.
class RequestCacheMutation {
public:
    explicit RequestCacheMutation(Request& request) : request_(request) {}

    RequestCacheMutation(const RequestCacheMutation&) = delete;
    RequestCacheMutation& operator=(const RequestCacheMutation&) = delete;

    TreeNode* MutableTerminalDeviceNode() { return request_.GetMutableDeviceNode(); }

    RequestLocalCacheState& LocalCache() { return *request_.GetLocalCache(); }

private:
    Request& request_;
};

}  // namespace tokenspeed
