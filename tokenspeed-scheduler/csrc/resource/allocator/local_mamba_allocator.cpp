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

#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"

namespace tokenspeed {

LocalMambaAllocator::LocalMambaAllocator(MambaChunkAllocator* allocator) : allocator_{allocator} {}

bool LocalMambaAllocator::AllocateWorking() {
    auto slot = allocator_->Allocate();
    if (!slot.has_value()) return false;
    working_ = std::make_unique<MambaSlot>(std::move(*slot));
    return true;
}

void LocalMambaAllocator::ReleaseWorking() {
    working_.reset();
}

std::int32_t LocalMambaAllocator::WorkingIndex() const {
    return working_ ? working_->Index() : -1;
}

bool LocalMambaAllocator::AllocateCheckpoint(std::int32_t raw_position) {
    auto slot = allocator_->Allocate();
    if (!slot.has_value()) return false;
    checkpoint_ = std::make_unique<MambaSlot>(std::move(*slot));
    checkpoint_position_ = raw_position;
    return true;
}

std::int32_t LocalMambaAllocator::CheckpointIndex() const {
    return checkpoint_ ? checkpoint_->Index() : -1;
}

std::unique_ptr<MambaSlot> LocalMambaAllocator::DetachCheckpoint() {
    checkpoint_position_ = -1;
    return std::move(checkpoint_);
}

std::unique_ptr<MambaSlot> LocalMambaAllocator::DetachWorking() {
    return std::move(working_);
}

}  // namespace tokenspeed
