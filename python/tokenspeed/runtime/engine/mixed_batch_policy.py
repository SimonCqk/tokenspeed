# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations


def mixed_prefill_fair_share_quantum(
    chunked_prefill_size: int,
    block_size: int,
    active_prefill_ranks: int,
) -> int:
    """Return the per-rank prefill token budget for a DP mixed forward step.

    Under global decode pressure, the DP group should not admit an unbounded
    prefill chunk on every prefill rank. Instead, active prefill ranks share one
    normal prefill chunk, so aggregate mixed-prefill work stays comparable to a
    non-mixed prefill step while prefill continues to progress.
    """

    chunked_prefill_size = int(chunked_prefill_size)
    if chunked_prefill_size <= 0:
        return 0

    block_size = max(1, int(block_size))
    if chunked_prefill_size <= block_size:
        return chunked_prefill_size

    active_prefill_ranks = max(1, int(active_prefill_ranks))
    raw_quantum = max(block_size, chunked_prefill_size // active_prefill_ranks)
    aligned_quantum = (raw_quantum // block_size) * block_size
    return max(block_size, min(chunked_prefill_size, aligned_quantum))
