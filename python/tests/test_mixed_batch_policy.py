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

from tokenspeed.runtime.engine.mixed_batch_policy import (
    mixed_prefill_fair_share_quantum,
)


def test_mixed_prefill_quantum_uses_active_prefill_rank_fair_share():
    assert mixed_prefill_fair_share_quantum(8192, 64, 1) == 8192
    assert mixed_prefill_fair_share_quantum(8192, 64, 2) == 4096
    assert mixed_prefill_fair_share_quantum(8192, 64, 4) == 2048
    assert mixed_prefill_fair_share_quantum(8192, 64, 8) == 1024


def test_mixed_prefill_quantum_is_block_aligned_without_exceeding_share():
    assert mixed_prefill_fair_share_quantum(1000, 64, 4) == 192


def test_mixed_prefill_quantum_preserves_progress_for_small_chunks():
    assert mixed_prefill_fair_share_quantum(32, 64, 4) == 32
    assert mixed_prefill_fair_share_quantum(128, 64, 8) == 64
    assert mixed_prefill_fair_share_quantum(0, 64, 4) == 0
    assert mixed_prefill_fair_share_quantum(8192, 64, 0) == 8192
