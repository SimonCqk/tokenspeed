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

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from tokenspeed.runtime.configs.paged_cache_spec import (
    PagedCacheGroupSpec,
    full_history_lcm_group_capacities,
)


def flat_cache_debug_enabled() -> bool:
    """Whether expensive, GPU-synchronizing FlatKV validation is enabled."""
    return os.environ.get("TOKENSPEED_FLAT_DEBUG") == "1"


def require_positive_int(name: str, value: object) -> int:
    """Validate that ``value`` is a positive, non-boolean integer.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Returns:
        ``value`` unchanged, typed as ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not an int, or not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class FlatPagedCacheRuntimeContract:
    """Immutable runtime geometry for one scheduler-visible FlatKV group union.

    ``block_size`` is the scheduler's base allocation/hash grain. Individual
    groups may span more raw tokens per child page through
    ``PagedCacheGroupSpec.block_size``; consumers must use
    ``group_block_sizes`` rather than assuming the base grain is uniform.

    ``token_capacity`` is the scheduler admission ceiling. It can be smaller
    than the device pool's physical child-address extent when heterogeneous
    groups pack different numbers of child pages into each LCM parent.
    """

    block_size: int
    num_lcm_blocks: int
    token_capacity: int
    group_specs: tuple[PagedCacheGroupSpec, ...]
    group_page_counts: Mapping[str, int]
    group_block_sizes: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        block_size = require_positive_int("block_size", self.block_size)
        num_lcm_blocks = require_positive_int("num_lcm_blocks", self.num_lcm_blocks)
        token_capacity = require_positive_int("token_capacity", self.token_capacity)
        if not isinstance(self.group_specs, tuple) or not self.group_specs:
            raise ValueError("group_specs must be a non-empty tuple")
        if any(not isinstance(spec, PagedCacheGroupSpec) for spec in self.group_specs):
            raise ValueError("group_specs must contain PagedCacheGroupSpec values")
        group_ids = tuple(spec.group_id for spec in self.group_specs)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group_specs contain duplicate group IDs")
        counts = dict(self.group_page_counts)
        actual_group_ids = set(counts)
        expected_group_ids = set(group_ids)
        if actual_group_ids != expected_group_ids:
            raise ValueError(
                "group_page_counts keys must match group_specs: "
                f"missing={sorted(expected_group_ids - actual_group_ids)} "
                f"extra={sorted(actual_group_ids - expected_group_ids)}"
            )
        counts = {
            group_id: require_positive_int(
                f"group page count for {group_id!r}", counts[group_id]
            )
            for group_id in group_ids
        }
        group_block_sizes = {
            spec.group_id: require_positive_int(
                f"block_size for {spec.group_id!r}",
                spec.block_size if spec.block_size is not None else block_size,
            )
            for spec in self.group_specs
        }
        non_multiple_groups = {
            group_id: group_block_size
            for group_id, group_block_size in group_block_sizes.items()
            if group_block_size % block_size
        }
        if non_multiple_groups:
            raise ValueError(
                "group block sizes must be integer multiples of the scheduler "
                f"base block_size={block_size}: {non_multiple_groups}"
            )
        expected_counts = {
            spec.group_id: num_lcm_blocks
            * require_positive_int(
                f"cache_blocks_per_lcm_block for {spec.group_id!r}",
                spec.cache_blocks_per_lcm_block,
            )
            + 1
            for spec in self.group_specs
        }
        if counts != expected_counts:
            raise ValueError(
                "group page counts must equal num_lcm_blocks * "
                "cache_blocks_per_lcm_block + 1: "
                f"expected={expected_counts}, got={counts}"
            )
        full_history_capacities = full_history_lcm_group_capacities(
            self.group_specs,
            base_block_size=block_size,
            num_lcm_blocks=num_lcm_blocks,
        )
        if full_history_capacities and token_capacity > min(
            full_history_capacities.values()
        ):
            raise ValueError(
                "token_capacity exceeds a full-history group's child-page "
                "capacity: "
                f"token_capacity={token_capacity}, "
                f"group_capacities={full_history_capacities}"
            )
        object.__setattr__(self, "group_page_counts", MappingProxyType(counts))
        object.__setattr__(
            self,
            "group_block_sizes",
            MappingProxyType(group_block_sizes),
        )
