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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from tokenspeed.runtime.configs.flat_cache_runtime import (
    FlatPagedCacheRuntimeContract,
    require_positive_int,
)
from tokenspeed.runtime.engine.scheduler_utils import (
    flat_block_tables_from_forward_op,
    flat_cache_batch_from_forward_op,
)


def resolve_flat_runtime_contracts(
    *,
    target_pool: Any,
    target_backend: Any,
    draft_pool: Any = None,
    draft_backend: Any = None,
    flat_kvcache_ext: bool,
) -> tuple[
    FlatPagedCacheRuntimeContract | None,
    FlatPagedCacheRuntimeContract | None,
    bool,
]:
    """Resolve metadata contracts and the union-wide compact table decision.

    Runtime contracts alone select the metadata transport. Backend capability
    only selects whether a Flat scheduler may trim leading holes and publish
    logical bases. A target/draft pair shares one scheduler representation, so
    its common groups and compact capability must agree.
    """

    target_contract = target_pool.runtime_contract
    draft_contract = draft_pool.runtime_contract if draft_pool is not None else None
    target_capable = bool(target_backend.supports_compact_flat_block_tables)
    draft_capable = bool(
        draft_backend.supports_compact_flat_block_tables
        if draft_backend is not None
        else False
    )

    if draft_contract is not None and target_contract is None:
        raise RuntimeError(
            "a FlatKV draft runtime contract requires a target runtime "
            "contract for the scheduler-owned group union"
        )
    if target_contract is not None and draft_contract is not None:
        if target_contract.block_size != draft_contract.block_size:
            raise RuntimeError(
                "target and draft FlatKV contracts disagree on base block_size: "
                f"target={target_contract.block_size}, "
                f"draft={draft_contract.block_size}"
            )
        target_specs = {spec.group_id: spec for spec in target_contract.group_specs}
        draft_specs = {spec.group_id: spec for spec in draft_contract.group_specs}
        missing = set(draft_specs).difference(target_specs)
        if missing:
            raise RuntimeError(
                "draft FlatKV contract is not a subset of the target scheduler "
                f"union: missing={sorted(missing)}"
            )
        for group_id, draft_spec in draft_specs.items():
            target_spec = target_specs[group_id]
            fields = {
                "group_block_size": (
                    target_contract.group_block_sizes[group_id],
                    draft_contract.group_block_sizes[group_id],
                ),
                "retention": (target_spec.retention, draft_spec.retention),
                "family": (target_spec.family, draft_spec.family),
                "cache_blocks_per_lcm_block": (
                    target_spec.cache_blocks_per_lcm_block,
                    draft_spec.cache_blocks_per_lcm_block,
                ),
                "group_page_count": (
                    target_contract.group_page_counts[group_id],
                    draft_contract.group_page_counts[group_id],
                ),
            }
            mismatches = {
                field: {"target": target, "draft": draft}
                for field, (target, draft) in fields.items()
                if target != draft
            }
            if mismatches:
                raise RuntimeError(
                    "target and draft FlatKV contracts disagree for common "
                    f"group {group_id!r}: {mismatches}"
                )

    if flat_kvcache_ext:
        if target_capable and target_contract is None:
            raise RuntimeError(
                "compact FlatKV target backend requires a runtime contract"
            )
        if draft_backend is not None and draft_capable and draft_contract is None:
            raise RuntimeError(
                "compact FlatKV draft backend requires a runtime contract"
            )
        if draft_backend is not None and target_capable != draft_capable:
            raise RuntimeError(
                "target and draft FlatKV backends disagree on compact "
                "block-table capability; the scheduler publishes one union "
                "table representation"
            )

    compact_tables = bool(flat_kvcache_ext and target_capable and target_contract)
    return target_contract, draft_contract, compact_tables


@dataclass(frozen=True, init=False)
class FlatCacheBatchMetadata:
    """Factory-only views tied to one scheduler forward operation.

    Attributes:
        group_ids: Cache group IDs in runtime-contract order.
        num_requests: Number of request rows in each group table.
        max_page_ids: Inclusive maximum page ID accepted for each group.
        block_size: Scheduler base grain retained for uniform legacy consumers.
        full_attention_group_id: The unique ``family="history"`` +
            ``retention="full_history"`` group ID, or ``None`` when the
            contract does not contain exactly one such group.
        compact_tables: Whether group tables use row-local logical bases.
    """

    group_ids: tuple[str, ...]
    _group_tables: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    _group_base_offsets: Mapping[str, torch.Tensor] = field(
        repr=False,
        compare=False,
    )
    num_requests: int
    max_page_ids: Mapping[str, int]
    block_size: int
    full_attention_group_id: str | None
    compact_tables: bool
    _forward_op: Any = field(repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FlatCacheBatchMetadata is factory-only; use from_forward_op()")

    @classmethod
    def from_forward_op(
        cls,
        forward_op: Any,
        *,
        device: torch.device | str,
        contract: FlatPagedCacheRuntimeContract,
        num_requests: int,
        compact_tables: bool,
    ) -> FlatCacheBatchMetadata:
        """Validate CPU exports, pack once, and retain operation provenance."""
        if forward_op is None:
            raise ValueError("forward_op must not be None")
        require_positive_int("num_reqs", num_requests)
        group_ids = tuple(spec.group_id for spec in contract.group_specs)
        max_page_ids = {
            group_id: require_positive_int(
                f"max page ID for {group_id!r}",
                contract.group_page_counts[group_id] - 1,
            )
            for group_id in group_ids
        }
        if (
            not group_ids
            or any(
                not isinstance(group_id, str) or not group_id for group_id in group_ids
            )
            or len(group_ids) != len(set(group_ids))
        ):
            raise ValueError(
                "runtime contract must provide ordered nonempty unique group IDs"
            )
        block_size = require_positive_int("contract block_size", contract.block_size)
        full_attention_ids = tuple(
            spec.group_id
            for spec in contract.group_specs
            if spec.family == "history" and spec.retention == "full_history"
        )
        if compact_tables:
            tables, base_offsets = flat_cache_batch_from_forward_op(
                forward_op,
                device,
                num_reqs=num_requests,
                expected_group_ids=group_ids,
                max_page_ids=max_page_ids,
                required_base_offset_group_ids=frozenset(group_ids),
            )
        else:
            tables = flat_block_tables_from_forward_op(
                forward_op,
                device,
                num_reqs=num_requests,
                expected_group_ids=group_ids,
                max_page_ids=max_page_ids,
            )
            if not isinstance(tables, dict):
                raise TypeError("absolute Flat table bridge returned invalid metadata")
            base_offsets = {}
        return cls._from_validated_tables(
            group_ids=group_ids,
            group_tables=tables,
            group_base_offsets=base_offsets,
            num_requests=num_requests,
            max_page_ids=max_page_ids,
            block_size=block_size,
            full_attention_group_id=(
                full_attention_ids[0] if len(full_attention_ids) == 1 else None
            ),
            compact_tables=compact_tables,
            forward_op=forward_op,
        )

    @classmethod
    def _from_validated_tables(
        cls,
        *,
        group_ids: tuple[str, ...],
        group_tables: Mapping[str, torch.Tensor],
        group_base_offsets: Mapping[str, torch.Tensor],
        num_requests: int,
        max_page_ids: Mapping[str, int],
        block_size: int,
        full_attention_group_id: str | None,
        compact_tables: bool,
        forward_op: Any,
    ) -> FlatCacheBatchMetadata:
        if tuple(group_tables) != group_ids:
            raise ValueError(
                "flat group table mapping must exactly match contract order"
            )
        if compact_tables and tuple(group_base_offsets) != group_ids:
            raise ValueError(
                "flat group base-offset mapping must exactly match contract order"
            )
        if not compact_tables and group_base_offsets:
            raise ValueError("absolute flat metadata must not carry base offsets")
        table_device: torch.device | None = None
        ordered = dict(group_tables)
        ordered_bases = dict(group_base_offsets)
        for group_id, table in ordered.items():
            if not isinstance(table, torch.Tensor):
                raise ValueError(f"flat group {group_id!r} must be a tensor")
            if table.dtype != torch.int32:
                raise ValueError(f"flat group {group_id!r} must use int32")
            if table.ndim != 2 or table.shape[0] != num_requests:
                raise ValueError(f"flat group {group_id!r} has invalid shape")
            if table.shape[1] == 0:
                raise ValueError(f"flat group {group_id!r} has zero width")
            if table.device.type not in ("cpu", "cuda"):
                raise ValueError(f"flat group {group_id!r} must be on CPU or CUDA")
            if compact_tables:
                bases = ordered_bases[group_id]
                if not isinstance(bases, torch.Tensor):
                    raise ValueError(
                        f"flat group {group_id!r} base offsets must be a tensor"
                    )
                if bases.dtype != torch.int32:
                    raise ValueError(
                        f"flat group {group_id!r} base offsets must use int32"
                    )
                if bases.ndim != 1 or bases.shape[0] != num_requests:
                    raise ValueError(f"flat group {group_id!r} has invalid base shape")
                if bases.device != table.device:
                    raise ValueError(
                        f"flat group {group_id!r} table/base devices disagree"
                    )
            if table_device is None:
                table_device = table.device
            elif table.device != table_device:
                raise ValueError("flat group tables must use one CPU/CUDA device")
        nonempty = [tensor for tensor in ordered.values() if tensor.numel()]
        if compact_tables:
            nonempty.extend(
                tensor for tensor in ordered_bases.values() if tensor.numel()
            )
        pointers = {tensor.untyped_storage().data_ptr() for tensor in nonempty}
        if len(pointers) != 1:
            suffix = " and bases" if compact_tables else ""
            raise ValueError(f"flat group tables{suffix} must share packed storage")

        metadata = object.__new__(cls)
        object.__setattr__(metadata, "group_ids", group_ids)
        object.__setattr__(metadata, "_group_tables", MappingProxyType(ordered))
        object.__setattr__(
            metadata,
            "_group_base_offsets",
            MappingProxyType(ordered_bases),
        )
        object.__setattr__(metadata, "num_requests", num_requests)
        object.__setattr__(
            metadata, "max_page_ids", MappingProxyType(dict(max_page_ids))
        )
        object.__setattr__(metadata, "block_size", block_size)
        object.__setattr__(metadata, "full_attention_group_id", full_attention_group_id)
        object.__setattr__(metadata, "compact_tables", bool(compact_tables))
        # A strong reference makes Python/nanobind object identity safe against
        # id reuse until all metadata views become unreachable.
        object.__setattr__(metadata, "_forward_op", forward_op)
        return metadata

    def for_groups(
        self,
        group_ids: tuple[str, ...],
        *,
        owner: str,
    ) -> FlatCacheBatchMetadata:
        """Return one zero-copy owner view over this validated scheduler union."""

        group_ids = tuple(group_ids)
        if not group_ids:
            raise ValueError(f"{owner} flat cache group projection must not be empty")
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"{owner} flat cache group projection has duplicates")
        missing = set(group_ids).difference(self.group_ids)
        if missing:
            raise ValueError(
                f"{owner} flat cache groups are absent from the scheduler union: "
                f"{sorted(missing)}"
            )
        if group_ids == self.group_ids:
            return self
        full_attention_group_id = (
            self.full_attention_group_id
            if self.full_attention_group_id in group_ids
            else None
        )
        return self._from_validated_tables(
            group_ids=group_ids,
            group_tables={
                group_id: self._group_tables[group_id] for group_id in group_ids
            },
            group_base_offsets=(
                {group_id: self._group_base_offsets[group_id] for group_id in group_ids}
                if self.compact_tables
                else {}
            ),
            num_requests=self.num_requests,
            max_page_ids={
                group_id: self.max_page_ids[group_id] for group_id in group_ids
            },
            block_size=self.block_size,
            full_attention_group_id=full_attention_group_id,
            compact_tables=self.compact_tables,
            forward_op=self._forward_op,
        )

    def _validate_active_forward_op(self, active_forward_op: Any) -> None:
        if active_forward_op is not self._forward_op:
            raise RuntimeError(
                "stale flat cache metadata does not match the active forward operation"
            )

    def tables(self, *, active_forward_op: Any) -> Mapping[str, torch.Tensor]:
        """Return all immutable table views after freshness validation."""
        self._validate_active_forward_op(active_forward_op)
        return self._group_tables

    def base_offsets(
        self,
        *,
        active_forward_op: Any,
    ) -> Mapping[str, torch.Tensor]:
        """Return row-aligned logical bases after freshness validation."""

        self._validate_active_forward_op(active_forward_op)
        return self._group_base_offsets

    def require_table(
        self,
        group_id: str,
        *,
        active_forward_op: Any,
    ) -> torch.Tensor:
        """Return one required table after freshness validation."""
        self._validate_active_forward_op(active_forward_op)
        try:
            return self._group_tables[group_id]
        except KeyError:
            raise KeyError(f"missing flat cache group {group_id!r}") from None

    def require_base_offsets(
        self,
        group_id: str,
        *,
        active_forward_op: Any,
    ) -> torch.Tensor:
        """Return one group's row-aligned logical bases."""

        self._validate_active_forward_op(active_forward_op)
        try:
            return self._group_base_offsets[group_id]
        except KeyError:
            raise KeyError(f"missing flat cache group {group_id!r}") from None

    def table_for_layer(
        self,
        pool: object,
        layer_id: int,
        *,
        active_forward_op: Any,
    ) -> torch.Tensor:
        """Resolve a layer and return its fresh operation-bound table."""
        self._validate_active_forward_op(active_forward_op)
        group_id = pool.group_id_for_layer(layer_id)
        return self.require_table(
            group_id,
            active_forward_op=active_forward_op,
        )

    def require_full_attention_table(self, *, active_forward_op: Any) -> torch.Tensor:
        """Return the unique full-history history-group table.

        Args:
            active_forward_op: The scheduler forward operation this batch is
                executing; must be the operation the metadata was built from.

        Returns:
            The ``[num_requests, max_pages]`` int32 page table of the single
            ``family="history"``, ``retention="full_history"`` group.

        Raises:
            RuntimeError: If the metadata is stale, or the contract does not
                contain exactly one full-attention history group.
        """
        self._validate_active_forward_op(active_forward_op)
        if self.full_attention_group_id is None:
            raise RuntimeError(
                "runtime contract does not define exactly one full-history "
                "history group; the MLA flat path requires it"
            )
        return self.require_table(
            self.full_attention_group_id,
            active_forward_op=active_forward_op,
        )
