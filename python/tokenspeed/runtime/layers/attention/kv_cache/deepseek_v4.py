# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE,
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    build_v4_cache_specs,
    build_v4_flat_cache_specs,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.configs.flat_cache_runtime import (
    FlatPagedCacheRuntimeContract,
)
from tokenspeed.runtime.configs.paged_cache_spec import (
    PagedCacheGroupSpec,
    compute_paged_cache_group_page_counts,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_compressed_slot_mapping,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.lcm import LcmCachePool
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.lcm_setup import LcmPoolSpec

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    layer_ratio: tuple[int, ...]
    head_dim: int
    rope_head_dim: int
    page_size: int
    use_fp4_indexer_cache: bool
    index_head_dim: int = 128

    @property
    def swa_token_stride(self) -> int:
        return deepseek_v4_swa_token_stride(self.head_dim, self.rope_head_dim)

    @property
    def swa_scale_dim(self) -> int:
        return deepseek_v4_swa_scale_dim(self.head_dim, self.rope_head_dim)

    @property
    def swa_row_bytes(self) -> int:
        return self.swa_token_stride + self.swa_scale_dim

    def swa_block_bytes(self, rows_per_page: int | None = None) -> int:
        if rows_per_page is None:
            rows_per_page = self.page_size
        block_bytes = rows_per_page * self.swa_row_bytes
        alignment = self.swa_token_stride
        return ((block_bytes + alignment - 1) // alignment) * alignment

    def swa_cell_bytes(self) -> int:
        block_bytes = self.swa_block_bytes()
        return (block_bytes + self.page_size - 1) // self.page_size

    def storage_block_size(self, compress_ratio: int) -> int:
        if compress_ratio > 1:
            return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // compress_ratio)
        return self.page_size

    def compressor_state_block_size(self, compress_ratio: int) -> int:
        if compress_ratio == 4:
            return 4
        if compress_ratio == 128:
            return 8
        return self.page_size

    def compressed_cell_bytes(self, compress_ratio: int) -> int:
        block_bytes = self.swa_block_bytes(self.storage_block_size(compress_ratio))
        return (block_bytes + self.page_size - 1) // self.page_size

    @property
    def indexer_row_bytes(self) -> int:
        if self.use_fp4_indexer_cache:
            return deepseek_v4_indexer_mxfp4_row_bytes(self.index_head_dim)
        return deepseek_v4_indexer_fp8_row_bytes(self.index_head_dim)

    def state_width(self, layer_id: int, *, indexer: bool = False) -> int:
        if indexer:
            return self.index_head_dim * 2
        return self.head_dim * (2 if self.layer_ratio[layer_id] == 4 else 1)

    def cache_cell_size(self, layer_num: int | None = None) -> int:
        """Return bytes per token for the current V4 cache allocation layout."""
        if layer_num is None:
            layer_num = len(self.layer_ratio)
        if layer_num > len(self.layer_ratio):
            raise ValueError(
                "DeepSeek V4 cache layout has fewer layer ratios "
                f"({len(self.layer_ratio)}) than requested layers ({layer_num})"
            )

        fp32_size = torch._utils._element_size(torch.float32)
        cell_size = 0
        for layer_id in range(layer_num):
            ratio = self.layer_ratio[layer_id]
            cell_size += self.swa_cell_bytes()
            if ratio > 1:
                cell_size += self.compressed_cell_bytes(ratio)
                cell_size += self.state_width(layer_id) * 2 * fp32_size
            if ratio == 4:
                indexer_block_bytes = (
                    self.storage_block_size(ratio) * self.indexer_row_bytes
                )
                cell_size += (
                    indexer_block_bytes + self.page_size - 1
                ) // self.page_size
                cell_size += self.state_width(layer_id, indexer=True) * 2 * fp32_size
        return cell_size


def _deepseek_v4_cache_group_page_bytes(
    layout: DeepseekV4CacheLayout,
    specs: Sequence[PagedCacheGroupSpec],
    layer_num: int,
) -> dict[str, int]:
    """Return legacy per-group page bytes from the shared V4 field recipe."""
    from tokenspeed.runtime.configs.lcm_layouts import deepseek_v4_lcm_fields

    if layer_num > len(layout.layer_ratio):
        raise ValueError(
            "DeepSeek V4 cache layout has fewer layer ratios "
            f"({len(layout.layer_ratio)}) than requested layers ({layer_num})"
        )
    profile_layout = (
        layout
        if layer_num == len(layout.layer_ratio)
        else replace(layout, layer_ratio=layout.layer_ratio[:layer_num])
    )
    page_bytes = {spec.group_id: 0 for spec in specs}
    for field_spec in deepseek_v4_lcm_fields(
        layout=profile_layout,
        group_specs=specs,
    ):
        page_bytes[field_spec.group_id] += field_spec.payload_bytes
    return page_bytes


def profile_deepseek_v4_max_num_pages(
    *,
    layout: DeepseekV4CacheLayout,
    hf_config: Any,
    layer_num: int,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_context_len: int,
    available_cache_memory_bytes: int,
    draft_cache_cell_size: int = 0,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return the largest scheduler page budget that fits V4 grouped caches."""
    page_size = int(layout.page_size)
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if available_cache_memory_bytes <= 0:
        return 0
    if draft_cache_cell_size < 0:
        raise ValueError(
            f"draft_cache_cell_size must be >= 0, got {draft_cache_cell_size}"
        )

    draft_cache_cell_size = int(draft_cache_cell_size)
    max_live_requests = int(max_live_requests)
    max_scheduled_tokens = max(0, int(max_scheduled_tokens))
    max_context_len = int(max_context_len)
    specs = tuple(build_v4_cache_specs(hf_config, layer_ratio=layout.layer_ratio))
    page_bytes = _deepseek_v4_cache_group_page_bytes(layout, specs, layer_num)

    def _bytes_for_pages(num_pages: int) -> int:
        num_tokens = int(num_pages) * page_size
        counts = compute_paged_cache_group_page_counts(
            specs,
            max_live_requests=max_live_requests,
            max_scheduled_tokens=max_scheduled_tokens,
            max_total_tokens=num_tokens,
            max_context_len=max_context_len,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        cache_bytes = sum(
            int(counts[gid]) * bytes_per_page
            for gid, bytes_per_page in page_bytes.items()
        )
        return int(cache_bytes + num_tokens * draft_cache_cell_size)

    if _bytes_for_pages(1) > available_cache_memory_bytes:
        return 0

    if not any(int(ratio) > 1 for ratio in layout.layer_ratio[:layer_num]):
        return max(
            1,
            (int(max_live_requests) * int(max_context_len) + page_size - 1)
            // page_size,
        )

    # Fixed bytes cover resident sliding windows, request fragments, and dummy
    # pages. Variable bytes are piecewise linear before and after the global
    # scheduled-token write budget is capped.
    fixed_counts = compute_paged_cache_group_page_counts(
        specs,
        max_live_requests=max_live_requests,
        max_scheduled_tokens=max_scheduled_tokens,
        max_total_tokens=0,
        max_context_len=max_context_len,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    fixed_bytes = sum(
        int(fixed_counts[gid]) * bytes_per_page
        for gid, bytes_per_page in page_bytes.items()
    )
    full_history_slope = Fraction(page_size * draft_cache_cell_size, 1)
    scheduled_slope = Fraction(0, 1)
    scheduled_cap_bytes = 0
    for spec in specs:
        bytes_per_page = page_bytes[spec.group_id]
        if bytes_per_page == 0:
            continue
        raw_per_page = int(spec.rows_per_page) * int(spec.entry_stride_tokens)
        if spec.retention == "full_history":
            full_history_slope += Fraction(page_size * bytes_per_page, raw_per_page)
        elif spec.retention == "sliding_window":
            scheduled_slope += Fraction(page_size * bytes_per_page, raw_per_page)
            scheduled_cap_bytes += (
                ceil_div(max_scheduled_tokens, raw_per_page) * bytes_per_page
            )

    def _pages_from_budget(extra_bytes: int, slope: Fraction) -> int:
        if extra_bytes <= 0 or slope <= 0:
            return 0
        return int(extra_bytes * slope.denominator // slope.numerator)

    cap_pages = ceil_div(max_scheduled_tokens, page_size)
    candidate = 0
    pre_cap_slope = full_history_slope + scheduled_slope
    if cap_pages > 0:
        pre_cap_pages = _pages_from_budget(
            available_cache_memory_bytes - fixed_bytes,
            pre_cap_slope,
        )
        candidate = min(pre_cap_pages, cap_pages - 1)

    post_cap_fixed_bytes = fixed_bytes + scheduled_cap_bytes
    post_cap_pages = _pages_from_budget(
        available_cache_memory_bytes - post_cap_fixed_bytes,
        full_history_slope,
    )
    if post_cap_pages >= cap_pages:
        candidate = max(candidate, post_cap_pages)
    candidate = max(1, candidate)

    while candidate > 0 and _bytes_for_pages(candidate) > available_cache_memory_bytes:
        candidate -= 1
    while _bytes_for_pages(candidate + 1) <= available_cache_memory_bytes:
        candidate += 1
    return int(candidate)


def _split_paged_cache_block_tables_into_v4_metadata(
    paged_cache_block_tables: dict[str, torch.Tensor],
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
]:
    """Split paged-cache dict into V4-named tables + per-sliding-group offsets.

    Returns (swa, {ratio: compressor_state}, indexer_state, swa_base,
    {ratio: compressor_state_base}, indexer_state_base). Unknown group ids
    are ignored. Base offsets are None / missing when the input lacks them.
    """
    offsets = paged_cache_block_table_base_offsets or {}
    swa = paged_cache_block_tables.get(V4_SWA_KV_GROUP_ID)
    indexer_state = paged_cache_block_tables.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    swa_base = offsets.get(V4_SWA_KV_GROUP_ID)
    indexer_state_base = offsets.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    compressor_state: dict[int, torch.Tensor] = {}
    compressor_state_base: dict[int, torch.Tensor] = {}
    for gid, table in paged_cache_block_tables.items():
        ratio = parse_v4_compressor_state_group_id(gid)
        if ratio is None:
            continue
        compressor_state[ratio] = table
        base = offsets.get(gid)
        if base is not None:
            compressor_state_base[ratio] = base
    return (
        swa,
        compressor_state,
        indexer_state,
        swa_base,
        compressor_state_base,
        indexer_state_base,
    )


def _safe_page_ids(
    block_table: torch.Tensor,
    req_indices: torch.Tensor,
    page_indices: torch.Tensor,
) -> torch.Tensor:
    req_i64 = req_indices.to(torch.int64)
    page_i64 = page_indices.to(torch.int64)
    sentinel = torch.full_like(page_i64, -1, dtype=torch.int64)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if rows <= 0 or cols <= 0:
        return sentinel
    valid = (req_i64 >= 0) & (req_i64 < rows) & (page_i64 >= 0) & (page_i64 < cols)
    safe_req = req_i64.clamp(0, rows - 1)
    safe_page = page_i64.clamp(0, cols - 1)
    page_ids = block_table[safe_req, safe_page].to(torch.int64)
    return torch.where(valid, page_ids, sentinel)


def _expand_group_values_for_tokens(
    values: torch.Tensor,
    num_tokens: int,
    name: str,
) -> torch.Tensor:
    if values.numel() == num_tokens:
        return values
    if values.numel() <= 0 or num_tokens % values.numel() != 0:
        raise RuntimeError(
            f"DeepSeek V4 {name} has incompatible shape for packed tokens: "
            f"{values.numel()} entries for {num_tokens} tokens"
        )
    return values.repeat_interleave(num_tokens // values.numel())


def _group_slot_mapping_from_raw(
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_page: int,
    entry_stride_tokens: int = 1,
    base_offsets: torch.Tensor | None = None,
    capacity_pages: int | None = None,
) -> torch.Tensor:
    if rows_per_page <= 0:
        raise ValueError(f"rows_per_page must be > 0, got {rows_per_page}")
    if entry_stride_tokens <= 0:
        raise ValueError(f"entry_stride_tokens must be > 0, got {entry_stride_tokens}")
    if capacity_pages is not None and capacity_pages < 0:
        raise ValueError(f"capacity_pages must be >= 0, got {capacity_pages}")
    pos_i64 = positions.to(torch.int64)
    logical_row = torch.div(pos_i64, entry_stride_tokens, rounding_mode="floor")
    logical_page = torch.div(logical_row, rows_per_page, rounding_mode="floor")
    offsets = logical_row % rows_per_page
    req_indices = _expand_group_values_for_tokens(
        req_indices,
        positions.numel(),
        "request indices",
    )
    table_page = logical_page
    if base_offsets is not None:
        req_i64 = req_indices.to(torch.int64)
        rows = int(base_offsets.shape[0])
        if rows <= 0:
            table_page = logical_page.new_full(logical_page.shape, -1)
        else:
            valid_req = (req_i64 >= 0) & (req_i64 < rows)
            safe_req = req_i64.clamp(0, rows - 1)
            base = base_offsets.to(
                device=logical_page.device,
                dtype=torch.int64,
            )[safe_req]
            table_page = torch.where(valid_req, logical_page - base, -1)
    page_ids = _safe_page_ids(block_table, req_indices, table_page)
    slots = page_ids * rows_per_page + offsets
    # Page 0 is the zero-initialized null page. It may be read by padded rows,
    # but no V4 producer is ever allowed to write through it. The upper bound is
    # the actual owner/component allocation, not a scheduler-global page count.
    # Flat CPU staging already rejects out-of-pool IDs before H2D; this mask is
    # the device-side defense for stale graph buffers and direct/legacy metadata,
    # and keeps every writer kernel safe without a hot-path host sync.
    valid_pages = page_ids > 0
    if capacity_pages is not None:
        valid_pages &= page_ids < capacity_pages
    return torch.where(valid_pages, slots, torch.full_like(slots, -1))


def _mask_invalid_graph_tokens(
    slot_mapping: torch.Tensor,
    is_valid_token: torch.Tensor | None,
) -> torch.Tensor:
    if is_valid_token is None:
        return slot_mapping
    valid = _expand_group_values_for_tokens(
        is_valid_token,
        slot_mapping.numel(),
        "slot validity mask",
    ).to(
        device=slot_mapping.device,
        dtype=torch.bool,
    )
    return torch.where(valid, slot_mapping, torch.full_like(slot_mapping, -1))


def _compressed_boundary_mask(
    positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio <= 1:
        return torch.ones_like(positions, dtype=torch.bool)
    return ((positions.to(torch.int64) + 1) % compress_ratio) == 0


@dataclass
class DeepseekV4CacheMetadata:
    page_size: int
    block_table: torch.Tensor
    allow_legacy_block_table: bool = True
    paged_cache_block_tables: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-group [num_reqs] int32 logical-page bases accompanying compact tables.
    # Sliding consumers index logical_page - base; full-history bases stay zero.
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] = field(
        default_factory=dict
    )
    swa_block_table: torch.Tensor | None = None
    swa_base_logical_page: torch.Tensor | None = None
    compressor_state_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    compressor_state_base_logical_pages: dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    indexer_state_block_table: torch.Tensor | None = None
    indexer_state_base_logical_page: torch.Tensor | None = None
    decode_compressed_slot_mappings: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_compressed_capacity_pages: dict[tuple[int, int], int] = field(
        default_factory=dict
    )

    @property
    def has_legacy_block_table(self) -> bool:
        """Whether the radix-compatible single-table fallback is available."""
        return (
            self.allow_legacy_block_table
            and self.block_table.ndim == 2
            and self.block_table.shape[1] > 0
        )

    def compressed_block_table(
        self,
        compress_ratio: int,
        kv_cache_block_size: int | None = None,
    ) -> torch.Tensor:
        del kv_cache_block_size
        if compress_ratio <= 1:
            if not self.has_legacy_block_table:
                raise RuntimeError(
                    "DeepSeek V4 flat cache metadata cannot use the legacy "
                    "single block_table fallback for an uncompressed group"
                )
            return self.block_table
        table = self.paged_cache_block_tables.get(
            v4_compressed_kv_group_id(compress_ratio)
        )
        if table is None:
            raise RuntimeError(
                "DeepSeek V4 missing paged-cache block table for compressed "
                f"KV group {v4_compressed_kv_group_id(compress_ratio)!r}"
            )
        return table

    @staticmethod
    def safe_page_ids(
        block_table: torch.Tensor,
        req_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _safe_page_ids(block_table, req_indices, page_indices)

    def _update_decode_compressed_slot_mapping(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
        kv_cache_block_size: int,
        capacity_pages: int | None,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = token_to_req_indices.shape[0]
        key = (compress_ratio, kv_cache_block_size)
        out = self.decode_compressed_slot_mappings.get(key)
        if out is None or out.shape[0] < num_tokens or out.device != seq_lens.device:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 compressed slot metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
            self.decode_compressed_slot_mappings[key] = out

        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        if block_table is not self.block_table:
            req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
            query_starts = query_start_loc[req_idx].to(torch.int64)
            query_lens = query_start_loc[req_idx + 1].to(torch.int64) - query_starts
            seq_lens_for_token = seq_lens[req_idx].to(torch.int64)
            token_offsets = torch.arange(
                num_tokens,
                dtype=torch.int64,
                device=seq_lens.device,
            )
            positions = seq_lens_for_token - query_lens + token_offsets - query_starts
            compressed_pos = torch.div(
                positions,
                compress_ratio,
                rounding_mode="floor",
            )
            page_indices = torch.div(
                compressed_pos,
                kv_cache_block_size,
                rounding_mode="floor",
            )
            offsets = compressed_pos % kv_cache_block_size
            base_offsets = self.paged_cache_block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(block_table, req_idx, page_indices)
            valid_pages = page_ids > 0
            if capacity_pages is not None:
                valid_pages &= page_ids < capacity_pages
            valid_slots = valid_pages & _compressed_boundary_mask(
                positions,
                compress_ratio,
            )
            slot_mapping = torch.where(
                valid_slots,
                page_ids * kv_cache_block_size + offsets,
                torch.full_like(page_ids, -1),
            )
            out.copy_(_mask_invalid_graph_tokens(slot_mapping, is_valid_token))
            return out

        mapping = deepseek_v4_compressed_slot_mapping(
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=self.block_table,
            block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            out=out,
        )
        if capacity_pages is not None:
            capacity_slots = capacity_pages * kv_cache_block_size
            valid_slots = (mapping >= kv_cache_block_size) & (mapping < capacity_slots)
            mapping.copy_(
                torch.where(valid_slots, mapping, torch.full_like(mapping, -1))
            )
        if is_valid_token is not None:
            mapping.copy_(_mask_invalid_graph_tokens(mapping, is_valid_token))
        return mapping

    def refresh_decode_compressed_slot_mappings(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        is_valid_token: torch.Tensor | None = None,
    ) -> None:
        for compress_ratio, kv_cache_block_size in list(
            self.decode_compressed_slot_mappings
        ):
            key = (compress_ratio, kv_cache_block_size)
            self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                capacity_pages=self.decode_compressed_capacity_pages.get(key),
                is_valid_token=is_valid_token,
            )

    def compressed_slot_mapping(
        self,
        positions: torch.Tensor,
        compress_ratio: int,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache_block_size: int | None = None,
        capacity_pages: int | None = None,
        use_decode_cache: bool = False,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache_block_size is None:
            kv_cache_block_size = self.page_size
        if capacity_pages is not None and capacity_pages < 0:
            raise ValueError(f"capacity_pages must be >= 0, got {capacity_pages}")
        if not self.has_legacy_block_table and capacity_pages is None:
            raise RuntimeError(
                "DeepSeek V4 flat compressed slot mapping requires the owner "
                "component page capacity"
            )
        key = (compress_ratio, kv_cache_block_size)
        ratio_capacities = {
            existing_capacity
            for (existing_ratio, _), existing_capacity in (
                self.decode_compressed_capacity_pages.items()
            )
            if existing_ratio == compress_ratio
        }
        if capacity_pages is not None and any(
            existing_capacity != capacity_pages
            for existing_capacity in ratio_capacities
        ):
            raise RuntimeError(
                "DeepSeek V4 co-indexed compressed components disagree on page "
                f"capacity for ratio={compress_ratio}: "
                f"first={min(ratio_capacities)}, current={capacity_pages}"
            )
        if capacity_pages is not None:
            self.decode_compressed_capacity_pages[key] = capacity_pages
        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        if (
            use_decode_cache
            and positions.is_cuda
            and (block_table.is_cuda or self.block_table.is_cuda)
        ):
            cached = self.decode_compressed_slot_mappings.get(key)
            if (
                cached is not None
                and cached.shape[0] >= positions.numel()
                and cached.device == seq_lens.device
            ):
                return cached[: positions.numel()]
            mapping = self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                capacity_pages=capacity_pages,
                is_valid_token=is_valid_token,
            )
            return mapping[: positions.numel()]
        compressed_pos = torch.div(
            positions.to(torch.int64), compress_ratio, rounding_mode="floor"
        )
        page_indices = torch.div(
            compressed_pos, kv_cache_block_size, rounding_mode="floor"
        )
        offsets = compressed_pos % kv_cache_block_size
        req_idx = token_to_req_indices[: positions.numel()].long()
        if block_table is self.block_table:
            page_ids = block_table[req_idx, page_indices.long()].to(torch.int64)
        else:
            base_offsets = self.paged_cache_block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(block_table, req_idx, page_indices.long())
        slots = page_ids.to(torch.int64) * kv_cache_block_size + offsets
        valid_pages = page_ids > 0
        if capacity_pages is not None:
            valid_pages &= page_ids < capacity_pages
        valid_slots = valid_pages & _compressed_boundary_mask(
            positions,
            compress_ratio,
        )
        slot_mapping = torch.where(
            valid_slots,
            slots,
            torch.full_like(slots, -1),
        )
        return _mask_invalid_graph_tokens(slot_mapping, is_valid_token)


def deepseek_v4_cache_layout_from_config(
    hf_config,
    page_size: int,
    use_fp4_indexer_cache: bool,
    layer_indices: Iterable[int] | None = None,
) -> DeepseekV4CacheLayout:
    compress_ratios = tuple(hf_config.compress_ratios)
    if layer_indices is None:
        layer_ratios = compress_ratios
    else:
        layer_indices = tuple(layer_indices)
        if any(idx < 0 or idx >= len(compress_ratios) for idx in layer_indices):
            raise ValueError(
                "DeepSeek V4 cache layout layer index out of range: "
                f"indices={layer_indices}, ratios={len(compress_ratios)}"
            )
        layer_ratios = [compress_ratios[idx] for idx in layer_indices]
    raw_layer_ratios = tuple(int(x) for x in layer_ratios)
    for ratio in raw_layer_ratios:
        if ratio not in (0, 1, 4, 128):
            raise ValueError(
                "Unsupported DeepSeek V4 cache compress_ratio="
                f"{ratio}; expected one of 0, 1, 4, or 128"
            )

    return DeepseekV4CacheLayout(
        layer_ratio=tuple(max(1, ratio) for ratio in raw_layer_ratios),
        head_dim=int(hf_config.head_dim),
        rope_head_dim=int(hf_config.qk_rope_head_dim),
        page_size=page_size,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
        index_head_dim=int(getattr(hf_config, "index_head_dim", 128)),
    )


class DeepseekV4TokenToKVPool(BaseTokenToKVPool):
    """DeepSeek V4 fp8_ds_mla cache pool.

    SWA, compressed, compressor-state, and CSA indexer caches are exposed as
    scheduler-visible logical groups. Radix uses distinct group buffers; Flat
    binds the same logical contract to one shared LCM arena. The indexer KV
    field shares the ``v4.c{ratio}a.compressed_kv`` group rather than owning a
    separate page table.
    """

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        layout: DeepseekV4CacheLayout,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        hf_config: Any,
        max_scheduled_tokens: int,
        decode_input_tokens: int = 1,
        overlap_schedule_depth: int = 0,
        lcm_spec: LcmPoolSpec | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"DeepSeek V4 KV pool size must be positive, got {size}")
        if layer_num != len(layout.layer_ratio):
            raise ValueError(
                "DeepSeek V4 KV pool layer_num must match cache layout ratios: "
                f"layer_num={layer_num}, ratios={len(layout.layer_ratio)}"
            )
        super().__init__(
            size=size,
            dtype=torch.uint8,
            device=device,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.model_dtype = model_dtype
        self.layout = layout
        self.layer_num = layer_num
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self._lcm_memory_plan = lcm_spec.memory_plan if lcm_spec is not None else None
        self.lcm_pool: LcmCachePool | None = None
        self.runtime_contract: FlatPagedCacheRuntimeContract | None = None
        self.supports_hierarchical_kv_cache = lcm_spec is None
        self.flat_kv_requires_page_zeroing = lcm_spec is not None

        spec_builder = (
            build_v4_flat_cache_specs if lcm_spec is not None else build_v4_cache_specs
        )
        expected_specs = tuple(spec_builder(hf_config, layer_ratio=layout.layer_ratio))
        if lcm_spec is None:
            group_specs = expected_specs
            group_counts = compute_paged_cache_group_page_counts(
                group_specs,
                max_live_requests=max_batch_size,
                max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
                max_total_tokens=size,
                max_context_len=max_context_len,
                decode_input_tokens=decode_input_tokens,
                overlap_schedule_depth=overlap_schedule_depth,
            )
            self.num_pages = (size + page_size - 1) // page_size + 1
        else:
            plan = lcm_spec.memory_plan
            if size != lcm_spec.pool_size:
                raise ValueError(
                    "DeepSeek V4 LCM pool size must equal its child-address "
                    f"geometry: size={size}, geometry={lcm_spec.pool_size}"
                )
            group_specs = tuple(lcm_spec.extra_paged_groups)
            expected_by_id = {spec.group_id: spec for spec in expected_specs}
            actual_by_id = {spec.group_id: spec for spec in group_specs}
            if actual_by_id.keys() != expected_by_id.keys():
                raise ValueError(
                    "DeepSeek V4 LCM groups disagree with the model layout: "
                    f"expected={sorted(expected_by_id)}, "
                    f"actual={sorted(actual_by_id)}"
                )
            groups_by_id = {group.group_id: group for group in plan.groups}
            if groups_by_id.keys() != actual_by_id.keys():
                raise ValueError(
                    "DeepSeek V4 LCM plan and published groups disagree: "
                    f"plan={sorted(groups_by_id)}, "
                    f"published={sorted(actual_by_id)}"
                )
            for group_id, spec in actual_by_id.items():
                expected = expected_by_id[group_id]
                if (
                    spec.retention != expected.retention
                    or spec.rows_per_page != expected.rows_per_page
                    or spec.entry_stride_tokens != expected.entry_stride_tokens
                    or spec.sliding_window_tokens != expected.sliding_window_tokens
                    or spec.family != expected.family
                    or spec.block_size != expected.block_size
                ):
                    raise ValueError(
                        "DeepSeek V4 LCM group geometry disagrees with the "
                        f"model layout for {group_id!r}"
                    )
                if (
                    spec.cache_blocks_per_lcm_block
                    != groups_by_id[group_id].cache_blocks_per_lcm_block
                ):
                    raise ValueError(
                        "DeepSeek V4 LCM group packing disagrees with the "
                        f"memory plan for {group_id!r}"
                    )
            group_counts = {group.group_id: group.page_count for group in plan.groups}
            self.num_pages = max(group_counts.values())
            with self.memory_saver_adapter.region(
                tag="kv_cache",
                enable_cpu_backup=False,
            ):
                self.lcm_pool = LcmCachePool(plan, device)
            self.runtime_contract = FlatPagedCacheRuntimeContract(
                block_size=plan.logical_block_tokens,
                num_lcm_blocks=plan.num_lcm_blocks,
                token_capacity=lcm_spec.token_capacity,
                group_specs=group_specs,
                group_page_counts=group_counts,
            )

        self.paged_cache_group_specs = tuple(group_specs)
        self.paged_cache_group_page_counts = dict(group_counts)
        self._paged_cache_group_specs_by_id = {
            spec.group_id: spec for spec in self.paged_cache_group_specs
        }
        self._paged_cache_scheduler: object | None = None
        self._paged_cache_state_group_ids = tuple(
            str(spec.group_id)
            for spec in self.paged_cache_group_specs
            if spec.family == "state"
        )

        def group_rows(group_id: str) -> int:
            try:
                return int(self._paged_cache_group_specs_by_id[group_id].rows_per_page)
            except KeyError as exc:
                raise ValueError(
                    f"DeepSeek V4 cache layout is missing group {group_id!r}"
                ) from exc

        is_lcm = self.lcm_pool is not None
        self.swa_block_size = group_rows(V4_SWA_KV_GROUP_ID)
        self._legacy_state_block_size = page_size
        self.swa_block_bytes = layout.swa_block_bytes(self.swa_block_size)
        self.compressed_block_sizes = tuple(
            (
                group_rows(v4_compressed_kv_group_id(ratio))
                if is_lcm and ratio > 1
                else layout.storage_block_size(ratio) if ratio > 1 else page_size
            )
            for ratio in layout.layer_ratio
        )
        self.indexer_block_sizes = tuple(
            (
                group_rows(v4_compressed_kv_group_id(ratio))
                if is_lcm and ratio == 4
                else (
                    max(V4_KERNEL_BLOCK_ROWS, self.compressed_block_sizes[layer_id])
                    if ratio == 4
                    else 0
                )
            )
            for layer_id, ratio in enumerate(layout.layer_ratio)
        )
        self.compressor_state_block_sizes = tuple(
            (
                group_rows(v4_compressor_state_group_id(ratio))
                if ratio > 1
                else page_size
            )
            for ratio in layout.layer_ratio
        )
        self.indexer_state_block_sizes = tuple(
            (group_rows(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID) if ratio == 4 else 0)
            for ratio in layout.layer_ratio
        )
        if self.lcm_pool is not None:
            self.swa_kv_buffer = tuple(
                self.lcm_pool.field(f"layer.{layer_id}.swa_kv", torch.uint8)
                for layer_id in range(layer_num)
            )
            compressed_buffers: list[torch.Tensor | None] = []
            compressor_state_buffers: list[torch.Tensor | None] = []
            indexer_buffers: list[torch.Tensor | None] = []
            indexer_state_buffers: list[torch.Tensor | None] = []
            for layer_id, ratio in enumerate(layout.layer_ratio):
                if ratio > 1:
                    compressed_buffers.append(
                        self.lcm_pool.field(
                            f"layer.{layer_id}.compressed_kv",
                            torch.uint8,
                        )
                    )
                    compressor_state_buffers.append(
                        self.lcm_pool.field(
                            f"layer.{layer_id}.compressor_state",
                            torch.float32,
                        )
                    )
                else:
                    compressed_buffers.append(None)
                    compressor_state_buffers.append(None)
                if ratio == 4:
                    indexer_buffers.append(
                        self.lcm_pool.field(
                            f"layer.{layer_id}.indexer_kv",
                            torch.uint8,
                        )
                    )
                    indexer_state_buffers.append(
                        self.lcm_pool.field(
                            f"layer.{layer_id}.indexer_state",
                            torch.float32,
                        )
                    )
                else:
                    indexer_buffers.append(None)
                    indexer_state_buffers.append(None)
            self.compressed_kv_buffer = tuple(compressed_buffers)
            self.compressor_state_buffer = tuple(compressor_state_buffers)
            self.indexer_kv_buffer = tuple(indexer_buffers)
            self.indexer_state_buffer = tuple(indexer_state_buffers)
        else:
            swa_pages = self.paged_cache_group_page_counts.get(
                V4_SWA_KV_GROUP_ID,
                self.num_pages,
            )
            with self.memory_saver_adapter.region(
                tag="kv_cache",
                enable_cpu_backup=False,
            ):
                self.swa_kv_buffer = [
                    torch.zeros(
                        (swa_pages, self.swa_block_bytes),
                        dtype=torch.uint8,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.compressed_kv_buffer: list[torch.Tensor | None] = []
                self.compressor_state_buffer: list[torch.Tensor | None] = []
                self.indexer_kv_buffer: list[torch.Tensor | None] = []
                self.indexer_state_buffer: list[torch.Tensor | None] = []
                for layer_id, ratio in enumerate(layout.layer_ratio):
                    has_compressed = ratio > 1
                    has_indexer = ratio == 4
                    compressed_block_size = self.compressed_block_sizes[layer_id]
                    compressed_group_id = v4_compressed_kv_group_id(ratio)
                    compressed_pages = self.num_pages
                    if has_compressed:
                        compressed_pages = self.paged_cache_group_page_counts.get(
                            compressed_group_id,
                            self.num_pages,
                        )
                    self.compressed_kv_buffer.append(
                        torch.zeros(
                            (
                                compressed_pages,
                                layout.swa_block_bytes(compressed_block_size),
                            ),
                            dtype=torch.uint8,
                            device=device,
                        )
                        if has_compressed
                        else None
                    )
                    state_block_size = self.compressor_state_block_sizes[layer_id]
                    state_group_id = v4_compressor_state_group_id(ratio)
                    state_pages = self.num_pages
                    if has_compressed:
                        state_pages = self.paged_cache_group_page_counts.get(
                            state_group_id,
                            self.num_pages,
                        )
                    self.compressor_state_buffer.append(
                        torch.empty(
                            (
                                state_pages,
                                state_block_size,
                                layout.state_width(layer_id) * 2,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        if has_compressed
                        else None
                    )
                    indexer_block_size = self.indexer_block_sizes[layer_id]
                    self.indexer_kv_buffer.append(
                        torch.zeros(
                            (
                                compressed_pages,
                                indexer_block_size * layout.indexer_row_bytes,
                            ),
                            dtype=torch.uint8,
                            device=device,
                        )
                        if has_indexer
                        else None
                    )
                    index_state_block_size = self.indexer_state_block_sizes[layer_id]
                    index_state_pages = self.num_pages
                    if has_indexer:
                        index_state_pages = self.paged_cache_group_page_counts.get(
                            V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                            self.num_pages,
                        )
                    self.indexer_state_buffer.append(
                        torch.empty(
                            (
                                index_state_pages,
                                index_state_block_size,
                                layout.state_width(layer_id, indexer=True) * 2,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        if has_indexer
                        else None
                    )

        logger.info(
            "Initialized DeepSeek V4 %s KV pool: %d max group pages, %d layers, "
            "fp4 indexer=%s, compressed block sizes=%s",
            "LCM" if is_lcm else "radix",
            self.num_pages,
            layer_num,
            layout.use_fp4_indexer_cache,
            self.compressed_block_sizes,
        )

    @property
    def prefix_cache_required_group_ids(self) -> tuple[str, ...]:
        return tuple(
            str(spec.group_id)
            for spec in self.paged_cache_group_specs
            if spec.family == "history"
        )

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        self._paged_cache_scheduler = scheduler

    def maybe_log_paged_cache_group_pages(self) -> None:
        scheduler = self._paged_cache_scheduler
        if self.rank != 0 or scheduler is None or not self._paged_cache_state_group_ids:
            return
        if not logger.isEnabledFor(logging.DEBUG):
            return

        parts = []
        for group_id in self._paged_cache_state_group_ids:
            total = scheduler.paged_cache_group_total_pages(group_id)
            available = scheduler.paged_cache_group_available_pages(group_id)
            failed = scheduler.paged_cache_group_failed_alloc_count(group_id)
            parts.append(
                f"{group_id}: used={total - available}/{total}, "
                f"available={available}, failed_alloc={failed}"
            )
        logger.debug("DeepSeek V4 paged-cache state group pages. %s", "; ".join(parts))

    def _require(
        self,
        buffers: Sequence[torch.Tensor | None],
        layer_id: int,
        name: str,
    ) -> torch.Tensor:
        self._wait_for_layer_loadback(layer_id)
        buf = buffers[layer_id]
        if buf is None:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no {name} cache")
        return buf

    def _wait_for_layer_loadback(self, layer_id: int) -> None:
        counter = self.layer_transfer_counter
        if counter is not None:
            counter.wait_until(layer_id)

    def get_swa_kv_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer_loadback(layer_id)
        return self.swa_kv_buffer[layer_id]

    @property
    def state_block_size(self) -> int:
        """Legacy/radix state-block fallback; Flat requires a group table."""
        if self.runtime_contract is not None:
            raise RuntimeError(
                "DeepSeek V4 Flat KV state geometry requires its group-specific "
                "block table"
            )
        return self._legacy_state_block_size

    @property
    def swa_capacity_pages(self) -> int:
        """Writable owner-local SWA capacity shared by every layer, in pages."""

        if not self.swa_kv_buffer:
            return 0
        return int(self.swa_kv_buffer[0].shape[0])

    @property
    def swa_capacity_slots(self) -> int:
        """Writable SWA cache capacity shared by every layer, in token slots.

        Every layer's SWA buffer is allocated with the same page count, so a
        single capacity (pages * tokens per block) bounds the write-slot
        mapping shared across layers. Returns 0 when no SWA buffers exist;
        callers must then mask all slots rather than skip the bounds check.
        """
        return self.swa_capacity_pages * int(self.swa_block_size)

    def get_compressed_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressed_kv_buffer, layer_id, "compressed KV")

    def get_compressed_block_size(self, layer_id: int) -> int:
        return self.compressed_block_sizes[layer_id]

    def get_indexer_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer cache")
        return block_size

    def get_compressor_state_block_size(self, layer_id: int) -> int:
        block_size = self.compressor_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(
                f"DeepSeek V4 layer {layer_id} has no compressor state cache"
            )
        return block_size

    def get_compressor_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressor_state_buffer, layer_id, "compressor state")

    def get_compressor_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_compressor_state_buffer(layer_id)
        block_size = self.get_compressor_state_block_size(layer_id)
        if buf.shape[1] != block_size:
            raise ValueError("compressor-state buffer disagrees with its block size")
        return buf

    def get_indexer_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_kv_buffer, layer_id, "indexer KV")

    def get_indexer_state_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer state cache")
        return block_size

    def get_indexer_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_state_buffer, layer_id, "indexer state")

    def get_indexer_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_indexer_state_buffer(layer_id)
        block_size = self.get_indexer_state_block_size(layer_id)
        if buf.shape[1] != block_size:
            raise ValueError("indexer-state buffer disagrees with its block size")
        return buf

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        buf = self.get_swa_kv_buffer(layer_id)
        return buf, buf

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DeepSeek V4 writes KV cache through V4 attention helpers"
        )

    def _move_fp8_ds_mla_rows(
        self,
        buf: torch.Tensor,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        if tgt_loc.numel() == 0:
            return
        pages = buf.flatten(start_dim=1)
        tgt = tgt_loc.to(torch.int64)
        src = src_loc.to(torch.int64)
        tgt_page = torch.div(tgt, block_size, rounding_mode="floor")
        src_page = torch.div(src, block_size, rounding_mode="floor")
        tgt_pos = tgt % block_size
        src_pos = src % block_size
        token_stride = self.layout.swa_token_stride
        scale_dim = self.layout.swa_scale_dim

        value_offsets = torch.arange(
            token_stride,
            dtype=torch.int64,
            device=buf.device,
        )
        tgt_value = tgt_pos[:, None] * token_stride + value_offsets[None, :]
        src_value = src_pos[:, None] * token_stride + value_offsets[None, :]
        value_rows = pages[src_page[:, None], src_value].clone()
        pages[tgt_page[:, None], tgt_value] = value_rows

        scale_offsets = torch.arange(
            scale_dim,
            dtype=torch.int64,
            device=buf.device,
        )
        scale_base = block_size * token_stride
        tgt_scale = scale_base + tgt_pos[:, None] * scale_dim + scale_offsets[None, :]
        src_scale = scale_base + src_pos[:, None] * scale_dim + scale_offsets[None, :]
        scale_rows = pages[src_page[:, None], src_scale].clone()
        pages[tgt_page[:, None], tgt_scale] = scale_rows

    def _move_rows(
        self,
        buf: torch.Tensor,
        row_bytes: int,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        pages = buf.flatten(start_dim=1)
        offsets = torch.arange(row_bytes, dtype=torch.int64, device=buf.device)
        tgt = tgt_loc.to(torch.int64)
        src = src_loc.to(torch.int64)
        tgt_page = torch.div(tgt, block_size, rounding_mode="floor")
        src_page = torch.div(src, block_size, rounding_mode="floor")
        tgt_offsets = (tgt % block_size)[:, None] * row_bytes + offsets[None, :]
        src_offsets = (src % block_size)[:, None] * row_bytes + offsets[None, :]
        rows = pages[src_page[:, None], src_offsets].clone()
        pages[tgt_page[:, None], tgt_offsets] = rows

    def _compressed_locs_from_token_locs(
        self,
        loc: torch.Tensor,
        *,
        ratio: int,
        block_size: int,
    ) -> torch.Tensor:
        page = torch.div(loc.to(torch.int64), self.page_size, rounding_mode="floor")
        pos = loc.to(torch.int64) % self.page_size
        return page * block_size + torch.div(pos, ratio, rounding_mode="floor")

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if self.lcm_pool is not None:
            raise RuntimeError(
                "DeepSeek V4 LCM pages are relocated by the shared parent "
                "allocator, not the radix token compactor"
            )
        if tgt_loc.numel() == 0:
            return
        for layer_id in range(self.layer_num):
            self._move_fp8_ds_mla_rows(
                self.swa_kv_buffer[layer_id],
                tgt_loc,
                src_loc,
                self.swa_block_size,
            )
            buf = self.compressed_kv_buffer[layer_id]
            if buf is not None:
                ratio = self.layout.layer_ratio[layer_id]
                block_size = self.get_compressed_block_size(layer_id)
                self._move_fp8_ds_mla_rows(
                    buf,
                    self._compressed_locs_from_token_locs(
                        tgt_loc, ratio=ratio, block_size=block_size
                    ),
                    self._compressed_locs_from_token_locs(
                        src_loc, ratio=ratio, block_size=block_size
                    ),
                    block_size,
                )
            for buffers, row_bytes in (
                (self.indexer_kv_buffer, self.layout.indexer_row_bytes),
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    ratio = self.layout.layer_ratio[layer_id]
                    block_size = self.get_indexer_block_size(layer_id)
                    self._move_rows(
                        buf,
                        row_bytes,
                        self._compressed_locs_from_token_locs(
                            tgt_loc, ratio=ratio, block_size=block_size
                        ),
                        self._compressed_locs_from_token_locs(
                            src_loc, ratio=ratio, block_size=block_size
                        ),
                        block_size,
                    )
            for buffers in (self.compressor_state_buffer, self.indexer_state_buffer):
                buf = buffers[layer_id]
                if buf is not None:
                    self._move_rows(
                        buf,
                        buf.shape[-1],
                        tgt_loc,
                        src_loc,
                        buf.shape[1],
                    )

    def _all_buffers(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for layer_id in range(self.layer_num):
            out.append(self.swa_kv_buffer[layer_id])
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    out.append(buf)
        return out

    def zero_new_pages(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            if self.lcm_pool is None:
                raise RuntimeError("page zeroing is only valid for V4 LCM cache")
            self.lcm_pool.zero_pages(new_page_ids)

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        if self.lcm_pool is not None:
            self.lcm_pool.backing.zero_()
            return
        super().clear_kv_buffers()

    def get_kv_size_bytes(self) -> int:
        if self.lcm_pool is not None:
            return int(self.lcm_pool.backing.nbytes)
        return int(
            sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self._all_buffers())
        )

    def get_contiguous_buf_infos(self):
        if self.lcm_pool is not None:
            raise RuntimeError("DeepSeek V4 LCM cache does not support legacy transfer")
        buffers = self._all_buffers()
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        offsets = []
        cursor = start_idx
        for layer_id in range(self.layer_num):
            layer_offsets = [cursor]
            cursor += 1
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                if buffers[layer_id] is not None:
                    layer_offsets.append(cursor)
                    cursor += 1
            offsets.append(layer_offsets)
        return offsets

    def get_cpu_copy(self, token_indices: list[int]) -> list[torch.Tensor]:
        del token_indices
        raise NotImplementedError(
            "DeepSeek V4 does not support legacy token-indexed KV cache offload; "
            "use the group-paged L2 KVStore path instead."
        )

    def load_cpu_copy(self, kv_cache_cpu, token_indices: list[int]) -> None:
        del kv_cache_cpu, token_indices
        raise NotImplementedError(
            "DeepSeek V4 does not support legacy token-indexed KV cache reload; "
            "use the group-paged L2 KVStore path instead."
        )
