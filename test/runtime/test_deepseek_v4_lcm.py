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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY.

"""Focused DeepSeek V4 contracts on the shared LCM planner and arena."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (  # noqa: E402
    build_v4_cache_specs,
    build_v4_flat_cache_specs,
)
from tokenspeed.runtime.configs.lcm_layouts import (  # noqa: E402
    deepseek_v4_lcm_fields,
)
from tokenspeed.runtime.configs.lcm_memory_plan import (  # noqa: E402
    plan_lcm_fields,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (  # noqa: E402
    read_deepseek_v4_indexer_fp8_cache,
    write_deepseek_v4_indexer_fp8_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (  # noqa: E402
    DeepseekV4CacheLayout,
    DeepseekV4TokenToKVPool,
    deepseek_v4_cache_layout_from_config,
)
from tokenspeed.runtime.layers.attention.lcm_setup import (  # noqa: E402
    LcmPoolSpec,
    _deepseek_v4_lcm_blocks_needed,
    _deepseek_v4_token_capacity,
)

# Snapshot of deepseek-ai/DeepSeek-V4-Flash config.json. The production layout
# builder normalizes 0 to SWA, assigns indices [0, 43) to the target, and index
# 43 to the MTP draft. Keep the raw serialized order here so this regression
# cannot accidentally size one synthetic 44-layer target.
_PRODUCTION_RAW_RATIOS = (0, 0) + (4, 128) * 20 + (4, 0)
_PRODUCTION_HF_CONFIG = SimpleNamespace(
    compress_ratios=_PRODUCTION_RAW_RATIOS,
    head_dim=512,
    qk_rope_head_dim=64,
    index_head_dim=128,
    sliding_window=128,
    num_hidden_layers=43,
    num_nextn_predict_layers=1,
)


def _production_layouts(
    *,
    use_fp4_indexer_cache=True,
) -> tuple[DeepseekV4CacheLayout, DeepseekV4CacheLayout]:
    target_end = _PRODUCTION_HF_CONFIG.num_hidden_layers
    draft_end = target_end + _PRODUCTION_HF_CONFIG.num_nextn_predict_layers
    kwargs = dict(
        hf_config=_PRODUCTION_HF_CONFIG,
        page_size=64,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
    )
    return (
        deepseek_v4_cache_layout_from_config(
            **kwargs,
            layer_indices=range(target_end),
        ),
        deepseek_v4_cache_layout_from_config(
            **kwargs,
            layer_indices=range(target_end, draft_end),
        ),
    )


_PRODUCTION_TARGET_RATIOS = _production_layouts()[0].layer_ratio
# These are byte-derived planner outputs, not model constants. They lock the
# shared ordinal-plane overlay and prevent regressions to equal-token packing
# or one independent plane set per cache group.
_PRODUCTION_TARGET_PARENT_BYTES = 3_073_536
_PRODUCTION_DRAFT_PARENT_BYTES = 37_632
_PRODUCTION_TARGET_PACKING = {
    "v4.swa_kv": 1,
    "v4.c4a.compressor_state": 2,
    "v4.c4a.compressed_kv": 1,
    "v4.c128a.compressor_state": 2,
    "v4.c128a.compressed_kv": 46,
    "v4.c4a.indexer_compressor_state": 9,
}


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(sliding_window=128)


def _layout(
    ratios=(1, 4, 128),
    *,
    use_fp4_indexer_cache=True,
) -> DeepseekV4CacheLayout:
    return DeepseekV4CacheLayout(
        layer_ratio=tuple(ratios),
        head_dim=512,
        rope_head_dim=64,
        page_size=64,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
        index_head_dim=128,
    )


def _lcm_spec(*, ratios=(1, 4, 128), num_lcm_blocks=2) -> LcmPoolSpec:
    layout = _layout(ratios)
    specs = tuple(
        build_v4_flat_cache_specs(_hf_config(), layer_ratio=layout.layer_ratio)
    )
    fields = deepseek_v4_lcm_fields(layout=layout, group_specs=specs)
    reference = plan_lcm_fields(
        fields,
        logical_block_tokens=math.gcd(*(int(spec.block_size) for spec in specs)),
        num_lcm_blocks=1,
        alignment=256,
        max_padding_fraction=8.0,
    )
    packing = {
        group.group_id: group.cache_blocks_per_lcm_block for group in reference.groups
    }
    plan = plan_lcm_fields(
        fields,
        logical_block_tokens=reference.logical_block_tokens,
        num_lcm_blocks=num_lcm_blocks,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        max_padding_fraction=8.0,
    )
    packed_specs = tuple(
        replace(
            spec,
            cache_blocks_per_lcm_block=packing[spec.group_id],
        )
        for spec in specs
    )
    return LcmPoolSpec(
        memory_plan=plan,
        layer_types=("deepseek_v4",) * len(ratios),
        layer_group_ids=("deepseek_v4",) * len(ratios),
        state_field_dtypes={},
        token_capacity=max(1, num_lcm_blocks * reference.logical_block_tokens),
        extra_paged_groups=packed_specs,
    )


@pytest.mark.parametrize(
    ("group_id", "block_size"),
    (
        ("v4.swa_kv", 64),
        ("v4.c4a.compressor_state", 4),
        ("v4.c4a.compressed_kv", 256),
        ("v4.c128a.compressor_state", 8),
        ("v4.c128a.compressed_kv", 256),
        ("v4.c4a.indexer_compressor_state", 4),
    ),
)
def test_v4_lcm_preserves_heterogeneous_page_spans(group_id, block_size) -> None:
    spec = _lcm_spec()
    groups = {group.group_id: group for group in spec.memory_plan.groups}
    published = {group.group_id: group for group in spec.extra_paged_groups}
    radix = {
        group.group_id: group
        for group in build_v4_cache_specs(
            _hf_config(),
            layer_ratio=_layout().layer_ratio,
        )
    }

    assert spec.memory_plan.logical_block_tokens == 4
    assert radix[group_id].block_size is None
    assert published[group_id].block_size == block_size
    assert (
        published[group_id].cache_blocks_per_lcm_block
        == groups[group_id].cache_blocks_per_lcm_block
    )


def test_v4_production_config_splits_target_and_mtp_before_planning() -> None:
    target_layout, draft_layout = _production_layouts()

    assert len(target_layout.layer_ratio) == 43
    assert target_layout.layer_ratio == tuple(
        max(1, ratio) for ratio in _PRODUCTION_RAW_RATIOS[:43]
    )
    assert draft_layout.layer_ratio == (1,)

    draft_specs = tuple(
        build_v4_flat_cache_specs(
            _PRODUCTION_HF_CONFIG,
            layer_ratio=draft_layout.layer_ratio,
        )
    )
    draft_fields = deepseek_v4_lcm_fields(
        layout=draft_layout,
        group_specs=draft_specs,
    )
    draft_packing = {
        spec.group_id: _PRODUCTION_TARGET_PACKING[spec.group_id] for spec in draft_specs
    }
    draft_plan = plan_lcm_fields(
        draft_fields,
        logical_block_tokens=4,
        num_lcm_blocks=1,
        cache_blocks_per_lcm_block=draft_packing,
        alignment=256,
        max_padding_fraction=8.0,
    )

    assert draft_plan.lcm_block_bytes == _PRODUCTION_DRAFT_PARENT_BYTES
    assert draft_packing == {"v4.swa_kv": 1}


@pytest.mark.parametrize(
    (
        "ratios",
        "use_fp4_indexer_cache",
        "expected_parent_bytes",
        "expected_packing",
    ),
    (
        ((1, 4, 128), True, None, None),
        (
            _PRODUCTION_TARGET_RATIOS,
            True,
            _PRODUCTION_TARGET_PARENT_BYTES,
            _PRODUCTION_TARGET_PACKING,
        ),
        (
            _PRODUCTION_TARGET_RATIOS,
            False,
            _PRODUCTION_TARGET_PARENT_BYTES,
            _PRODUCTION_TARGET_PACKING,
        ),
    ),
    ids=("minimal-mixed", "production-fp4-indexer", "production-fp8-indexer"),
)
def test_v4_lcm_uses_shared_slot_planes_and_bounded_padding(
    ratios,
    use_fp4_indexer_cache,
    expected_parent_bytes,
    expected_packing,
) -> None:
    layout = _layout(
        ratios,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
    )
    specs = tuple(build_v4_flat_cache_specs(_hf_config(), layer_ratio=ratios))
    fields = deepseek_v4_lcm_fields(layout=layout, group_specs=specs)
    plan = plan_lcm_fields(
        fields,
        logical_block_tokens=math.gcd(*(int(spec.block_size) for spec in specs)),
        num_lcm_blocks=1,
        alignment=256,
        max_padding_fraction=8.0,
    )
    packing = {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    }
    raw_by_group = {}
    for field in fields:
        raw_by_group[field.group_id] = (
            raw_by_group.get(field.group_id, 0) + field.payload_bytes
        )

    unshared_bytes = sum(
        packing[field.group_id] * field.payload_bytes for field in fields
    )
    padding = {
        group_id: (plan.lcm_block_bytes // packing[group_id] - payload_bytes)
        / payload_bytes
        for group_id, payload_bytes in raw_by_group.items()
    }

    assert len(plan.planes) == len(ratios)
    if expected_parent_bytes is not None:
        assert plan.lcm_block_bytes == expected_parent_bytes
        assert packing == expected_packing
    assert plan.lcm_block_bytes < unshared_bytes
    assert max(padding.values()) <= 8.0
    assert all(field.page_stride_bytes >= field.payload_bytes for field in plan.fields)
    assert any(
        field.page_stride_bytes > field.payload_bytes
        and field.field_id.endswith(("swa_kv", "compressed_kv", "indexer_kv"))
        for field in plan.fields
    )


def test_v4_admitted_capacity_accounts_for_all_group_parent_demand() -> None:
    num_lcm_blocks = 256
    spec = _lcm_spec(
        ratios=_PRODUCTION_TARGET_RATIOS,
        num_lcm_blocks=num_lcm_blocks,
    )
    groups = {group.group_id: group for group in spec.memory_plan.groups}
    unpacked_specs = tuple(
        replace(group, cache_blocks_per_lcm_block=1)
        for group in spec.extra_paged_groups
    )
    packing = {
        group_id: group.cache_blocks_per_lcm_block for group_id, group in groups.items()
    }
    sizing = dict(
        max_scheduled_tokens=64,
        max_live_requests=2,
        max_context_len=4096,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    capacity = _deepseek_v4_token_capacity(
        unpacked_specs,
        packing,
        num_lcm_blocks=num_lcm_blocks,
        upper_bound_tokens=None,
        **sizing,
    )

    assert (
        _deepseek_v4_lcm_blocks_needed(
            unpacked_specs,
            packing,
            token_capacity=capacity,
            **sizing,
        )
        <= num_lcm_blocks
    )
    assert (
        _deepseek_v4_lcm_blocks_needed(
            unpacked_specs,
            packing,
            token_capacity=capacity + 1,
            **sizing,
        )
        > num_lcm_blocks
    )


def test_v4_fp8_fallback_honors_padded_lcm_page_stride() -> None:
    block_size = 4
    index_head_dim = 128
    payload_bytes = block_size * (index_head_dim + 4)
    padded_stride = payload_bytes + 256
    backing = torch.zeros((3, padded_stride), dtype=torch.uint8)
    padded = backing[:, :payload_bytes]
    contiguous = torch.zeros((3, payload_bytes), dtype=torch.uint8)
    keys = torch.linspace(-2, 2, 2 * index_head_dim, dtype=torch.float32).view(
        2,
        index_head_dim,
    )
    slots = torch.tensor([block_size, 2 * block_size + 1], dtype=torch.int64)

    write_deepseek_v4_indexer_fp8_cache(keys, padded, slots, block_size)
    write_deepseek_v4_indexer_fp8_cache(keys, contiguous, slots, block_size)

    assert torch.equal(padded, contiguous)
    assert torch.equal(
        read_deepseek_v4_indexer_fp8_cache(padded, slots, block_size),
        read_deepseek_v4_indexer_fp8_cache(contiguous, slots, block_size),
    )
    assert not backing[:, payload_bytes:].any()


def test_v4_pool_binds_all_fields_to_one_lcm_backing() -> None:
    layout = _layout()
    spec = _lcm_spec()
    pool = DeepseekV4TokenToKVPool(
        size=spec.pool_size,
        model_dtype=torch.bfloat16,
        layout=layout,
        layer_num=len(layout.layer_ratio),
        device="cpu",
        enable_memory_saver=False,
        max_batch_size=2,
        max_context_len=4096,
        page_size=layout.page_size,
        rank=0,
        hf_config=_hf_config(),
        max_scheduled_tokens=64,
        lcm_spec=spec,
    )

    assert pool.runtime_contract is not None
    assert pool.runtime_contract.block_size == 4
    assert pool.runtime_contract.token_capacity == spec.token_capacity
    assert pool.size == spec.pool_size
    assert pool.runtime_contract.num_lcm_blocks == spec.memory_plan.num_lcm_blocks
    backing = pool.lcm_pool.backing.untyped_storage().data_ptr()
    assert {tensor.untyped_storage().data_ptr() for tensor in pool._all_buffers()} == {
        backing
    }

    pool.lcm_pool.backing.fill_(1)
    pool.clear_kv_buffers()
    assert not pool.lcm_pool.backing.any()
    with pytest.raises(RuntimeError, match="shared parent allocator"):
        pool.move_kv_cache(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
        )
