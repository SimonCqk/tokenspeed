from types import SimpleNamespace

import pytest

# ruff: noqa: E402

torch = pytest.importorskip("torch")

from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import (
    DeepseekV4PagedCachePool,
)
from tokenspeed.runtime.cache.executor.memory_executor import (
    MemoryExecutor,
    MemoryExecutorConfig,
)
from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4CacheLayout,
    DeepseekV4TokenToKVPool,
)


def _make_pool() -> DeepseekV4TokenToKVPool:
    layout = DeepseekV4CacheLayout(
        layer_ratio=(4,),
        head_dim=64,
        rope_head_dim=64,
        page_size=4,
        use_fp4_indexer_cache=False,
        index_head_dim=128,
    )
    return DeepseekV4TokenToKVPool(
        size=64,
        model_dtype=torch.bfloat16,
        layout=layout,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        max_batch_size=2,
        max_context_len=64,
        page_size=4,
        rank=0,
        hf_config=SimpleNamespace(sliding_window=8),
        max_scheduled_tokens=8,
    )


def _fake_group_pool(group_id: str, pages: int):
    spec = PagedCacheGroupSpec(
        group_id=group_id,
        retention="full_history",
        rows_per_page=4,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
    )
    return SimpleNamespace(
        paged_cache_group_specs=(spec,),
        paged_cache_group_page_counts={group_id: pages},
    )


def _assert_host_shapes_match_device_pages(
    transfer_pool: DeepseekV4PagedCachePool,
    host_total_pages: int,
) -> None:
    assert transfer_pool.page_size() == 1
    layer_buffers = transfer_pool._buffers_by_layer[0]
    assert layer_buffers
    for device_buf, host_buf in layer_buffers:
        assert host_buf.shape == (host_total_pages, *device_buf.shape[1:])
        assert host_buf.dtype == device_buf.dtype


def test_v4_host_page_counts_scale_usable_pages_and_preserve_dummy_page() -> None:
    cfg = MemoryExecutorConfig(layer_num=1, host_ratio=2.0)
    counts = MemoryExecutor._deepseek_v4_host_page_counts(
        _fake_group_pool("g", pages=5),
        cfg,
    )
    assert counts == {"g": 9}

    cfg.host_ratio = 0.5
    counts = MemoryExecutor._deepseek_v4_host_page_counts(
        _fake_group_pool("g", pages=5),
        cfg,
    )
    assert counts == {"g": 3}


def test_v4_host_page_counts_zero_ratio_disables_l2_group() -> None:
    cfg = MemoryExecutorConfig(layer_num=1, host_ratio=0.0)
    counts = MemoryExecutor._deepseek_v4_host_page_counts(
        _fake_group_pool("g", pages=5),
        cfg,
    )
    assert counts == {"g": 0}

    cfg.host_ratio = 2.0
    counts = MemoryExecutor._deepseek_v4_host_page_counts(
        _fake_group_pool("g", pages=1),
        cfg,
    )
    assert counts == {"g": 0}


def test_v4_swa_paged_cache_pool_round_trips_pages() -> None:
    device_pool = _make_pool()
    transfer_pool = DeepseekV4PagedCachePool(
        device_pool=device_pool,
        group_id=V4_SWA_KV_GROUP_ID,
        host_total_pages=8,
        io_backend="kernel",
    )
    _assert_host_shapes_match_device_pages(transfer_pool, host_total_pages=8)
    src = torch.tensor([1, 2], dtype=torch.int64)
    host = torch.tensor([3, 4], dtype=torch.int64)
    dst = torch.tensor([5, 6], dtype=torch.int64)
    device_pool.swa_kv_buffer[0][src[0]] = 3
    device_pool.swa_kv_buffer[0][src[1]] = 5

    transfer_pool.writeback(src, host)
    device_pool.swa_kv_buffer[0][dst] = 0
    transfer_pool.loadback(host, dst, layer_idx=0)

    torch.testing.assert_close(
        device_pool.swa_kv_buffer[0][dst],
        device_pool.swa_kv_buffer[0][src],
    )


def test_v4_compressed_group_copies_indexer_kv_with_shared_page_ids() -> None:
    device_pool = _make_pool()
    transfer_pool = DeepseekV4PagedCachePool(
        device_pool=device_pool,
        group_id=v4_compressed_kv_group_id(4),
        host_total_pages=8,
        io_backend="kernel",
    )
    _assert_host_shapes_match_device_pages(transfer_pool, host_total_pages=8)
    assert len(transfer_pool._buffers_by_layer[0]) == 2
    src = torch.tensor([1], dtype=torch.int64)
    host = torch.tensor([2], dtype=torch.int64)
    dst = torch.tensor([3], dtype=torch.int64)
    device_pool.compressed_kv_buffer[0][src] = 7
    device_pool.indexer_kv_buffer[0][src] = 11

    transfer_pool.writeback(src, host)
    device_pool.compressed_kv_buffer[0][dst] = 0
    device_pool.indexer_kv_buffer[0][dst] = 0
    transfer_pool.loadback(host, dst, layer_idx=0)

    torch.testing.assert_close(
        device_pool.compressed_kv_buffer[0][dst],
        device_pool.compressed_kv_buffer[0][src],
    )
    torch.testing.assert_close(
        device_pool.indexer_kv_buffer[0][dst],
        device_pool.indexer_kv_buffer[0][src],
    )


def test_v4_state_groups_round_trip_pages() -> None:
    device_pool = _make_pool()
    compressor_pool = DeepseekV4PagedCachePool(
        device_pool=device_pool,
        group_id=v4_compressor_state_group_id(4),
        host_total_pages=8,
        io_backend="kernel",
    )
    indexer_state_pool = DeepseekV4PagedCachePool(
        device_pool=device_pool,
        group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
        host_total_pages=8,
        io_backend="kernel",
    )
    _assert_host_shapes_match_device_pages(compressor_pool, host_total_pages=8)
    _assert_host_shapes_match_device_pages(indexer_state_pool, host_total_pages=8)
    assert len(compressor_pool._buffers_by_layer[0]) == 1
    assert len(indexer_state_pool._buffers_by_layer[0]) == 1

    src = torch.tensor([1], dtype=torch.int64)
    host = torch.tensor([2], dtype=torch.int64)
    dst = torch.tensor([3], dtype=torch.int64)
    device_pool.compressor_state_buffer[0][src] = 13
    device_pool.indexer_state_buffer[0][src] = 17

    compressor_pool.writeback(src, host)
    indexer_state_pool.writeback(src, host)
    device_pool.compressor_state_buffer[0][dst] = 0
    device_pool.indexer_state_buffer[0][dst] = 0
    compressor_pool.loadback(host, dst, layer_idx=0)
    indexer_state_pool.loadback(host, dst, layer_idx=0)

    torch.testing.assert_close(
        device_pool.compressor_state_buffer[0][dst],
        device_pool.compressor_state_buffer[0][src],
    )
    torch.testing.assert_close(
        device_pool.indexer_state_buffer[0][dst],
        device_pool.indexer_state_buffer[0][src],
    )
