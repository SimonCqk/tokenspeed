from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_deepseek_v4_disable_kvstore_hides_host_groups_from_scheduler():
    from tokenspeed.runtime.engine.event_loop import (
        _needs_memory_executor,
        _paged_cache_host_group_pages_for_scheduler,
    )

    host_pool = SimpleNamespace(paged_cache_group_page_counts={"v4.swa_kv": 1})

    assert _paged_cache_host_group_pages_for_scheduler(False, host_pool) == {}
    assert _paged_cache_host_group_pages_for_scheduler(True, host_pool) == {
        "v4.swa_kv": 1
    }
    assert not _needs_memory_executor(enable_kvstore=False, enable_mamba_l2=False)
    assert _needs_memory_executor(enable_kvstore=True, enable_mamba_l2=False)
    assert _needs_memory_executor(enable_kvstore=False, enable_mamba_l2=True)


def _make_v4_pool():
    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
        DeepseekV4TokenToKVPool,
        deepseek_v4_cache_layout_from_config,
    )

    hf_config = SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    layout = deepseek_v4_cache_layout_from_config(
        hf_config,
        page_size=64,
        use_fp4_indexer_cache=True,
    )
    pool = DeepseekV4TokenToKVPool(
        size=512,
        model_dtype=torch.bfloat16,
        layout=layout,
        layer_num=3,
        device="cpu",
        enable_memory_saver=False,
        max_batch_size=2,
        max_context_len=512,
        page_size=64,
        rank=0,
        hf_config=hf_config,
        max_scheduled_tokens=64,
    )
    return pool


def _make_host_pool(device_pool, ratio: float = 1.5):
    from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
        DeepseekV4TokenToKVPoolHost,
    )

    return DeepseekV4TokenToKVPoolHost(
        device_pool,
        host_to_device_ratio=ratio,
        host_size_gb=0,
        register_host=False,
    )


def _make_ratio4_paged_transfers():
    return [
        SimpleNamespace(
            group_id="v4.c4a.compressed_kv",
            src_pages=[3, 1, 3],
            dst_pages=[9, 7, 9],
        ),
        SimpleNamespace(
            group_id="v4.c4a.compressed_kv",
            src_pages=[2, 1],
            dst_pages=[8, 7],
        ),
    ]


def _patch_direct_transfer_recorder(monkeypatch):
    from tokenspeed.runtime.cache.transfer import deepseek_v4_pool

    calls = []

    def fake_transfer_kv_direct(
        *,
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        page_size,
    ):
        calls.append(
            (
                src_layers,
                dst_layers,
                src_indices.tolist(),
                dst_indices.tolist(),
                page_size,
            )
        )

    monkeypatch.setattr(
        deepseek_v4_pool,
        "transfer_kv_direct",
        fake_transfer_kv_direct,
    )
    return calls


def _assert_ratio4_direct_call(call, src_tensors, dst_tensors):
    src_layers, dst_layers, src_indices, dst_indices, page_size = call
    assert src_indices == [1, 2, 3]
    assert dst_indices == [7, 8, 9]
    assert page_size == 1
    assert src_layers[0] is src_tensors[0]
    assert src_layers[1] is src_tensors[1]
    assert dst_layers[0] is dst_tensors[0]
    assert dst_layers[1] is dst_tensors[1]


def test_deepseek_v4_host_pool_shapes_and_group_counts():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool, ratio=1.5)

    for group_id, device_pages in device_pool.paged_cache_group_page_counts.items():
        assert host_pool.paged_cache_group_page_counts[group_id] >= device_pages

    assert (
        host_pool.swa_kv_buffer[0].shape[1:] == device_pool.swa_kv_buffer[0].shape[1:]
    )
    assert host_pool.compressed_kv_buffer[1].shape[1:] == (
        device_pool.compressed_kv_buffer[1].shape[1:]
    )
    assert host_pool.indexer_kv_buffer[1].shape[0] == (
        host_pool.compressed_kv_buffer[1].shape[0]
    )
    assert host_pool.indexer_state_buffer[1].shape[1:] == (
        device_pool.indexer_state_buffer[1].shape[1:]
    )
    assert host_pool.page_num > 0
    assert host_pool.total_bytes > 0


def test_deepseek_v4_host_group_page_sizing_keeps_one_usable_page_per_group():
    from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
        _allocate_host_group_pages,
    )

    ratio_counts = _allocate_host_group_pages(
        device_counts={"a": 3, "b": 1},
        page_bytes={"a": 100, "b": 200},
        host_ratio=0.1,
        host_size_gb=0,
    )
    assert ratio_counts == {"a": 2, "b": 2}

    size_counts = _allocate_host_group_pages(
        device_counts={"a": 3, "b": 1},
        page_bytes={"a": 100, "b": 200},
        host_ratio=1.0,
        host_size_gb=1,
    )
    assert size_counts["a"] >= 2
    assert size_counts["b"] >= 2

    capped_counts = _allocate_host_group_pages(
        device_counts={"a": 100, "b": 100},
        page_bytes={"a": 10, "b": 30},
        host_ratio=10.0,
        host_size_gb=0,
        host_budget_bytes=120,
    )
    assert capped_counts["a"] >= 2
    assert capped_counts["b"] >= 2
    assert (
        sum(
            capped_counts[group_id] * {"a": 10, "b": 30}[group_id]
            for group_id in capped_counts
        )
        <= 120
    )

    with pytest.raises(ValueError, match="too small"):
        _allocate_host_group_pages(
            device_counts={"a": 100, "b": 100},
            page_bytes={"a": 10, "b": 30},
            host_ratio=10.0,
            host_size_gb=0,
            host_budget_bytes=79,
        )


def test_deepseek_v4_shadow_capacity_uses_complete_history_limit():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    host_pool.paged_cache_group_page_counts.update(
        {
            "v4.c4a.compressed_kv": 2,
            "v4.c128a.compressed_kv": 32,
        }
    )

    expected_usable_pages = min(
        (
            (host_pool.paged_cache_group_page_counts[spec.group_id] - 1)
            * spec.rows_per_page
            * spec.entry_stride_tokens
            + device_pool.page_size
            - 1
        )
        // device_pool.page_size
        for spec in device_pool.paged_cache_group_specs
        if spec.family == "history"
    )

    assert host_pool._compute_shadow_page_num(device_pool) == expected_usable_pages + 1


def test_deepseek_v4_descriptor_expansion_maps_paged_groups():
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="direct")

    checks = [
        (
            "v4.swa_kv",
            None,
            [("swa_kv_buffer", 0), ("swa_kv_buffer", 1), ("swa_kv_buffer", 2)],
        ),
        (
            "v4.c4a.compressed_kv",
            1,
            [("compressed_kv_buffer", 1), ("indexer_kv_buffer", 1)],
        ),
        ("v4.c4a.compressor_state", 1, [("compressor_state_buffer", 1)]),
        ("v4.c128a.compressor_state", 2, [("compressor_state_buffer", 2)]),
        ("v4.c4a.indexer_compressor_state", 1, [("indexer_state_buffer", 1)]),
    ]
    for group_id, layer_idx, expected in checks:
        refs = transfer_pool.tensor_refs_for_group(group_id, layer_idx=layer_idx)
        assert len(refs) == len(expected)
        for ref, (buffer_name, layer_id) in zip(refs, expected):
            assert ref.layer_id == layer_id
            assert ref.device_tensor is getattr(device_pool, buffer_name)[layer_id]
            assert ref.host_tensor is getattr(host_pool, buffer_name)[layer_id]
            assert ref.page_bytes == ref.device_tensor[0].nbytes


def test_deepseek_v4_paged_pool_prepares_coalesced_transfers():
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="kernel")
    transfers = _make_ratio4_paged_transfers()

    prepared = transfer_pool.prepare_paged_transfers(transfers)

    assert len(prepared) == 1
    assert prepared[0].page_count == 3
    assert prepared[0].span_count == 1
    assert prepared[0].src_indices.tolist() == [1, 2, 3]
    assert prepared[0].dst_indices.tolist() == [7, 8, 9]


def test_deepseek_v4_paged_pool_writeback_prepared_tracks_stats(monkeypatch):
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="kernel")
    calls = _patch_direct_transfer_recorder(monkeypatch)
    prepared = transfer_pool.prepare_paged_transfers(_make_ratio4_paged_transfers())
    c4_bytes = (
        device_pool.compressed_kv_buffer[1][0].nbytes
        + device_pool.indexer_kv_buffer[1][0].nbytes
    )

    transfer_pool.writeback_prepared_paged(prepared)

    assert len(calls) == 1
    _assert_ratio4_direct_call(
        calls[-1],
        (device_pool.compressed_kv_buffer[1], device_pool.indexer_kv_buffer[1]),
        (host_pool.compressed_kv_buffer[1], host_pool.indexer_kv_buffer[1]),
    )
    assert transfer_pool.get_transfer_stats()["D2H"]["v4.c4a.compressed_kv"] == {
        "calls": 1,
        "pages": 3,
        "bytes": 3 * c4_bytes,
    }


def test_deepseek_v4_paged_pool_loadback_prepared_tracks_stats(monkeypatch):
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="kernel")
    calls = _patch_direct_transfer_recorder(monkeypatch)
    prepared = transfer_pool.prepare_paged_transfers(_make_ratio4_paged_transfers())
    c4_bytes = (
        device_pool.compressed_kv_buffer[1][0].nbytes
        + device_pool.indexer_kv_buffer[1][0].nbytes
    )

    transfer_pool.loadback_prepared_paged(prepared, layer_idx=1)

    assert len(calls) == 1
    _assert_ratio4_direct_call(
        calls[-1],
        (host_pool.compressed_kv_buffer[1], host_pool.indexer_kv_buffer[1]),
        (device_pool.compressed_kv_buffer[1], device_pool.indexer_kv_buffer[1]),
    )
    assert transfer_pool.get_transfer_stats()["H2D"]["v4.c4a.compressed_kv"] == {
        "calls": 1,
        "pages": 3,
        "bytes": 3 * c4_bytes,
    }


def test_deepseek_v4_paged_pool_writeback_failure_does_not_update_stats(
    monkeypatch,
):
    from tokenspeed.runtime.cache.transfer import deepseek_v4_pool
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="kernel")
    transfers = _make_ratio4_paged_transfers()

    def failing_transfer_kv_direct(**kwargs):
        del kwargs
        raise RuntimeError("copy submission failed")

    monkeypatch.setattr(
        deepseek_v4_pool,
        "transfer_kv_direct",
        failing_transfer_kv_direct,
    )
    with pytest.raises(RuntimeError, match="copy submission failed"):
        prepared = transfer_pool.prepare_paged_transfers(transfers)
        transfer_pool.writeback_prepared_paged(prepared)
    assert transfer_pool.get_transfer_stats() == {"D2H": {}, "H2D": {}}


def test_deepseek_v4_layer_getters_wait_on_registered_counter():
    device_pool = _make_v4_pool()
    waits = []

    class FakeCounter:
        def wait_until(self, layer_id: int):
            waits.append(layer_id)

    device_pool.register_layer_transfer_counter(FakeCounter())

    device_pool.get_swa_kv_buffer(0)
    device_pool.get_compressed_kv_buffer_2d(1)
    device_pool.get_compressor_state_buffer(1)
    device_pool.get_indexer_kv_buffer_2d(1)
    device_pool.get_indexer_state_buffer(1)
    device_pool.get_kv_buffer(2)

    assert waits == [0, 1, 1, 1, 1, 2]


def test_scheduler_config_accepts_paged_cache_host_group_pages():
    from tokenspeed.runtime.engine.scheduler_utils import make_config

    cfg = make_config(
        num_device_pages=8,
        max_scheduled_tokens=4,
        max_batch_size=2,
        page_size=64,
        num_host_pages=0,
        disable_l2_cache=False,
        enable_l3_storage=False,
        prefetch_threshold=4,
        role="null",
        paged_cache_host_group_pages={"v4.swa_kv": 17},
    )

    assert cfg.paged_cache_host_group_pages == {"v4.swa_kv": 17}
