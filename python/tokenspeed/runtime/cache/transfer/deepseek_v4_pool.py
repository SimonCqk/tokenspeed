from __future__ import annotations

import torch

from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
)


def _empty_host_like(device_tensor: torch.Tensor, pages: int) -> torch.Tensor:
    shape = (int(pages), *device_tensor.shape[1:])
    kwargs = {
        "dtype": device_tensor.dtype,
        "device": "cpu",
    }
    if device_tensor.is_cuda:
        kwargs["pin_memory"] = True
    try:
        return torch.empty(shape, **kwargs)
    except RuntimeError:
        kwargs.pop("pin_memory", None)
        return torch.empty(shape, **kwargs)


class DeepseekV4PagedCachePool:
    """Host transfer pool for one DeepSeek V4 paged-cache group."""

    def __init__(
        self,
        device_pool,
        group_id: str,
        host_total_pages: int,
        io_backend: str,
    ) -> None:
        if host_total_pages <= 0:
            raise ValueError(
                f"DeepSeek V4 host group {group_id!r} requires positive pages"
            )
        self.kind = str(group_id)
        self.device_pool = device_pool
        self.io_backend = io_backend
        self.host_layout = "layer_first"
        self.page_num = int(host_total_pages)
        self._counter = LayerDoneCounter(self.num_layers())
        device_pool.register_layer_transfer_counter(self._counter)
        self._buffers_by_layer = self._build_host_buffers()

    @property
    def device(self):
        return self.device_pool.device

    def page_size(self) -> int:
        return 1

    def num_layers(self) -> int:
        return int(self.device_pool.layer_num)

    def supports_layerwise_loadback(self) -> bool:
        return True

    def get_layer_done_counter(self) -> LayerDoneCounter:
        return self._counter

    def local_layer_idx(self, global_layer_id: int) -> int:
        return int(global_layer_id)

    def _build_host_buffers(self) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        out: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for layer_id, ratio in enumerate(self.device_pool.layout.layer_ratio):
            layer_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
            if self.kind == V4_SWA_KV_GROUP_ID:
                device_buf = self.device_pool.swa_kv_buffer[layer_id]
                layer_buffers.append(
                    (device_buf, _empty_host_like(device_buf, self.page_num))
                )
            compressed_group_id = v4_compressed_kv_group_id(ratio)
            if ratio > 1 and self.kind == compressed_group_id:
                compressed = self.device_pool.compressed_kv_buffer[layer_id]
                if compressed is not None:
                    layer_buffers.append(
                        (compressed, _empty_host_like(compressed, self.page_num))
                    )
                indexer = self.device_pool.indexer_kv_buffer[layer_id]
                if indexer is not None:
                    layer_buffers.append(
                        (indexer, _empty_host_like(indexer, self.page_num))
                    )
            compressor_ratio = parse_v4_compressor_state_group_id(self.kind)
            if ratio > 1 and compressor_ratio == ratio:
                state = self.device_pool.compressor_state_buffer[layer_id]
                if state is not None:
                    layer_buffers.append(
                        (state, _empty_host_like(state, self.page_num))
                    )
            if ratio == 4 and self.kind == V4_INDEXER_COMPRESSOR_STATE_GROUP_ID:
                indexer_state = self.device_pool.indexer_state_buffer[layer_id]
                if indexer_state is not None:
                    layer_buffers.append(
                        (indexer_state, _empty_host_like(indexer_state, self.page_num))
                    )
            out.append(layer_buffers)
        if not any(out):
            raise ValueError(
                f"DeepSeek V4 paged-cache group {self.kind!r} has no buffers"
            )
        return out

    def _copy_device_to_host(
        self, device_buf: torch.Tensor, host_buf: torch.Tensor, src, dst
    ) -> None:
        src = src.to(device=device_buf.device, dtype=torch.int64, non_blocking=True)
        dst = dst.to(device="cpu", dtype=torch.int64, non_blocking=True)
        host_buf.index_copy_(
            0, dst, device_buf.index_select(0, src).to("cpu", non_blocking=True)
        )

    def _copy_host_to_device(
        self, host_buf: torch.Tensor, device_buf: torch.Tensor, src, dst
    ) -> None:
        src = src.to(device="cpu", dtype=torch.int64, non_blocking=True)
        dst = dst.to(device=device_buf.device, dtype=torch.int64, non_blocking=True)
        device_buf.index_copy_(
            0,
            dst,
            host_buf.index_select(0, src).to(device_buf.device, non_blocking=True),
        )

    def writeback(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        block_quota: int | None = None,
    ) -> None:
        del block_quota
        if src_indices.numel() == 0:
            return
        for layer_buffers in self._buffers_by_layer:
            for device_buf, host_buf in layer_buffers:
                self._copy_device_to_host(
                    device_buf, host_buf, src_indices, dst_indices
                )

    def loadback(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None:
        if src_indices.numel() == 0:
            return
        for device_buf, host_buf in self._buffers_by_layer[int(layer_idx)]:
            self._copy_host_to_device(host_buf, device_buf, src_indices, dst_indices)

    def alloc_host(self, n: int):
        del n
        return None

    def free_host(self, indices: torch.Tensor) -> None:
        del indices

    def host_available(self) -> int:
        return 0
