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

"""Top-level memory executor that coordinates host and storage executors."""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

try:
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPool as MHATokenToKVPoolPaged,
    )
except ImportError:
    MHATokenToKVPoolPaged = None
try:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import (
        MLATokenToKVPool as MLATokenToKVPoolPaged,
    )
except (ImportError, AttributeError):
    MLATokenToKVPoolPaged = None
try:
    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
        DeepseekV4TokenToKVPool,
    )
except (ImportError, AttributeError):
    DeepseekV4TokenToKVPool = None

from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.executor.host_executor import HostExecutor
from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor
from tokenspeed.runtime.cache.kv_cache_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from tokenspeed.runtime.cache.mamba_cache_host import MambaPoolHost
from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4PagedCachePool
from tokenspeed.runtime.cache.transfer.kv_pool import KVCachePool
from tokenspeed.runtime.cache.transfer.mamba_pool import MambaCachePool
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass(slots=True)
class MemoryExecutorConfig:
    layer_num: int
    page_size: int = 64
    host_ratio: float = 2.0
    host_size_gb: int = 0
    io_backend: str = "kernel"
    host_layout: str = "layer_first"
    storage_backend: Optional[str] = "mooncake"
    storage_backend_extra_config: Optional[str] = None
    model_name: Optional[str] = None
    enable_mamba_l2: bool = False
    mamba_l2_host_slots: int = 0
    mamba_l2_layout: str = "layer_first"
    mamba_l2_io_backend: str = "kernel"


class MemoryExecutor:
    """Coordinate host-memory and storage-backed cache operations."""

    def __init__(
        self,
        device_pool,
        config: MemoryExecutorConfig,
        is_dp_attention_enabled: bool,
        tp_group=None,
        draft_device_pool=None,
        mamba_pool=None,
    ):
        self.page_size = config.page_size

        _mha_types = (MHATokenToKVPool,)
        if MHATokenToKVPoolPaged is not None:
            _mha_types = (MHATokenToKVPool, MHATokenToKVPoolPaged)

        _mla_types = (MLATokenToKVPool,)
        if MLATokenToKVPoolPaged is not None:
            _mla_types = (MLATokenToKVPool, MLATokenToKVPoolPaged)
        _v4_types = (
            (DeepseekV4TokenToKVPool,) if DeepseekV4TokenToKVPool is not None else ()
        )

        # Unwrap LayerMappedKVPool (hybrid GDN models) to get the inner MHA pool.
        actual_pool = device_pool
        if hasattr(device_pool, "inner") and not isinstance(
            device_pool, (*_mha_types, *_mla_types, *_v4_types)
        ):
            actual_pool = device_pool.inner

        v4_pools = None
        is_deepseek_v4_pool = isinstance(actual_pool, _v4_types)
        if is_deepseek_v4_pool:
            host_counts = self._deepseek_v4_host_page_counts(actual_pool, config)
            actual_pool.paged_cache_group_host_page_counts = host_counts
            v4_pools = [
                DeepseekV4PagedCachePool(
                    device_pool=actual_pool,
                    group_id=str(spec.group_id),
                    host_total_pages=host_counts[str(spec.group_id)],
                    io_backend=config.io_backend,
                )
                for spec in actual_pool.paged_cache_group_specs
                if host_counts.get(str(spec.group_id), 0) > 0
            ]
            self.host_pool = None
        elif isinstance(actual_pool, _mha_types):
            self.host_pool = MHATokenToKVPoolHost(
                actual_pool,
                config.host_ratio,
                config.host_size_gb,
                config.page_size,
                config.host_layout,
            )
        elif isinstance(actual_pool, _mla_types):
            self.host_pool = MLATokenToKVPoolHost(
                actual_pool,
                config.host_ratio,
                config.host_size_gb,
                config.page_size,
                config.host_layout,
            )
        else:
            raise ValueError(
                f"host_pool only supports MHA, MLA, and DeepSeek V4, got {type(actual_pool)} "
                f"from module {type(actual_pool).__module__}"
            )

        # Draft model L2 cache: draft shares the same page mapping as the base
        # model, so its host pool must hold exactly the same number of tokens.
        # Pass host_size_tokens directly to bypass ratio/GB recalculation.
        if draft_device_pool is not None and not is_deepseek_v4_pool:
            actual_draft_pool = draft_device_pool
            if hasattr(draft_device_pool, "inner") and not isinstance(
                draft_device_pool, (*_mha_types, *_mla_types)
            ):
                actual_draft_pool = draft_device_pool.inner
            if isinstance(actual_draft_pool, _mha_types):
                self.draft_host_pool = MHATokenToKVPoolHost(
                    actual_draft_pool,
                    config.host_ratio,
                    config.host_size_gb,
                    config.page_size,
                    config.host_layout,
                    host_size_tokens=self.host_pool.size,
                )
            elif isinstance(actual_draft_pool, _mla_types):
                self.draft_host_pool = MLATokenToKVPoolHost(
                    actual_draft_pool,
                    config.host_ratio,
                    config.host_size_gb,
                    config.page_size,
                    config.host_layout,
                    host_size_tokens=self.host_pool.size,
                )
            else:
                raise ValueError(
                    f"draft_device_pool only supports MHA and MLA, "
                    f"got {type(actual_draft_pool)}"
                )
            draft_host_bytes = (
                self.draft_host_pool.size * self.draft_host_pool.size_per_token
            )
            logger.info(
                "Allocating %.2f GB host memory for draft model L2 cache (pool_type=%s size_tokens=%s size_per_token=%s layer_num=%s)",
                draft_host_bytes / 1e9,
                type(self.draft_host_pool).__name__,
                self.draft_host_pool.size,
                self.draft_host_pool.size_per_token,
                actual_draft_pool.layer_num,
            )
            draft_layer_num = actual_draft_pool.layer_num
        else:
            self.draft_host_pool = None
            draft_layer_num = 0

        pools = v4_pools
        self.mamba_host_pool = None
        if (
            config.enable_mamba_l2
            and mamba_pool is not None
            and config.mamba_l2_host_slots > 0
        ):
            self.mamba_host_pool = MambaPoolHost(
                mamba_pool,
                host_size_slots=config.mamba_l2_host_slots,
                layout=config.mamba_l2_layout,
            )
            mamba_transfer_pool = MambaCachePool(
                device_pool=mamba_pool,
                host_pool=self.mamba_host_pool,
                io_backend=config.mamba_l2_io_backend,
            )
            if pools is None:
                pools = [
                    KVCachePool(
                        device_pool=device_pool,
                        host_pool=self.host_pool,
                        io_backend=config.io_backend,
                        layer_num=actual_pool.layer_num,
                        draft_device_pool=(
                            actual_draft_pool if draft_device_pool is not None else None
                        ),
                        draft_host_pool=self.draft_host_pool,
                        draft_layer_num=draft_layer_num,
                    ),
                    mamba_transfer_pool,
                ]
            else:
                pools = [*pools, mamba_transfer_pool]
            logger.debug(
                "[cache_op] MemoryExecutor init pools=%s host_pools=%s draft=%s mamba=%s io_backend=%s host_layout=%s",
                [str(pool.kind) for pool in pools],
                [type(self.host_pool).__name__, type(self.mamba_host_pool).__name__],
                self.draft_host_pool is not None,
                True,
                config.io_backend,
                config.host_layout,
            )

        if pools is not None:
            self.host_exec = HostExecutor(pools=pools, io_backend=config.io_backend)
        else:
            self.host_exec = HostExecutor(
                page_size=config.page_size,
                device_pool=device_pool,
                host_pool=self.host_pool,
                io_backend=config.io_backend,
                layer_num=actual_pool.layer_num,
                draft_device_pool=(
                    actual_draft_pool if draft_device_pool is not None else None
                ),
                draft_host_pool=self.draft_host_pool,
                draft_layer_num=draft_layer_num,
            )
        storage_backend = None if is_deepseek_v4_pool else config.storage_backend
        self.storage_exec = StorageExecutor(
            page_size=config.page_size,
            device_pool=device_pool,
            host_pool=self.host_pool,
            storage_backend_type=storage_backend,
            storage_backend_extra_config=config.storage_backend_extra_config,
            model_name=config.model_name,
            is_dp_attention_enabled=is_dp_attention_enabled,
            tp_group=tp_group,
        )

    @staticmethod
    def _deepseek_v4_host_page_counts(
        device_pool, config: MemoryExecutorConfig
    ) -> dict[str, int]:
        ratio = max(float(config.host_ratio), 0.0)
        counts: dict[str, int] = {}
        for spec in device_pool.paged_cache_group_specs:
            group_id = str(spec.group_id)
            device_pages = int(device_pool.paged_cache_group_page_counts[group_id])
            if ratio <= 0.0 or device_pages <= 1:
                counts[group_id] = 0
                continue
            device_usable_pages = device_pages - 1
            host_usable_pages = max(int(device_usable_pages * ratio), 1)
            counts[group_id] = host_usable_pages + 1
        return counts

    @staticmethod
    def _page_groups_by_kind(op) -> dict[CacheKind, tuple[list, list]]:
        src_by_kind = getattr(op, "src_pages_by_kind", None)
        dst_by_kind = getattr(op, "dst_pages_by_kind", None)
        if src_by_kind is None or dst_by_kind is None:
            return {CacheKind.KV: (op.src_pages, op.dst_pages)}
        groups: dict[CacheKind, tuple[list, list]] = {}
        for kind in CacheKind:
            src_pages = src_by_kind.get(kind.value, [])
            dst_pages = dst_by_kind.get(kind.value, [])
            groups[kind] = (src_pages, dst_pages)
        return groups

    @staticmethod
    def _page_groups_by_paged_group(op) -> dict[str, tuple[list, list]]:
        src_by_group = getattr(op, "src_pages_by_paged_group", None) or {}
        dst_by_group = getattr(op, "dst_pages_by_paged_group", None) or {}
        groups: dict[str, tuple[list, list]] = {}
        for group_id, src_pages in src_by_group.items():
            groups[str(group_id)] = (src_pages, dst_by_group.get(group_id, []))
        return groups

    def submit_plan(self, plan, producer_stream=None) -> None:
        if plan.cache:
            logger.debug("[cache_op] submit_plan: %s cache ops", len(plan.cache))
        self.host_exec.fence_writeback_after(producer_stream)
        for op in plan.cache:
            self.submit(op)
        self.host_exec.flush()

    def submit(self, op) -> None:
        if isinstance(op, Cache.WriteBackOp):
            logger.debug(
                "[cache_op] writeback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            groups = self._page_groups_by_kind(op)
            paged_groups = self._page_groups_by_paged_group(op)
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                is_retract = bool(getattr(op, "is_retract", [False])[i])
                scheduled_any = False
                for kind, (src_groups, dst_groups) in groups.items():
                    if kind not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    if kind == CacheKind.MAMBA:
                        logger.debug(
                            "[cache_op][mamba_l2] writeback schedule "
                            "op_id=%s slots=%s device_slots=%s host_slots=%s "
                            "is_retract=%s",
                            op_id,
                            len(src_pages),
                            src_pages[:8],
                            dst_pages[:8],
                            is_retract,
                        )
                    self.host_exec.enqueue_writeback(
                        op_id,
                        src_pages,
                        dst_pages,
                        is_retract=is_retract,
                        kind=kind,
                    )
                    scheduled_any = True
                for group_id, (src_groups, dst_groups) in paged_groups.items():
                    if group_id not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    logger.debug(
                        "[cache_op][paged_l2] writeback schedule "
                        "op_id=%s group=%s pages=%s device_pages=%s host_pages=%s",
                        op_id,
                        group_id,
                        len(src_pages),
                        src_pages[:8],
                        dst_pages[:8],
                    )
                    self.host_exec.enqueue_writeback(
                        op_id,
                        src_pages,
                        dst_pages,
                        is_retract=is_retract,
                        kind=group_id,
                    )
                    scheduled_any = True
                if not scheduled_any:
                    self.host_exec.completed_writebacks.append(op_id)
        elif isinstance(op, Cache.LoadBackOp):
            logger.debug(
                "[cache_op] loadback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            groups = self._page_groups_by_kind(op)
            paged_groups = self._page_groups_by_paged_group(op)
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                scheduled_any = False
                for kind, (src_groups, dst_groups) in groups.items():
                    if kind not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    if kind == CacheKind.MAMBA:
                        logger.debug(
                            "[cache_op][mamba_l2] loadback schedule "
                            "op_id=%s slots=%s host_slots=%s device_slots=%s",
                            op_id,
                            len(src_pages),
                            src_pages[:8],
                            dst_pages[:8],
                        )
                    self.host_exec.enqueue_loadback(
                        op_id, src_pages, dst_pages, kind=kind
                    )
                    scheduled_any = True
                for group_id, (src_groups, dst_groups) in paged_groups.items():
                    if group_id not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    logger.debug(
                        "[cache_op][paged_l2] loadback schedule "
                        "op_id=%s group=%s pages=%s host_pages=%s device_pages=%s",
                        op_id,
                        group_id,
                        len(src_pages),
                        src_pages[:8],
                        dst_pages[:8],
                    )
                    self.host_exec.enqueue_loadback(
                        op_id, src_pages, dst_pages, kind=group_id
                    )
                    scheduled_any = True
                if not scheduled_any:
                    self.host_exec.completed_loadbacks.append(op_id)

        elif isinstance(op, Cache.PrefetchOp):
            logger.debug(
                "[cache_op] prefetch op_id=%s dst_pages=%s", op.op_id, len(op.dst_pages)
            )
            self.storage_exec.submit_prefetch(op)
        elif isinstance(op, Cache.BackUpOp):
            logger.debug(
                "[cache_op] backup op_id=%s src_pages=%s", op.op_id, len(op.src_pages)
            )
            self.storage_exec.submit_backup(op)
        else:
            raise ValueError("unsupported cache op kind")

    def poll_results(self) -> list:
        results: list = []
        results.extend(self.host_exec.drain())
        results.extend(self.storage_exec.drain())
        if results:
            for r in results:
                logger.debug(
                    "[cache_op] done op_id=%s success=%s type=%s",
                    r.op_id,
                    r.success,
                    type(r).__name__,
                )
        return results

    def get_producer_index(
        self, kind_or_op_id: CacheKind | str | int, op_id: int | None = None
    ) -> Optional[int]:
        return self.host_exec.get_producer_index(kind_or_op_id, op_id)

    def set_consumer(
        self,
        kind_or_producer_index: CacheKind | str | int | Iterable[int],
        producer_index: int | Iterable[int] | None = None,
    ) -> None:
        self.host_exec.set_consumer(kind_or_producer_index, producer_index)

    def query_l3_pages(self, hashes: list[str]) -> int:
        return self.storage_exec.query_exists(hashes)

    def shutdown(self) -> None:
        self.host_exec.shutdown()
        self.storage_exec.shutdown()

    def reset(self) -> None:
        self.host_exec.reset()
        self.storage_exec.drain()
