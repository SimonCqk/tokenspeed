from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _op(tables, bases=None):
    table_arrays = {
        group_id: (
            np.asarray(rows, dtype=np.int32)
            if len(rows) > 0
            else np.empty((0, 0), dtype=np.int32)
        )
        for group_id, rows in tables.items()
    }
    base_arrays = {
        group_id: np.asarray(rows, dtype=np.int32)
        for group_id, rows in (bases or {}).items()
    }
    return SimpleNamespace(
        flat_block_tables=tables,
        flat_block_tables_arrays=lambda: table_arrays,
        paged_cache_block_table_base_offsets_arrays=lambda: base_arrays,
    )


class FlatBlockTablesBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                flat_block_tables_from_forward_op,
                flat_cache_batch_from_forward_op,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise unittest.SkipTest(
                f"flat bridge unavailable (needs scheduler extension): {exc}"
            ) from exc
        cls.tables_bridge = staticmethod(flat_block_tables_from_forward_op)
        cls.batch_bridge = staticmethod(flat_cache_batch_from_forward_op)

    def test_legacy_table_bridge_preserves_values_and_packs_groups(self):
        out = self.tables_bridge(
            _op({"full": [[11, 12], [13, 0]], "swa": [[21], [0]]}),
            "cpu",
            num_reqs=2,
        )
        self.assertEqual(out["full"].tolist(), [[11, 12], [13, 0]])
        self.assertEqual(out["swa"].tolist(), [[21], [0]])
        self.assertEqual(
            out["full"].untyped_storage().data_ptr(),
            out["swa"].untyped_storage().data_ptr(),
        )

    def test_contract_batch_uses_canonical_paged_bases_in_one_storage(self):
        tables, bases = self.batch_bridge(
            _op(
                {"full": [[11], [12]], "swa": [[21, 22], [31, 0]]},
                {"swa": [7, 19]},
            ),
            "cpu",
            num_reqs=2,
            expected_group_ids=("full", "swa"),
            max_page_ids={"full": 32, "swa": 32},
            required_base_offset_group_ids=frozenset({"swa"}),
        )
        self.assertEqual(bases["full"].tolist(), [0, 0])
        self.assertEqual(bases["swa"].tolist(), [7, 19])
        pointers = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*tables.values(), *bases.values())
        }
        self.assertEqual(len(pointers), 1)

    def test_invalid_compact_base_contract_fails_closed(self):
        cases = (
            (
                "missing sliding base",
                _op({"swa": [[1], [2]]}),
                frozenset({"swa"}),
                r"missing logical base offsets.*swa",
            ),
            (
                "negative base",
                _op({"swa": [[1]]}, {"swa": [-1]}),
                frozenset({"swa"}),
                "negative logical base",
            ),
            (
                "row mismatch",
                _op({"swa": [[1], [2]]}, {"swa": [3]}),
                frozenset({"swa"}),
                r"swa.*1 rows",
            ),
        )
        for name, op, required, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                self.batch_bridge(
                    op,
                    "cpu",
                    num_reqs=len(op.flat_block_tables["swa"]),
                    expected_group_ids=("swa",),
                    max_page_ids={"swa": 8},
                    required_base_offset_group_ids=required,
                )

    def test_table_row_shape_contract(self):
        failure_cases = (
            ("row count mismatch", {"full": [[1, 2]]}, r"full.*1 rows.*num_reqs=2"),
            (
                "empty group on live batch",
                {"full": [[1, 2], [3, 4]], "swa": []},
                r"swa.*0 rows.*num_reqs=2",
            ),
        )
        for name, tables, error in failure_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                self.tables_bridge(_op(tables), "cpu", num_reqs=2)

        empty_op = _op({"full": [], "swa": []})
        for num_reqs in (0, None):
            with self.subTest(name="empty operation", num_reqs=num_reqs):
                self.assertEqual(
                    self.tables_bridge(empty_op, "cpu", num_reqs=num_reqs),
                    {},
                )

    def test_radix_operation_without_flat_export_remains_empty(self):
        self.assertEqual(self.tables_bridge(SimpleNamespace(), "cpu"), {})


class FlatCacheBatchMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenspeed.runtime.configs.flat_cache_runtime import (
                FlatPagedCacheRuntimeContract,
            )
            from tokenspeed.runtime.configs.paged_cache_spec import (
                PagedCacheGroupSpec,
            )
            from tokenspeed.runtime.execution.cuda_graph_wrapper import (
                CudaGraphWrapper,
            )
            from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
            from tokenspeed.runtime.layers.attention.backends.flat_cache_metadata import (
                FlatCacheBatchMetadata,
                resolve_flat_runtime_contracts,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise unittest.SkipTest(f"flat metadata unavailable: {exc}") from exc
        cls.Contract = FlatPagedCacheRuntimeContract
        cls.Spec = PagedCacheGroupSpec
        cls.Metadata = FlatCacheBatchMetadata
        cls.resolve_contracts = staticmethod(resolve_flat_runtime_contracts)
        cls.Wrapper = CudaGraphWrapper
        cls.ForwardMode = ForwardMode

    def _specs(self):
        return (
            self.Spec(
                group_id="full",
                retention="full_history",
                rows_per_page=16,
                entry_stride_tokens=4,
                sliding_window_tokens=None,
                block_size=64,
            ),
            self.Spec(
                group_id="swa",
                retention="sliding_window",
                rows_per_page=4,
                entry_stride_tokens=1,
                sliding_window_tokens=8,
                family="state",
                block_size=4,
                cache_blocks_per_lcm_block=2,
            ),
        )

    def _contract(
        self,
        *,
        specs=None,
        block_size=4,
        num_lcm_blocks=4,
        token_capacity=256,
    ):
        specs = tuple(specs or self._specs())
        return self.Contract(
            block_size=block_size,
            num_lcm_blocks=num_lcm_blocks,
            token_capacity=token_capacity,
            group_specs=specs,
            group_page_counts={
                spec.group_id: (num_lcm_blocks * spec.cache_blocks_per_lcm_block + 1)
                for spec in specs
            },
        )

    def test_heterogeneous_spans_projection_and_forward_provenance(self):
        op = _op(
            {"full": [[1], [2]], "swa": [[3, 4], [5, 0]]},
            {"full": [0, 0], "swa": [7, 9]},
        )
        metadata = self.Metadata.from_forward_op(
            op,
            device="cpu",
            contract=self._contract(),
            num_requests=2,
            compact_tables=True,
        )
        owner = metadata.for_groups(("swa",), owner="draft")
        self.assertIs(
            owner.require_table("swa", active_forward_op=op),
            metadata.require_table("swa", active_forward_op=op),
        )
        self.assertIs(
            owner.require_base_offsets("swa", active_forward_op=op),
            metadata.require_base_offsets("swa", active_forward_op=op),
        )
        with self.assertRaisesRegex(RuntimeError, "stale flat cache metadata"):
            owner.tables(active_forward_op=SimpleNamespace())

    def test_runtime_contract_selects_transport_and_capability_selects_compaction(
        self,
    ):
        target_contract = self._contract()
        draft_contract = self._contract(specs=(self._specs()[0],))
        target_pool = SimpleNamespace(runtime_contract=target_contract)
        draft_pool = SimpleNamespace(runtime_contract=draft_contract)
        capable = SimpleNamespace(supports_compact_flat_block_tables=True)
        legacy = SimpleNamespace(supports_compact_flat_block_tables=False)

        for name, kwargs, expected in (
            (
                "legacy consumer keeps contract metadata with absolute tables",
                dict(target_pool=target_pool, target_backend=legacy),
                (target_contract, None, False),
            ),
            (
                "compact target",
                dict(target_pool=target_pool, target_backend=capable),
                (target_contract, None, True),
            ),
            (
                "compact target and draft",
                dict(
                    target_pool=target_pool,
                    target_backend=capable,
                    draft_pool=draft_pool,
                    draft_backend=capable,
                ),
                (target_contract, draft_contract, True),
            ),
            (
                "radix build keeps metadata but disables compaction",
                dict(target_pool=target_pool, target_backend=capable),
                (target_contract, None, False),
            ),
        ):
            with self.subTest(name=name):
                actual = self.resolve_contracts(
                    **kwargs,
                    flat_kvcache_ext=name
                    != "radix build keeps metadata but disables compaction",
                )
                self.assertIs(actual[0], expected[0])
                self.assertIs(actual[1], expected[1])
                self.assertIs(actual[2], expected[2])

    def test_compact_capability_contract_errors_fail_closed(self):
        contract = self._contract()
        contract_pool = SimpleNamespace(runtime_contract=contract)
        no_contract_pool = SimpleNamespace(runtime_contract=None)
        capable = SimpleNamespace(supports_compact_flat_block_tables=True)
        legacy = SimpleNamespace(supports_compact_flat_block_tables=False)

        cases = (
            (
                "target compact draft legacy",
                dict(
                    target_pool=contract_pool,
                    target_backend=capable,
                    draft_pool=SimpleNamespace(
                        runtime_contract=self._contract(specs=(self._specs()[0],))
                    ),
                    draft_backend=legacy,
                ),
                "disagree on compact",
            ),
            (
                "target legacy draft compact",
                dict(
                    target_pool=contract_pool,
                    target_backend=legacy,
                    draft_pool=SimpleNamespace(
                        runtime_contract=self._contract(specs=(self._specs()[0],))
                    ),
                    draft_backend=capable,
                ),
                "disagree on compact",
            ),
            (
                "compact target has no contract",
                dict(
                    target_pool=no_contract_pool,
                    target_backend=capable,
                ),
                "target backend requires a runtime contract",
            ),
            (
                "compact draft has no contract",
                dict(
                    target_pool=contract_pool,
                    target_backend=capable,
                    draft_pool=no_contract_pool,
                    draft_backend=capable,
                ),
                "draft backend requires a runtime contract",
            ),
        )
        for name, kwargs, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, error):
                self.resolve_contracts(
                    **kwargs,
                    flat_kvcache_ext=True,
                )

    def test_target_draft_contract_geometry_must_match(self):
        target = self._contract()
        full = self._specs()[0]
        cases = (
            (
                "base block size",
                self._contract(
                    specs=(full,),
                    block_size=8,
                ),
                "base block_size",
            ),
            (
                "draft group subset",
                self._contract(specs=(replace(full, group_id="other"),)),
                "not a subset",
            ),
            (
                "group block size",
                self._contract(specs=(replace(full, block_size=68),)),
                "group_block_size",
            ),
            (
                "retention",
                self._contract(
                    specs=(
                        replace(
                            full,
                            retention="sliding_window",
                            sliding_window_tokens=64,
                        ),
                    )
                ),
                "retention",
            ),
            (
                "family",
                self._contract(specs=(replace(full, family="state"),)),
                "family",
            ),
            (
                "cache blocks per LCM block",
                self._contract(specs=(replace(full, cache_blocks_per_lcm_block=2),)),
                "cache_blocks_per_lcm_block",
            ),
            (
                "group page count",
                self._contract(
                    specs=(full,),
                    num_lcm_blocks=3,
                    token_capacity=192,
                ),
                "group_page_count",
            ),
        )
        capable = SimpleNamespace(supports_compact_flat_block_tables=True)
        for name, draft, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, error):
                self.resolve_contracts(
                    target_pool=SimpleNamespace(runtime_contract=target),
                    target_backend=capable,
                    draft_pool=SimpleNamespace(runtime_contract=draft),
                    draft_backend=capable,
                    flat_kvcache_ext=True,
                )

    def test_absolute_contract_metadata_does_not_read_or_transfer_bases(self):
        op = _op({"full": [[1], [2]], "swa": [[3, 4], [5, 0]]})

        def fail_if_bases_are_read():
            raise AssertionError("absolute table transport must not read bases")

        op.paged_cache_block_table_base_offsets_arrays = fail_if_bases_are_read
        metadata = self.Metadata.from_forward_op(
            op,
            device="cpu",
            contract=self._contract(),
            num_requests=2,
            compact_tables=False,
        )
        self.assertFalse(metadata.compact_tables)
        self.assertEqual(metadata.base_offsets(active_forward_op=op), {})
        self.assertEqual(
            set(metadata.tables(active_forward_op=op)),
            {"full", "swa"},
        )

    def test_compact_contract_metadata_requires_every_group_base(self):
        op = _op(
            {"full": [[1], [2]], "swa": [[3, 4], [5, 0]]},
            {"swa": [7, 9]},
        )
        with self.assertRaisesRegex(
            ValueError,
            r"missing logical base offsets.*full",
        ):
            self.Metadata.from_forward_op(
                op,
                device="cpu",
                contract=self._contract(),
                num_requests=2,
                compact_tables=True,
            )

    def test_graph_replay_keeps_target_and_draft_owner_views_distinct(self):
        import torch

        op = _op(
            {"full": [[1], [2]], "swa": [[3, 4], [5, 0]]},
            {"full": [0, 0], "swa": [7, 9]},
        )
        target = self.Metadata.from_forward_op(
            op,
            device="cpu",
            contract=self._contract(),
            num_requests=2,
            compact_tables=True,
        )
        draft = target.for_groups(("full",), owner="draft")
        captured = {}

        class Backend:
            uses_paged_cache_groups = True
            uses_flat_cache_groups = True
            uses_padded_decode_token_mask = False
            flat_tables_self_padding = True

            def __init__(self, owner):
                self.owner = owner

            def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
                captured[self.owner] = kwargs

        wrapper = self.Wrapper.__new__(self.Wrapper)
        wrapper.attn_backend = Backend("target")
        wrapper.draft_attn_backend = Backend("draft")
        wrapper.drafter = SimpleNamespace(
            draft_seq_lens_buf=torch.zeros(2, dtype=torch.int32),
            req_to_page=torch.zeros((2, 1), dtype=torch.int32),
        )
        wrapper.max_tokens_per_req = 1
        wrapper._init_replay_metadata(
            padded_bs=2,
            actual_bs=2,
            req_pool_indices=torch.arange(2, dtype=torch.int32),
            seq_lens=torch.ones(2, dtype=torch.int32),
            req_to_page=torch.zeros((2, 1), dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables=target.tables(active_forward_op=op),
            paged_cache_block_table_base_offsets=target.base_offsets(
                active_forward_op=op
            ),
            flat_cache_metadata=target,
            draft_flat_cache_metadata=draft,
            flat_cache_forward_op=op,
        )

        self.assertEqual(set(captured["target"]["flat_block_tables"]), {"full", "swa"})
        self.assertEqual(set(captured["draft"]["flat_block_tables"]), {"full"})
        self.assertEqual(
            set(captured["draft"]["paged_cache_block_table_base_offsets"]),
            {"full"},
        )


if __name__ == "__main__":
    unittest.main()
