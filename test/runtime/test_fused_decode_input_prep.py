from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
cache_loc_kernel = pytest.importorskip(
    "tokenspeed.runtime.execution.cache_loc_kernel"
)


class _FakeKernelLauncher:
    def __init__(self) -> None:
        self.grid = None
        self.args = ()
        self.kwargs = {}

    def __getitem__(self, grid):
        self.grid = grid
        return self._launch

    def _launch(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def test_fused_decode_wrapper_forwards_group_keyed_dummy_mode(monkeypatch):
    launcher = _FakeKernelLauncher()
    monkeypatch.setattr(
        cache_loc_kernel,
        "fused_decode_input_prep_kernel",
        launcher,
    )
    req_pool_indices = SimpleNamespace(shape=(2,))
    req_to_pages = SimpleNamespace(shape=(3, 7))

    cache_loc_kernel.fused_decode_input_prep(
        object(),
        object(),
        object(),
        req_pool_indices,
        object(),
        4,
        req_to_pages,
        16,
        use_dummy_cache_loc=True,
        dummy_kv_slot=23,
    )

    assert launcher.grid == (2,)
    assert launcher.args[7] == 23
    assert launcher.kwargs["USE_DUMMY_CACHE_LOC"] is True
    assert launcher.kwargs["max_pages"] == 7

    cache_loc_kernel.fused_decode_input_prep(
        object(),
        object(),
        object(),
        req_pool_indices,
        object(),
        1,
        req_to_pages,
        16,
    )
    assert launcher.args[7] == 0
    assert launcher.kwargs["USE_DUMMY_CACHE_LOC"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused decode input preparation requires CUDA",
)
def test_fused_decode_input_prep_reuses_position_path_with_dummy_cache_locs():
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    valid_cache_lengths = torch.tensor([3, 5, 0], dtype=torch.int32, device="cuda")
    req_to_pages = torch.tensor(
        [
            [2, 3, 0, 0, 0, 0, 0],
            [4, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    cache_locs = torch.empty(2, dtype=torch.int32, device="cuda")
    positions = torch.empty(2, dtype=torch.int64, device="cuda")
    seq_lens = torch.empty(2, dtype=torch.int32, device="cuda")

    cache_loc_kernel.fused_decode_input_prep(
        cache_locs,
        positions,
        seq_lens,
        req_pool_indices,
        valid_cache_lengths,
        1,
        req_to_pages,
        4,
        use_dummy_cache_loc=True,
        dummy_kv_slot=23,
    )

    assert cache_locs.tolist() == [23, 23]
    assert positions.tolist() == [3, 5]
    assert seq_lens.tolist() == [4, 6]

    cache_loc_kernel.fused_decode_input_prep(
        cache_locs,
        positions,
        seq_lens,
        req_pool_indices,
        valid_cache_lengths,
        1,
        req_to_pages,
        4,
    )

    assert cache_locs.tolist() == [11, 21]
    assert positions.tolist() == [3, 5]
    assert seq_lens.tolist() == [4, 6]
