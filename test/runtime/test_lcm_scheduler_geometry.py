from types import SimpleNamespace

from tokenspeed.runtime.engine.scheduler_utils import scheduler_cache_geometry_from_pool


def test_radix_scheduler_geometry_uses_physical_page_units():
    geometry = scheduler_cache_geometry_from_pool(
        SimpleNamespace(runtime_contract=None),
        fallback_token_capacity=16_384,
        fallback_page_size=256,
    )

    assert geometry.page_size == 256
    assert geometry.num_device_pages == 64
    assert geometry.num_usable_pages == 64
    assert geometry.token_capacity == 16_384


def test_lcm_scheduler_geometry_counts_parent_blocks_and_reserves_null():
    contract = SimpleNamespace(
        num_lcm_blocks=37,
        block_size=128,
        token_capacity=10_000,
    )

    geometry = scheduler_cache_geometry_from_pool(
        SimpleNamespace(runtime_contract=contract),
        fallback_token_capacity=37 * 12 * 128,
        fallback_page_size=256,
    )

    assert geometry.page_size == 128
    assert geometry.num_device_pages == 38
    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 10_000
