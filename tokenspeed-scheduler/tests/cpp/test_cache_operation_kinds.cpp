#include <gtest/gtest.h>

#include "scheduler/operations/cache.h"

namespace tokenspeed::test {

TEST(CacheOperationKindTest, FlatWriteBackBucketsTransfersByKind) {
    WriteBackOperation op;
    op.op_id = 7;
    op.transfers = {
        TransferPair{CacheKind::kKV, 1, 11},
        TransferPair{CacheKind::kMamba, 2, 22},
        TransferPair{CacheKind::kKV, 1, 11},
        TransferPair{CacheKind::kMamba, 3, 23},
    };

    FlatWriteBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({7}));
    EXPECT_EQ(flat.src_pages[0], std::vector<std::int32_t>({1}));
    EXPECT_EQ(flat.dst_pages[0], std::vector<std::int32_t>({11}));
    EXPECT_EQ(flat.src_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({1}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({11}));
    EXPECT_EQ(flat.src_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({2, 3}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({22, 23}));
}

TEST(CacheOperationKindTest, FlatWriteBackBucketsPagedCacheTransfersByGroup) {
    WriteBackOperation op0;
    op0.op_id = 7;
    op0.paged_cache_transfers = {
        PagedCacheTransferGroup{
            .group_id = "v4.swa_kv",
            .src_pages = {1, 2, 2},
            .dst_pages = {11, 12, 12},
        },
        PagedCacheTransferGroup{
            .group_id = "v4.c4a.compressed_kv",
            .src_pages = {3},
            .dst_pages = {13},
        },
    };
    WriteBackOperation op1;
    op1.op_id = 8;
    op1.paged_cache_transfers = {
        PagedCacheTransferGroup{
            .group_id = "v4.swa_kv",
            .src_pages = {4},
            .dst_pages = {14},
        },
    };

    FlatWriteBackOperation flat({op0, op1});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({7, 8}));
    EXPECT_EQ(flat.src_pages_by_paged_group.at("v4.swa_kv")[0], std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(flat.dst_pages_by_paged_group.at("v4.swa_kv")[0], std::vector<std::int32_t>({11, 12}));
    EXPECT_EQ(flat.src_pages_by_paged_group.at("v4.swa_kv")[1], std::vector<std::int32_t>({4}));
    EXPECT_EQ(flat.dst_pages_by_paged_group.at("v4.swa_kv")[1], std::vector<std::int32_t>({14}));
    EXPECT_EQ(flat.src_pages_by_paged_group.at("v4.c4a.compressed_kv")[0], std::vector<std::int32_t>({3}));
    EXPECT_EQ(flat.dst_pages_by_paged_group.at("v4.c4a.compressed_kv")[0], std::vector<std::int32_t>({13}));
    EXPECT_TRUE(flat.src_pages_by_paged_group.at("v4.c4a.compressed_kv")[1].empty());
    EXPECT_TRUE(flat.dst_pages_by_paged_group.at("v4.c4a.compressed_kv")[1].empty());
}

TEST(CacheOperationKindTest, FlatLoadBackBucketsTransfersByKind) {
    LoadBackOperation op;
    op.op_id = 9;
    op.transfers = {
        TransferPair{CacheKind::kKV, 10, 20},
        TransferPair{CacheKind::kMamba, 30, 40},
    };

    FlatLoadBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({9}));
    EXPECT_EQ(flat.src_pages[0], std::vector<std::int32_t>({10}));
    EXPECT_EQ(flat.dst_pages[0], std::vector<std::int32_t>({20}));
    EXPECT_EQ(flat.src_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({10}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({20}));
    EXPECT_EQ(flat.src_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({30}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({40}));
}

TEST(CacheOperationKindTest, FlatLoadBackBucketsPagedCacheTransfersByGroup) {
    LoadBackOperation op;
    op.op_id = 9;
    op.paged_cache_transfers = {
        PagedCacheTransferGroup{
            .group_id = "v4.swa_kv",
            .src_pages = {10, 11},
            .dst_pages = {20, 21},
        },
    };

    FlatLoadBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({9}));
    EXPECT_EQ(flat.src_pages_by_paged_group.at("v4.swa_kv")[0], std::vector<std::int32_t>({10, 11}));
    EXPECT_EQ(flat.dst_pages_by_paged_group.at("v4.swa_kv")[0], std::vector<std::int32_t>({20, 21}));
}

TEST(CacheOperationKindTest, PagedCacheTransferRejectsInvalidGroupShape) {
    WriteBackOperation op;
    op.op_id = 1;
    op.paged_cache_transfers = {
        PagedCacheTransferGroup{
            .group_id = "v4.swa_kv",
            .src_pages = {1},
            .dst_pages = {2, 3},
        },
    };

    EXPECT_THROW((FlatWriteBackOperation({op})), std::invalid_argument);
}

}  // namespace tokenspeed::test
