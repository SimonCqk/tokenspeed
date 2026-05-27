// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Coverage: HybridPrefixCache::Match paged-cache adjunct branch.

#include "paged_cache_test_fixture.h"

#include <unordered_set>

namespace tokenspeed::test {

using PagedCachePrefixMatchTest = PagedCacheLargeFixture;

// 320 tokens: no snapshot caps to root; snapshot at 256 caps to 256.
TEST_F(PagedCachePrefixMatchTest, CapVsNoCap320) {
    const std::int32_t num_pages = 320 / kPageSize;  // 5 pages
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    EXPECT_EQ(terminal->DepthInTokens(), 320u);

    const auto tokens = MakeAlignedTokens(num_pages, kPageSize, /*start=*/1);

    // No snapshot: paged_cache empty; device/host capped to root.
    auto match = MatchPrefix(*hybrid_, tokens, kPageSize).compat_match;
    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 0);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_TRUE(match.device.last_node->IsRoot())
        << "device terminal must be capped to root when adjunct is enabled but no snapshot exists";
    ASSERT_NE(match.host.last_node, nullptr);
    EXPECT_TRUE(match.host.last_node->IsRoot())
        << "host terminal must be capped to root when adjunct is enabled but no snapshot exists";

    // A complete paged-cache snapshot at depth 256 caps to 256.
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    EXPECT_EQ(boundary_256->DepthInTokens(), 256u);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, boundary_256, MakeCompleteSnapshot(256));
    ASSERT_TRUE(boundary_256->HasPagedCacheSnapshot());
    EXPECT_TRUE(boundary_256->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(boundary_256->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));

    match = MatchPrefix(*hybrid_, tokens, kPageSize).compat_match;
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, boundary_256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_EQ(match.device.last_node->DepthInTokens(), 256u)
        << "device terminal must be capped to the deepest contiguous paged-cache node";
}

TEST_F(PagedCachePrefixMatchTest, SnapshotCompletenessTracksDeviceAndHostTiersSeparately) {
    TreeNode* terminal = InsertDevicePages(/*num_pages=*/4, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    auto device_snapshot = MakeCompleteSnapshot(/*prefix_len_tokens=*/256);
    ASSERT_TRUE(
        HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, terminal, std::move(device_snapshot)));
    const PagedCacheSnapshot* attached_device = terminal->GetPagedCacheSnapshot();
    ASSERT_NE(attached_device, nullptr);
    EXPECT_TRUE(attached_device->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(attached_device->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_FALSE(attached_device->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(attached_device->IsCompleteOnHostFor(PagedCacheGroupFamily::State));
    auto detached_device = HybridPrefixCacheTestPeer::DetachPagedCacheSnapshotFromNode(*hybrid_, terminal);
    detached_device.reset();

    PageAllocator host_alloc{/*page_size=*/1, /*total_pages=*/8};
    auto host_snapshot = std::make_unique<PagedCacheSnapshot>();
    host_snapshot->prefix_len_tokens = 256;
    PagedCacheGroupSnapshot fh{};
    fh.host_pages = host_alloc.Allocate(1);
    fh.raw_token_cursor = 256;
    host_snapshot->groups.emplace("fh", std::move(fh));
    PagedCacheGroupSnapshot swa{};
    swa.host_pages = host_alloc.Allocate(1);
    swa.raw_token_cursor = 256;
    swa.sliding = true;
    host_snapshot->groups.emplace("swa", std::move(swa));

    ASSERT_TRUE(
        HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, terminal, std::move(host_snapshot)));
    const PagedCacheSnapshot* attached_host = terminal->GetPagedCacheSnapshot();
    ASSERT_NE(attached_host, nullptr);
    EXPECT_FALSE(attached_host->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(attached_host->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_TRUE(attached_host->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(attached_host->IsCompleteOnHostFor(PagedCacheGroupFamily::State));
    auto detached_host = HybridPrefixCacheTestPeer::DetachPagedCacheSnapshotFromNode(*hybrid_, terminal);
    detached_host.reset();
}

TEST_F(PagedCachePrefixMatchTest, PagedCacheHostWriteBackPublishesHostResidencyOnlyAfterAck) {
    PagedCacheGroupConfig host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    host_cfg.total_pages = 8;
    auto host_owner = std::make_unique<PagedCacheGroupAllocator>(std::move(host_cfg));
    PagedCacheGroupAllocator* host_alloc = host_owner.get();
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(*hybrid_, std::move(host_owner));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/5, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, boundary_256, MakeCompleteSnapshot(256));
    const PagedCacheSnapshot* snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    ASSERT_TRUE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));

    const std::vector<std::int32_t> device_ids = snapshot->groups.at("fh").DevicePageIds();
    ASSERT_FALSE(device_ids.empty());
    const std::int32_t device_available_before = fh_alloc_->AvailablePages();
    const std::int32_t host_available_before = host_alloc->AvailablePages();

    auto plan = HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1);

    ASSERT_TRUE(plan.ok);
    ASSERT_EQ(plan.nodes, std::vector<TreeNode*>({boundary_256}));
    ASSERT_EQ(plan.transfers.size(), 1u);
    const CacheStatsSnapshot writeback_stats = hybrid_->Stats({.paged_cache_group_ids = {"fh"}});
    EXPECT_EQ(writeback_stats.paged_cache_host_writeback_pages_scheduled_total.at("fh"),
              static_cast<std::int64_t>(device_ids.size()));
    EXPECT_EQ(writeback_stats.paged_cache_host_evicted_pages_total.at("fh"), 0);
    EXPECT_EQ(plan.transfers[0].group_id, "fh");
    EXPECT_EQ(plan.transfers[0].src_pages, device_ids);
    EXPECT_EQ(static_cast<std::int32_t>(plan.transfers[0].dst_pages.size()),
              static_cast<std::int32_t>(device_ids.size()));
    EXPECT_EQ(fh_alloc_->AvailablePages(), device_available_before)
        << "device pages must stay resident until writeback ack";
    EXPECT_EQ(host_alloc->AvailablePages(), host_available_before - static_cast<std::int32_t>(device_ids.size()));
    EXPECT_FALSE(snapshot->groups.at("fh").HasHost());

    auto duplicate_plan =
        HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1);
    EXPECT_FALSE(duplicate_plan.ok) << "pending writeback should not be planned twice";

    ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "fh"));
    EXPECT_EQ(fh_alloc_->AvailablePages(), device_available_before + static_cast<std::int32_t>(device_ids.size()));
    EXPECT_FALSE(snapshot->groups.at("fh").HasDevice());
    EXPECT_TRUE(snapshot->groups.at("fh").HasHost());
    EXPECT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    EXPECT_EQ(snapshot->groups.at("fh").HostPageIds(), plan.transfers[0].dst_pages);

    auto detached = HybridPrefixCacheTestPeer::DetachPagedCacheSnapshotFromNode(*hybrid_, boundary_256);
    detached.reset();
    EXPECT_EQ(host_alloc->AvailablePages(), host_available_before);
}

TEST_F(PagedCachePrefixMatchTest, PagedCacheHostWriteBackAckKeepsDeviceCopyWhilePinned) {
    PagedCacheGroupConfig host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    host_cfg.total_pages = 8;
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(
        *hybrid_, std::make_unique<PagedCacheGroupAllocator>(std::move(host_cfg)));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/5, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, boundary_256, MakeCompleteSnapshot(256));
    ASSERT_TRUE(boundary_256->HasPagedCacheSnapshot());

    const std::vector<std::int32_t> device_ids_before =
        boundary_256->GetPagedCacheSnapshot()->groups.at("fh").DevicePageIds();
    ASSERT_FALSE(device_ids_before.empty());

    auto plan = HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh",
                                                                          /*required_device_pages=*/1);
    ASSERT_TRUE(plan.ok);
    ASSERT_FALSE(plan.transfers.empty());
    ASSERT_FALSE(boundary_256->GetPagedCacheSnapshot()->groups.at("fh").HasHost());

    {
        DeviceNodeRef device_ref{boundary_256};
        ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "fh"));
        const PagedCacheSnapshot* snapshot = boundary_256->GetPagedCacheSnapshot();
        ASSERT_NE(snapshot, nullptr);
        const auto& group = snapshot->groups.at("fh");
        EXPECT_TRUE(group.HasHost());
        EXPECT_TRUE(group.HasDevice()) << "a D2H ack must not recycle pages borrowed after writeback scheduling";
        EXPECT_EQ(group.DevicePageIds(), device_ids_before);
        EXPECT_TRUE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
        EXPECT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    }

    const std::int32_t available_before_demote = fh_alloc_->AvailablePages();
    EXPECT_EQ(HybridPrefixCacheTestPeer::DemotePagedCacheDeviceCopiesPresentOnHost(*hybrid_, "fh",
                                                                                   /*required_device_pages=*/1),
              static_cast<std::int32_t>(device_ids_before.size()));
    const PagedCacheSnapshot* snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    const auto& group = snapshot->groups.at("fh");
    EXPECT_TRUE(group.HasHost());
    EXPECT_FALSE(group.HasDevice());
    EXPECT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    EXPECT_EQ(fh_alloc_->AvailablePages(),
              available_before_demote + static_cast<std::int32_t>(device_ids_before.size()));
}

TEST_F(PagedCachePrefixMatchTest, PagedCacheHostWriteBackEvictsOldHostCopyWhenHostPoolIsFull) {
    PagedCacheGroupConfig host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    // PageAllocator reserves page 0, so this leaves exactly one usable L2 page.
    host_cfg.total_pages = 2;
    auto host_owner = std::make_unique<PagedCacheGroupAllocator>(std::move(host_cfg));
    PagedCacheGroupAllocator* host_alloc = host_owner.get();
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(*hybrid_, std::move(host_owner));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/8, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* old_host_node = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* device_node = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    ASSERT_NE(old_host_node, nullptr);
    ASSERT_NE(device_node, nullptr);

    auto old_host_snapshot = std::make_unique<PagedCacheSnapshot>();
    old_host_snapshot->prefix_len_tokens = 256;
    PagedCacheGroupSnapshot old_host_group{};
    old_host_group.host_pages = host_alloc->AcquireOwned(1);
    ASSERT_FALSE(old_host_group.host_pages.Empty());
    old_host_group.raw_token_cursor = 256;
    old_host_snapshot->groups.emplace("fh", std::move(old_host_group));
    ASSERT_TRUE(HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, old_host_node,
                                                                          std::move(old_host_snapshot)));
    ASSERT_EQ(host_alloc->AvailablePages(), 0);

    ASSERT_TRUE(
        HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, device_node, MakeHistoryOnlySnapshot(512)));
    const auto device_ids = device_node->GetPagedCacheSnapshot()->groups.at("fh").DevicePageIds();

    auto plan = HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1);

    ASSERT_TRUE(plan.ok);
    ASSERT_EQ(plan.nodes, std::vector<TreeNode*>({device_node}));
    ASSERT_EQ(plan.transfers.size(), 1u);
    const CacheStatsSnapshot stats = hybrid_->Stats({.paged_cache_group_ids = {"fh"}});
    EXPECT_EQ(stats.paged_cache_host_writeback_pages_scheduled_total.at("fh"), 1);
    EXPECT_EQ(stats.paged_cache_host_evicted_pages_total.at("fh"), 1);
    EXPECT_EQ(plan.transfers[0].src_pages, device_ids);
    EXPECT_EQ(plan.transfers[0].dst_pages.size(), 1u);
    EXPECT_FALSE(old_host_node->HasPagedCacheSnapshot())
        << "host-only stale snapshot should be dropped to make L2 room";
    EXPECT_EQ(host_alloc->AvailablePages(), 0)
        << "the evicted L2 page should be immediately reused by the pending writeback";
}

TEST_F(PagedCachePrefixMatchTest, PagedCacheHostEvictionSkipsProtectedLoadbackSources) {
    PagedCacheGroupConfig host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    host_cfg.total_pages = 2;
    auto host_owner = std::make_unique<PagedCacheGroupAllocator>(std::move(host_cfg));
    PagedCacheGroupAllocator* host_alloc = host_owner.get();
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(*hybrid_, std::move(host_owner));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/8, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* protected_host_node = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* device_node = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    ASSERT_NE(protected_host_node, nullptr);
    ASSERT_NE(device_node, nullptr);

    auto protected_snapshot = std::make_unique<PagedCacheSnapshot>();
    protected_snapshot->prefix_len_tokens = 256;
    PagedCacheGroupSnapshot protected_group{};
    protected_group.host_pages = host_alloc->AcquireOwned(1);
    ASSERT_FALSE(protected_group.host_pages.Empty());
    protected_group.raw_token_cursor = 256;
    protected_snapshot->groups.emplace("fh", std::move(protected_group));
    ASSERT_TRUE(HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, protected_host_node,
                                                                          std::move(protected_snapshot)));
    ASSERT_EQ(host_alloc->AvailablePages(), 0);

    ASSERT_TRUE(
        HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, device_node, MakeHistoryOnlySnapshot(512)));
    const std::unordered_set<TreeNode*> protected_nodes{protected_host_node};

    auto plan = HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1,
                                                                          protected_nodes);

    EXPECT_FALSE(plan.ok);
    ASSERT_TRUE(protected_host_node->HasPagedCacheSnapshot());
    EXPECT_TRUE(protected_host_node->GetPagedCacheSnapshot()->groups.at("fh").HasHost());
    EXPECT_EQ(host_alloc->AvailablePages(), 0);
}

TEST_F(PagedCachePrefixMatchTest, HostPagedCacheHitMaterializesDevicePagesBeforeForwardMetadata) {
    PagedCacheGroupConfig fh_host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    fh_host_cfg.total_pages = 8;
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(
        *hybrid_, std::make_unique<PagedCacheGroupAllocator>(std::move(fh_host_cfg)));

    PagedCacheGroupConfig swa_host_cfg =
        MakeGroupConfig("swa", kLargeFixtureParams.swa_rows_per_page, kLargeFixtureParams.swa_stride,
                        PagedCacheGroupConfig::Retention::SlidingWindow, kSlidingWindow, PagedCacheGroupFamily::State);
    swa_host_cfg.total_pages = 8;
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(
        *hybrid_, std::make_unique<PagedCacheGroupAllocator>(std::move(swa_host_cfg)));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/5, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, boundary_256, MakeCompleteSnapshot(256));

    const auto fh_writeback =
        HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1);
    ASSERT_TRUE(fh_writeback.ok);
    ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "fh"));
    const auto swa_writeback =
        HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "swa", /*required_device_pages=*/1);
    ASSERT_TRUE(swa_writeback.ok);
    ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "swa"));

    const PagedCacheSnapshot* snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    ASSERT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    ASSERT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::State));
    ASSERT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
    ASSERT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::State));
    const std::vector<std::int32_t> fh_host_ids = snapshot->groups.at("fh").HostPageIds();
    const std::vector<std::int32_t> swa_host_ids = snapshot->groups.at("swa").HostPageIds();

    auto match =
        MatchPrefix(*hybrid_, MakeAlignedTokens(/*num_pages=*/5, kPageSize, /*start=*/1), kPageSize).compat_match;
    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    ASSERT_NE(match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(match.paged_cache_host.last_node, boundary_256);
    EXPECT_EQ(match.paged_cache_host.per_group_page_ids.at("fh"), fh_host_ids);
    EXPECT_EQ(match.paged_cache_host.per_group_page_ids.at("swa"), swa_host_ids);

    const std::int32_t fh_available_before_loadback = fh_alloc_->AvailablePages();
    const std::int32_t swa_available_before_loadback = swa_alloc_->AvailablePages();
    auto simulated_free = hybrid_->InitialSimulatedFree();
    auto loadback = HybridPrefixCacheTestPeer::PreparePagedCacheDeviceLoadBack(*hybrid_, match, simulated_free);

    ASSERT_TRUE(loadback.ok);
    ASSERT_EQ(loadback.loadback_transfers.size(), 2u);
    const CacheStatsSnapshot fh_stats = hybrid_->Stats({.paged_cache_group_ids = {"fh"}});
    const CacheStatsSnapshot swa_stats = hybrid_->Stats({.paged_cache_group_ids = {"swa"}});
    EXPECT_EQ(fh_stats.paged_cache_device_loadback_pages_scheduled_total.at("fh"),
              static_cast<std::int64_t>(fh_host_ids.size()));
    EXPECT_EQ(swa_stats.paged_cache_device_loadback_pages_scheduled_total.at("swa"),
              static_cast<std::int64_t>(swa_host_ids.size()));
    EXPECT_EQ(fh_stats.paged_cache_device_loadback_failed_count.at("fh"), 0);
    EXPECT_EQ(swa_stats.paged_cache_device_loadback_failed_count.at("swa"), 0);
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, boundary_256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
    snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    ASSERT_TRUE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
    ASSERT_TRUE(snapshot->IsCompleteFor(PagedCacheGroupFamily::State));
    ASSERT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    ASSERT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::State));
    EXPECT_EQ(match.paged_cache.per_group_page_ids.at("fh"), snapshot->groups.at("fh").DevicePageIds());
    EXPECT_EQ(match.paged_cache.per_group_page_ids.at("swa"), snapshot->groups.at("swa").DevicePageIds());
    for (const auto& transfer : loadback.loadback_transfers) {
        if (transfer.group_id == "fh") {
            EXPECT_EQ(transfer.src_pages, fh_host_ids);
            EXPECT_EQ(transfer.dst_pages, snapshot->groups.at("fh").DevicePageIds());
        }
        if (transfer.group_id == "swa") {
            EXPECT_EQ(transfer.src_pages, swa_host_ids);
            EXPECT_EQ(transfer.dst_pages, snapshot->groups.at("swa").DevicePageIds());
        }
    }

    HybridPrefixCacheTestPeer::CancelPagedCacheDeviceLoadBack(*hybrid_, loadback.loadback_nodes_by_group,
                                                              simulated_free);
    snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    EXPECT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(snapshot->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(snapshot->IsCompleteOnHostFor(PagedCacheGroupFamily::State));
    EXPECT_FALSE(snapshot->groups.at("fh").HasDevice());
    EXPECT_FALSE(snapshot->groups.at("swa").HasDevice());
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_available_before_loadback);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_available_before_loadback);
}

TEST_F(PagedCachePrefixMatchTest, PagedCacheSnapshotRefPinsHostPagesDuringLoadback) {
    PagedCacheGroupConfig fh_host_cfg =
        MakeGroupConfig("fh", kLargeFixtureParams.fh_rows_per_page, kLargeFixtureParams.fh_stride,
                        PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0, PagedCacheGroupFamily::History);
    fh_host_cfg.total_pages = 8;
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(
        *hybrid_, std::make_unique<PagedCacheGroupAllocator>(std::move(fh_host_cfg)));

    PagedCacheGroupConfig swa_host_cfg =
        MakeGroupConfig("swa", kLargeFixtureParams.swa_rows_per_page, kLargeFixtureParams.swa_stride,
                        PagedCacheGroupConfig::Retention::SlidingWindow, kSlidingWindow, PagedCacheGroupFamily::State);
    swa_host_cfg.total_pages = 8;
    HybridPrefixCacheTestPeer::RegisterPagedCacheHostGroup(
        *hybrid_, std::make_unique<PagedCacheGroupAllocator>(std::move(swa_host_cfg)));

    TreeNode* terminal = InsertDevicePages(/*num_pages=*/5, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, boundary_256, MakeCompleteSnapshot(256));

    const auto fh_writeback =
        HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "fh", /*required_device_pages=*/1);
    ASSERT_TRUE(fh_writeback.ok);
    ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "fh"));
    const auto swa_writeback =
        HybridPrefixCacheTestPeer::PreparePagedCacheHostWriteBack(*hybrid_, "swa", /*required_device_pages=*/1);
    ASSERT_TRUE(swa_writeback.ok);
    ASSERT_TRUE(HybridPrefixCacheTestPeer::OnPagedCacheHostWriteBackDone(*hybrid_, boundary_256, "swa"));

    const PagedCacheSnapshot* snapshot = boundary_256->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    const std::vector<std::int32_t> fh_host_ids = snapshot->groups.at("fh").HostPageIds();
    ASSERT_FALSE(fh_host_ids.empty());

    {
        auto match =
            MatchPrefix(*hybrid_, MakeAlignedTokens(/*num_pages=*/5, kPageSize, /*start=*/1), kPageSize).compat_match;
        auto simulated_free = hybrid_->InitialSimulatedFree();
        auto loadback = HybridPrefixCacheTestPeer::PreparePagedCacheDeviceLoadBack(*hybrid_, match, simulated_free);

        ASSERT_TRUE(loadback.ok);
        ASSERT_FALSE(loadback.loadback_transfers.empty());
        ASSERT_EQ(loadback.loadback_snapshot_refs.size(), 1u);
        EXPECT_EQ(HybridPrefixCacheTestPeer::EvictPagedCacheHostPagesForGroup(*hybrid_, "fh",
                                                                              /*required_host_pages=*/1),
                  0)
            << "active H2D loadback must pin its source host snapshot";
        ASSERT_TRUE(boundary_256->HasPagedCacheSnapshot());
        EXPECT_EQ(boundary_256->GetPagedCacheSnapshot()->groups.at("fh").HostPageIds(), fh_host_ids);
    }

    EXPECT_EQ(HybridPrefixCacheTestPeer::EvictPagedCacheHostPagesForGroup(*hybrid_, "fh",
                                                                          /*required_host_pages=*/1),
              static_cast<std::int32_t>(fh_host_ids.size()));
    ASSERT_TRUE(boundary_256->HasPagedCacheSnapshot());
    EXPECT_FALSE(boundary_256->GetPagedCacheSnapshot()->groups.at("fh").HasHost());
    EXPECT_TRUE(boundary_256->GetPagedCacheSnapshot()->groups.at("fh").HasDevice());
}

// Snapshots at 256/512/768; detaching 512 makes Match fall back to 256.
TEST_F(PagedCachePrefixMatchTest, ContiguousChainBreakMid) {
    const std::int32_t num_pages = 768 / kPageSize;  // 12 pages
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* n512 = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    TreeNode* n768 = kv_cache_->GetRadixTree().SplitAt(terminal, 768);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n512, nullptr);
    ASSERT_NE(n768, nullptr);

    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, n256, MakeCompleteSnapshot(256));
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, n512, MakeCompleteSnapshot(512));
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, n768, MakeCompleteSnapshot(768));

    // Drop the middle snapshot; chain scan must stop at the gap.
    auto dropped = HybridPrefixCacheTestPeer::DetachPagedCacheSnapshotFromNode(*hybrid_, n512);
    EXPECT_TRUE(dropped != nullptr);
    EXPECT_FALSE(n512->HasPagedCacheSnapshot());
    ASSERT_TRUE(n768->HasPagedCacheSnapshot());
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));

    auto match = MatchPrefix(*hybrid_, MakeAlignedTokens(num_pages, kPageSize, /*start=*/1), kPageSize).compat_match;
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, n256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
}

}  // namespace tokenspeed::test
