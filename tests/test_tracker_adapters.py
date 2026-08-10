import numpy as np

from defense4uavswarm.trackers import ByteTrackLikeAdapter, SortAdapter


def test_bytetrack_adapter_interface() -> None:
    tracker = ByteTrackLikeAdapter()
    assert tracker.name == "bytetrack"
    assert tracker.update([1], 1) == [1]


def test_sort_adapter_confirms_track_without_reid() -> None:
    tracker = SortAdapter(max_age=2, min_hits=1, iou_threshold=0.3)
    det = np.asarray([[0, 0, 10, 10, 0.9, 2]], dtype=float)
    out1 = tracker.update(det, 1)
    out2 = tracker.update(np.asarray([[1, 0, 11, 10, 0.8, 2]], dtype=float), 2)
    assert tracker.name == "sort"
    assert out1.shape[1] == 7
    assert out2.shape[0] == 1
    assert int(out2[0, 4]) == int(out1[0, 4])


def test_sort_adapter_drops_stale_track() -> None:
    tracker = SortAdapter(max_age=1, min_hits=1, iou_threshold=0.3)
    tracker.update(np.asarray([[0, 0, 10, 10, 0.9, 2]], dtype=float), 1)
    tracker.update(np.empty((0, 6), dtype=float), 2)
    tracker.update(np.empty((0, 6), dtype=float), 3)
    assert len(tracker.trackers) == 0
