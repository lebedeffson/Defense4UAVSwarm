from defense4uavswarm.trackers import ByteTrackLikeAdapter


def test_bytetrack_adapter_interface() -> None:
    tracker = ByteTrackLikeAdapter()
    assert tracker.name == "bytetrack"
    assert tracker.update([1], 1) == [1]
