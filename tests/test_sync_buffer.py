from __future__ import annotations

import numpy as np

from src.shared.sync import FrameSignature, FrameSyncPacket, FrameSynchronizer


def _solid_frame(color: tuple[int, int, int], size: int = 16) -> np.ndarray:
    frame = np.zeros((size, size, 4), dtype=np.uint8)
    frame[:, :, 0] = color[0]
    frame[:, :, 1] = color[1]
    frame[:, :, 2] = color[2]
    frame[:, :, 3] = 255
    return frame


def test_frame_signature_matches_identical_frames() -> None:
    frame = _solid_frame((25, 50, 75))
    sig_a = FrameSignature.from_numpy(frame)
    sig_b = FrameSignature.from_numpy(frame.copy())
    assert sig_a.matches(sig_b)


def test_frame_signature_detects_distinct_frames() -> None:
    sig_a = FrameSignature.from_numpy(_solid_frame((0, 0, 0)))
    sig_b = FrameSignature.from_numpy(_solid_frame((255, 255, 255)))
    assert not sig_a.matches(sig_b)


def test_frame_synchronizer_aligns_streams() -> None:
    synchronizer = FrameSynchronizer(reference_stream="local", tolerance=0.05, buffer_duration=0.5)
    pairs = []
    for index in range(10):
        pts_local = index * 0.04
        pts_remote = pts_local + 0.012
        local_packet = FrameSyncPacket(
            stream_id="local",
            pts=pts_local,
            monotonic_ts=pts_local,
            frame_index=index,
            signature=None,
            payload={"time_pos": pts_local},
        )
        remote_packet = FrameSyncPacket(
            stream_id="server",
            pts=pts_remote,
            monotonic_ts=pts_remote,
            frame_index=index,
            signature=None,
            payload={"index": index},
        )
        pairs.extend(synchronizer.add_packet(local_packet))
        remote_pairs = synchronizer.add_packet(remote_packet)
        pairs.extend(remote_pairs)

    assert len(pairs) == 10
    for pair in pairs:
        reference, other = pair.ordered("local")
        assert reference.stream_id == "local"
        assert other.stream_id == "server"
        assert abs(pair.delta) <= synchronizer.tolerance + 1e-6
