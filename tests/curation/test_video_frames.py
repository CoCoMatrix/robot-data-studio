from __future__ import annotations

from types import SimpleNamespace

import av
import numpy as np
import pytest

from robot_data_studio.curation.sources import XJGT_DISPLAY_MATRIX_ROTATE_90
from robot_data_studio.curation.video_frames import (
    ExactPtsVideoFrameProvider,
    orient_encoded_frame,
)


def test_encoded_xjgt_frame_requires_frozen_display_matrix() -> None:
    encoded = np.zeros((6, 4, 3), dtype=np.uint8)
    encoded[0, 0] = [255, 0, 0]
    displayed = orient_encoded_frame(
        encoded,
        expected_width=6,
        expected_height=4,
        display_matrix=XJGT_DISPLAY_MATRIX_ROTATE_90,
    )
    assert displayed.shape == (4, 6, 3)
    np.testing.assert_array_equal(displayed, np.rot90(encoded, k=1))
    with pytest.raises(ValueError, match="no matching authority"):
        orient_encoded_frame(
            encoded, expected_width=6, expected_height=4, display_matrix=None
        )


def test_video_provider_uses_file_time_and_decoded_pts_tolerance(tmp_path) -> None:
    path = tmp_path / "train.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 48
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for index in range(4):
        image = np.full((32, 48, 3), index * 50, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    provider = ExactPtsVideoFrameProvider(
        expected_width=48,
        expected_height=32,
        tolerance_seconds=0.02,
    )
    frame = SimpleNamespace(
        media_references={
            "camera0": SimpleNamespace(uri=str(path), file_time=0.2)
        }
    )
    decoded = provider(frame)
    provider.close()
    assert decoded.shape == (32, 48, 3)
    assert float(decoded.mean()) > 80

    strict = ExactPtsVideoFrameProvider(
        expected_width=48,
        expected_height=32,
        tolerance_seconds=0.001,
    )
    frame.media_references["camera0"].file_time = 0.25
    with pytest.raises(ValueError, match="PTS tolerance"):
        strict(frame)
    strict.close()
