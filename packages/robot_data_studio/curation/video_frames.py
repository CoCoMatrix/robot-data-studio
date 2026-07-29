from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import av
import numpy as np

from .sources import XJGT_DISPLAY_MATRIX_ROTATE_90


def orient_encoded_frame(
    image: np.ndarray,
    *,
    expected_width: int,
    expected_height: int,
    display_matrix: Sequence[int] | None,
) -> np.ndarray:
    """Apply only the frozen container display authority; never guess orientation."""

    frame = np.asarray(image)
    if frame.shape[:2] == (expected_height, expected_width):
        return frame
    if tuple(display_matrix or ()) == XJGT_DISPLAY_MATRIX_ROTATE_90:
        rotated = np.rot90(frame, k=1)
        if rotated.shape[:2] == (expected_height, expected_width):
            return np.ascontiguousarray(rotated)
    raise ValueError(
        f"encoded/display shape mismatch: got {frame.shape[:2]}, "
        f"expected {(expected_height, expected_width)} with no matching authority"
    )


@dataclass
class _VideoCursor:
    container: Any
    stream: Any
    iterator: Any | None = None
    last_pts: float | None = None


class ExactPtsVideoFrameProvider:
    """Stream source video by authoritative file_time and verify decoded PTS."""

    def __init__(
        self,
        *,
        expected_width: int,
        expected_height: int,
        tolerance_seconds: float,
        display_matrix: Sequence[int] | None = None,
    ) -> None:
        self.expected_width = int(expected_width)
        self.expected_height = int(expected_height)
        self.tolerance_seconds = float(tolerance_seconds)
        self.display_matrix = tuple(display_matrix) if display_matrix is not None else None
        self._cursors: dict[str, _VideoCursor] = {}

    def _cursor(self, uri: str) -> _VideoCursor:
        cursor = self._cursors.get(uri)
        if cursor is None:
            path = Path(uri)
            if not path.is_file():
                raise FileNotFoundError(path)
            container = av.open(str(path))
            stream = container.streams.video[0]
            cursor = _VideoCursor(container=container, stream=stream)
            self._cursors[uri] = cursor
        return cursor

    def __call__(self, frame: Any) -> np.ndarray:
        reference = frame.media_references.get("camera0")
        if reference is None:
            raise ValueError("camera0 media reference missing")
        target = float(reference.file_time)
        cursor = self._cursor(reference.uri)
        if cursor.iterator is None or (
            cursor.last_pts is not None and target < cursor.last_pts - self.tolerance_seconds
        ):
            seek_pts = max(0, int(target / float(cursor.stream.time_base)))
            cursor.container.seek(seek_pts, stream=cursor.stream, any_frame=False, backward=True)
            cursor.iterator = cursor.container.decode(cursor.stream)
            cursor.last_pts = None
        closest = None
        closest_error = float("inf")
        for decoded in cursor.iterator:
            if decoded.pts is None:
                continue
            pts = float(decoded.pts * decoded.time_base)
            cursor.last_pts = pts
            error = abs(pts - target)
            if error < closest_error:
                closest = decoded
                closest_error = error
            if pts >= target:
                break
        if closest is None or closest_error > self.tolerance_seconds:
            raise ValueError(
                f"decoded PTS tolerance failed for {reference.uri}: "
                f"file_time={target:.9f}, error={closest_error:.9f}"
            )
        raw = closest.to_ndarray(format="rgb24")
        return orient_encoded_frame(
            raw,
            expected_width=self.expected_width,
            expected_height=self.expected_height,
            display_matrix=self.display_matrix,
        )

    def close(self) -> None:
        for cursor in self._cursors.values():
            cursor.container.close()
        self._cursors.clear()

    def __enter__(self) -> "ExactPtsVideoFrameProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
