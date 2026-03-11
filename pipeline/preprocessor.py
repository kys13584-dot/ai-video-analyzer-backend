"""
VideoPreprocessor — shared frame extraction utility.

Centralizes the cv2.VideoCapture loop so that both the visual
feature extractor and the object detector can reuse it without
duplicating frame-reading code.
"""
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import Generator, Dict, Any, Tuple


class VideoPreprocessor:
    """
    Provides a single, reusable interface for reading video frames.
    """

    def get_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Returns basic metadata without reading all frames.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with fps, total_frames, and duration (seconds).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
        }

    def get_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_width: int = 640,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields sampled frames from the video.

        Args:
            video_path:  Path to the video file.
            sample_rate: Yield 1 frame per every `sample_rate` frames.
            max_width:   Resize frames wider than this (preserves aspect ratio).

        Yields:
            (frame_index, frame_bgr) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % sample_rate == 0:
                    h, w = frame.shape[:2]
                    if w > max_width:
                        new_h = int((max_width / w) * h)
                        frame = cv2.resize(frame, (max_width, new_h))
                    yield frame_index, frame

                frame_index += 1
        finally:
            cap.release()
