"""
VisualFeatureExtractor — uses VideoPreprocessor for frame reading.

Extracts scene-change rate, first-3s visual intensity, and face detection.

얼굴 감지 전략 (MediaPipe Face Detection):
  - Google의 MediaPipe 라이브러리를 사용하여 빠르고 매우 높은 정확도의 얼굴 감지 수행
  - 배경이나 사물을 사람 얼굴로 오인(False Positive)하는 Haar Cascade의 고질적인 단점 해결
  - 다중 프레임 확인 — 여러 프레임에서 반복 감지 시에만 True (견고함 보장)

Object detection is intentionally NOT done here — that is handled by the
Detector stage in orchestrator.py.
"""
import os
import urllib.request
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Any

from pipeline.preprocessor import VideoPreprocessor

# ── MediaPipe 얼굴 감지 모델 다운로드 ──────────────────────────────────────────
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_FACE_MODEL_PATH = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")
_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

def _ensure_face_model_exists():
    if not os.path.exists(_FACE_MODEL_PATH):
        print(f"[Visual] MediaPipe 얼굴 감지 모델 다운로드 중... (최초 1회)")
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
        print(f"[Visual] 모델 다운로드 완료.")

# MediaPipe를 런타임에 임포트 (초기 로딩 시간 단축 위함)
def _detect_faces_mediapipe(frame: np.ndarray, face_detector) -> int:
    """
    MediaPipe Tasks FaceDetector를 이용해 얼굴을 감지합니다.
    BGR 프레임을 입력받아 MediaPipe 이미지 포맷으로 넘기고, 감지 수를 리턴.
    """
    import mediapipe as mp
    
    # MediaPipe는 RGB 포맷을 요구
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = face_detector.detect(mp_image)
    if detection_result and detection_result.detections:
        return len(detection_result.detections)
    return 0


def extract_visual_features(video_path: str) -> Dict[str, Any]:
    """
    Extracts visual features from a video file.

    Returns a dict with:
      - scene_frequency       : scene cuts per second
      - visual_intensity_3s   : average pixel-diff in first 3 seconds
      - has_faces             : bool, 실제 사람 얼굴이 감지된 경우만 True
      - objects_detected      : placeholder string (overwritten by detector stage)
      - subtitle_density      : mock (OCR not yet implemented)
    """
    preprocessor = VideoPreprocessor()
    meta = preprocessor.get_metadata(video_path)

    fps = meta["fps"]
    duration = meta["duration"]
    frames_in_first_3s = int(3 * fps) if fps > 0 else 0

    # ── MediaPipe 얼굴 감지기 설정 ──────────────────────────────────
    skip_mediapipe = os.getenv("SKIP_MEDIAPIPE", "false").lower() == "true"
    face_detector_ctx = None

    if not skip_mediapipe:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        _ensure_face_model_exists()
        base_options = python.BaseOptions(model_asset_path=_FACE_MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
        face_detector_ctx = vision.FaceDetector.create_from_options(options)

    import contextlib

    @contextlib.contextmanager
    def _maybe_detector():
        if face_detector_ctx is not None:
            with face_detector_ctx as fd:
                yield fd
        else:
            yield None

    with _maybe_detector() as face_detector:
        
        # 단일 프레임 오탐을 방지하기 위해 최소 2개 프레임 이상 감지 요건
        FACE_CONFIRM_THRESHOLD = 2
        face_detection_count = 0
        faces_detected = False

        scene_changes = 0
        visual_intensity_3s = 0.0
        prev_gray: np.ndarray | None = None

        for frame_idx, frame in preprocessor.get_frames(video_path, sample_rate=1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 장면 전환 분석 ──────────────────────────────────────────────
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = float(np.mean(diff))

                if frame_idx < frames_in_first_3s:
                    visual_intensity_3s += mean_diff

                if mean_diff > 30:
                    scene_changes += 1

            # ── 얼굴 감지 (MediaPipe 딥러닝) ─────────────────────────────
            if not faces_detected and face_detector is not None:
                count = _detect_faces_mediapipe(frame, face_detector)
                if count > 0:
                    face_detection_count += 1
                    print(f"[Visual] 얼굴 후보 감지 ({face_detection_count}/{FACE_CONFIRM_THRESHOLD}) @ frame {frame_idx}")

                    if face_detection_count >= FACE_CONFIRM_THRESHOLD:
                        faces_detected = True
                        print(f"[Visual] ✅ 얼굴 확정 감지 (frame {frame_idx})")

            prev_gray = gray

    if not faces_detected:
        print(f"[Visual] ✗ 얼굴 미감지 (후보 {face_detection_count}회)")

    scene_frequency = scene_changes / duration if duration > 0 else 0.0
    avg_intensity_3s = (
        visual_intensity_3s / frames_in_first_3s if frames_in_first_3s > 0 else 0.0
    )

    return {
        "scene_frequency": scene_frequency,
        "visual_intensity_3s": avg_intensity_3s,
        "has_faces": faces_detected,
        "objects_detected": "",       # Filled later by the detector stage
        "subtitle_density": 0.5,      # Mock — OCR would replace this
    }
