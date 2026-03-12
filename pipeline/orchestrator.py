"""
Refactored orchestrator — clean 6-stage pipeline with pluggable detector.

Pipeline stages:
  1. Collector      — download video
  2. Preprocessor   — read frames / metadata
  3. Detector       — run object detection (swappable via `detector_name`)
  4. VisualExtractor— scene + face + intensity features
  5. AudioAnalyzer  — STT / pacing / energy (Whisper + librosa)
  6. Scorer         — convert features → scores
  7. InsightGenerator — generate title + opinion
  8. DB             — persist results
"""

import os
import json
import concurrent.futures
from collections import Counter
from typing import Optional

import cv2  # type: ignore
from sqlalchemy.orm import Session  # type: ignore

from api import models  # type: ignore
from pipeline.collector import VideoCollector
from pipeline.preprocessor import VideoPreprocessor
from pipeline.detectors import get_detector
from pipeline.analyzer_visual import extract_visual_features
from pipeline.analyzer_audio_ai import AudioAnalyzer
from pipeline.scorer import calculate_scores
from pipeline.insight_generator import InsightGenerator
from pipeline.file_renamer import rename_video_file


def _run_object_detection(video_path: str, detector_name: str = "yolo") -> str:
    """
    Runs the chosen detector over sampled frames and returns top-5 objects.
    Isolated as a standalone function so it can be submitted to ThreadPoolExecutor.
    """
    detector = get_detector(detector_name)
    preprocessor = VideoPreprocessor()
    all_objects: list[str] = []

    for _idx, frame in preprocessor.get_frames(video_path, sample_rate=60):
        detections = detector.detect(frame)
        all_objects.extend(detections)

    if not all_objects:
        return ""

    counter = Counter(all_objects)
    return ",".join(item for item, _ in counter.most_common(5))


def _analyze_and_persist(
    video_id: int,
    file_path: str,
    original_title: str,
    db: Session,
    video,
    detector_name: str,
) -> None:
    """
    Stages 3-8: 병렬 분석 → 채점 → AI 제목 생성 → 파일명 변경 → DB 저장.
    URL 파이프라인과 로컬 파이프라인이 공유하는 공통 로직.
    """
    insight_gen = InsightGenerator()

    # ── Stages 3-5: Sequential analysis (low-memory mode) ─────────────
    import gc
    video_obj = db.query(models.Video).filter(models.Video.id == video_id).first()

    video_obj.progress_stage = "visual"
    db.commit()
    visual_features: dict = extract_visual_features(file_path)
    gc.collect()

    video_obj.progress_stage = "detection"
    db.commit()
    objects_detected: str = _run_object_detection(file_path, detector_name)
    gc.collect()

    video_obj.progress_stage = "audio"
    db.commit()
    audio_analyzer = AudioAnalyzer()
    audio_features: dict = audio_analyzer.analyze_audio(file_path)
    gc.collect()

    video_obj.progress_stage = "scoring"
    db.commit()

    if objects_detected:
        visual_features["objects_detected"] = objects_detected

    if "subtitle_density" in audio_features:
        visual_features["subtitle_density"] = audio_features.pop("subtitle_density")

    # ── Stage 6: Score ────────────────────────────────────────────────
    scores = calculate_scores(visual_features, audio_features)

    # ── Stage 7: Insights + AI Title + Opinion ───────────────────────
    video_obj.progress_stage = "insights"
    db.commit()
    all_features = {**visual_features, **audio_features}
    video.title = insight_gen.generate_title(scores, all_features, original_title)
    opinion_lines = insight_gen.generate_opinion(scores, all_features)
    video.opinion = json.dumps(opinion_lines, ensure_ascii=False)
    db.commit()

    # ── Stage 7.5: 파일명 → AI 제목 기반으로 변경 ────────────────────
    # db.commit() 이후 video.file_path가 DB에서 재로딩될 수 있으므로
    # 파라미터로 받은 file_path(확실히 올바른 경로)를 우선 사용
    rename_src = file_path or video.file_path
    if rename_src and video.title:
        new_path = rename_video_file(
            rename_src, video.title,
            source_title=video.source_title or "",
            db=db, video=video,
        )
        if new_path:
            video.file_path = new_path

    # ── Stage 8: Persist features & scores ───────────────────────────
    db.add(models.Feature(video_id=video_id, **visual_features, **audio_features))
    db.add(models.Score(video_id=video_id, **scores))
    video.status = "completed"
    db.commit()


def process_video_pipeline(
    video_id: int,
    db: Session,
    detector_name: str = "yolo",
) -> None:
    """
    Main pipeline entry point.

    Args:
        video_id:      DB primary key of the video to process.
        db:            SQLAlchemy session.
        detector_name: Which detector to use ("yolo" | "detectron2" | …).
    """
    try:
        # ── Stage 1: Fetch from DB ────────────────────────────────────────
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            return

        video.status = "downloading"
        db.commit()

        # ── Stage 2: Collect / Download ───────────────────────────────────
        collector = VideoCollector()
        metadata, error_msg = collector.download_video(video.source_url)

        if not metadata:
            video.status = "failed"
            video.error_message = error_msg
            db.commit()
            return

        original_title = metadata.get("title", "") or ""
        video.title = original_title
        video.source_title = original_title  # 원본 플랫폼 제목 보존
        video.duration = metadata.get("duration")
        video.resolution = metadata.get("resolution")
        video.file_path = metadata.get("file_path")
        video.status = "processing"
        db.commit()

        _analyze_and_persist(video_id, video.file_path, original_title, db, video, detector_name)

    except Exception as exc:
        print(f"[Pipeline] Error for video {video_id}: {exc}")
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video:
            video.status = "failed"
            video.error_message = str(exc)
            db.commit()


def process_local_video_pipeline(
    video_id: int,
    file_path: str,
    db: Session,
    detector_name: str = "yolo",
) -> None:
    """
    Pipeline entry point for locally uploaded video files.
    Skips the download stage and goes straight to analysis.

    Args:
        video_id:    DB primary key.
        file_path:   Absolute path to the already-saved video file.
        db:          SQLAlchemy session.
        detector_name: Which detector to use.
    """
    try:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            return

        # Get basic metadata from the file itself
        preprocessor = VideoPreprocessor()
        meta = preprocessor.get_metadata(file_path)

        video.file_path = file_path
        video.duration = meta.get("duration", 0.0)
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        video.resolution = f"{w}x{h}"
        video.status = "processing"
        db.commit()

        original_title = video.title or os.path.basename(file_path)
        if not video.source_title:
            video.source_title = original_title
            db.commit()

        _analyze_and_persist(video_id, file_path, original_title, db, video, detector_name)

    except Exception as exc:
        print(f"[Pipeline] Local upload error for video {video_id}: {exc}")
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video:
            video.status = "failed"
            video.error_message = str(exc)
            db.commit()

