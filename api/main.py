from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from sqlalchemy.orm import Session # type: ignore
import os
import cv2 # type: ignore
import base64
import shutil
import uuid
from fastapi.responses import JSONResponse # type: ignore

from api import models, schemas, database # type: ignore
from pipeline.orchestrator import process_video_pipeline, process_local_video_pipeline # type: ignore

# Create database tables & run safe migrations
models.Base.metadata.create_all(bind=database.engine)
database.run_migrations()

app = FastAPI(title="AI Video Analysis Platform")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Video Analysis Platform API"}

@app.post("/api/videos", response_model=schemas.VideoResponse)
def create_video(
    video_in: schemas.VideoCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db),
    force_reanalyze: bool = Query(False, description="True이면 이미 분석된 영상도 강제 재분석"),
):
    existing_video = db.query(models.Video).filter(models.Video.source_url == video_in.url).first()

    if existing_video:
        if force_reanalyze:
            # ── 재분석: 기존 분석 데이터 초기화 ─────────────────────────
            if existing_video.scores:
                db.delete(existing_video.scores)
            if existing_video.features:
                db.delete(existing_video.features)
            existing_video.status = "pending"
            existing_video.error_message = None
            existing_video.title = None
            db.commit()
            db.refresh(existing_video)
            background_tasks.add_task(process_video_pipeline, existing_video.id, db)
            return existing_video

        elif existing_video.status == "failed":
            # 실패한 영상은 자동 재시도
            existing_video.status = "pending"
            existing_video.error_message = None
            db.commit()
            background_tasks.add_task(process_video_pipeline, existing_video.id, db)
            return existing_video

        else:
            # 이미 완료된 영상 → 캐시 반환
            return existing_video

    # 신규 영상
    db_video = models.Video(source_url=video_in.url, status="pending")
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    background_tasks.add_task(process_video_pipeline, db_video.id, db)
    return db_video

@app.get("/api/videos", response_model=list[schemas.VideoResponse])
def get_videos(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    videos = db.query(models.Video).order_by(models.Video.created_at.desc()).offset(skip).limit(limit).all()
    return videos


@app.post("/api/videos/upload", response_model=schemas.VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(database.get_db),
):
    """
    Upload a local video file (mp4, mov, mkv, webm) for analysis.
    """
    allowed_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_exts)}"
        )

    from pipeline.collector import DOWNLOAD_DIR
    file_id = str(uuid.uuid4())
    dest_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{ext}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    source_label = f"local_upload::{file.filename}"

    db_video = models.Video(
        source_url=source_label,
        title=file.filename,
        status="pending",
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    background_tasks.add_task(process_local_video_pipeline, db_video.id, dest_path, db)
    return db_video


@app.get("/api/videos/{video_id}", response_model=schemas.VideoDetailResponse)
def get_video(video_id: int, db: Session = Depends(database.get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@app.get("/api/videos/{video_id}/frames")
def get_video_frames(video_id: int, db: Session = Depends(database.get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video or not video.file_path or not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    cap = cv2.VideoCapture(video.file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        raise HTTPException(status_code=500, detail="Invalid video duration")

    # Target timestamps in seconds
    targets = {
        "hook": min(1.5, duration * 0.1),
        "body": duration * 0.5,
        "outro": duration * 0.9
    }

    from typing import Dict, Optional
    frames_b64: Dict[str, Optional[str]] = {}

    for key, target_sec in targets.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
        ret, frame = cap.read()
        if ret:
            # Resize frame to be small for tooltip (e.g., width 300px)
            h, w = frame.shape[:2]
            new_w = 200
            new_h = int((new_w / w) * h)
            small_frame = cv2.resize(frame, (new_w, new_h))
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_b64[key] = b64_str
        else:
            frames_b64[key] = None

    cap.release()
    return JSONResponse(content=frames_b64)

@app.get("/api/insights")
def get_insights(db: Session = Depends(database.get_db)):
    from analytics.insight_extractor import InsightExtractor # type: ignore
    extractor = InsightExtractor(db)
    return extractor.extract_insights()
