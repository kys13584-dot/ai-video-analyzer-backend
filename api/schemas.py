from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class VideoCreate(BaseModel):
    url: str

class VideoResponse(BaseModel):
    id: int
    source_url: str
    title: Optional[str] = None
    duration: Optional[float] = None
    status: str
    progress_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True

class ScoreResponse(BaseModel):
    hook_score: Optional[float] = None
    engagement_score: Optional[float] = None
    storytelling_score: Optional[float] = None
    product_exposure_score: Optional[float] = None

    class Config:
        orm_mode = True

class FeatureResponse(BaseModel):
    scene_frequency: Optional[float] = None
    visual_intensity_3s: Optional[float] = None
    has_faces: Optional[bool] = None
    objects_detected: Optional[str] = None
    subtitle_density: Optional[float] = None
    speech_tempo: Optional[float] = None
    has_music: Optional[bool] = None
    emotional_tone: Optional[str] = None
    pacing: Optional[float] = None
    audio_energy: Optional[float] = None
    transcript: Optional[str] = None

    class Config:
        orm_mode = True

class VideoDetailResponse(VideoResponse):
    scores: Optional[ScoreResponse] = None
    features: Optional[FeatureResponse] = None
    opinion: Optional[str] = None  # JSON-encoded list[str]

    class Config:
        orm_mode = True
