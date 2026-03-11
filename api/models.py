from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from api.database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    source_url = Column(String, index=True)
    title = Column(String, nullable=True)        # AI 분석 제목
    source_title = Column(String, nullable=True) # 소셜 플랫폼 원본 제목
    duration = Column(Float, nullable=True)
    resolution = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending") # pending, processing, completed, failed
    error_message = Column(String, nullable=True)
    opinion = Column(String, nullable=True)  # JSON-encoded list[str] — AI 분석 의견
    created_at = Column(DateTime, default=datetime.utcnow)

    features = relationship("Feature", back_populates="video", uselist=False)
    scores = relationship("Score", back_populates="video", uselist=False)

class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    
    scene_frequency = Column(Float, nullable=True)
    visual_intensity_3s = Column(Float, nullable=True)
    has_faces = Column(Boolean, nullable=True)
    objects_detected = Column(String, nullable=True) # JSON string or comma-separated
    subtitle_density = Column(Float, nullable=True)
    speech_tempo = Column(Float, nullable=True)
    has_music = Column(Boolean, nullable=True)
    emotional_tone = Column(String, nullable=True)
    pacing = Column(Float, nullable=True)
    audio_energy = Column(Float, nullable=True)
    transcript = Column(String, nullable=True)

    video = relationship("Video", back_populates="features")

class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    
    hook_score = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True)
    storytelling_score = Column(Float, nullable=True)
    product_exposure_score = Column(Float, nullable=True)

    video = relationship("Video", back_populates="scores")

class ClusterResult(Base):
    __tablename__ = "cluster_results"

    id = Column(Integer, primary_key=True, index=True)
    cluster_name = Column(String, index=True)
    video_ids = Column(String) # Comma-separated or JSON
    avg_scores = Column(String) # JSON string representation
    avg_features = Column(String) # JSON string representation
    created_at = Column(DateTime, default=datetime.utcnow)
