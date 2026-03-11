import logging
import sys
from pipeline.orchestrator import process_video_pipeline
from api.database import SessionLocal
from api.models import Video

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

db = SessionLocal()
video = db.query(Video).filter(Video.id == 6).first()
if video:
    print(f"Testing video id {video.id} url {video.source_url}")
    try:
        process_video_pipeline(video.id, db)
    except Exception as e:
        logging.exception("Error during processing:")
else:
    print("Video not found")
