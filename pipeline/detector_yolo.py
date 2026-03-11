import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os

class ObjectDetector:
    def __init__(self):
        # Initialize YOLOv8s (small model for balance between speed and accuracy)
        # It will automatically download the weights if not present
        self.model = YOLO("yolov8n.pt") 

    def analyze_objects(self, video_path: str, sample_rate: int = 60) -> str:
        """
        Detects objects in the video using YOLO.
        Samples frames based on sample_rate to improve performance.
        Args:
            video_path: Path to the video file
            sample_rate: Analyze 1 out of every `sample_rate` frames.
        Returns:
            A comma-separated string of the top 5 most common objects detected.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        all_detected_objects = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Resize frame to speed up inference if it's very large
                # Resize width to 640px while preserving aspect ratio
                h, w = frame.shape[:2]
                if w > 640:
                    new_h = int((640.0 / w) * h)
                    frame = cv2.resize(frame, (640, new_h))

                # Run YOLO inference
                results = self.model(frame, verbose=False)
                
                # Extract class names from results
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        all_detected_objects.append(class_name)

            frame_count += 1

        cap.release()

        if not all_detected_objects:
            return ""

        # Count frequencies of detected objects
        counter = Counter(all_detected_objects)
        
        # Get top 5 most common objects
        top_objects = [item[0] for item in counter.most_common(5)]
        
        return ",".join(top_objects)
