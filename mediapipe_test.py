import urllib.request
import mediapipe as mp
import cv2
import numpy as np

model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
model_path = "blaze_face_short_range.tflite"

print("Downloading model...")
if not __import__('os').path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)
print("Model downloaded.")

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    # Use dummy image
    rgb_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    print("Detections length:", len(result.detections))
