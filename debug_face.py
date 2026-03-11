import sys
import os
import cv2
import numpy as np
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.collector import VideoCollector
from pipeline.preprocessor import VideoPreprocessor

def debug_face_detection_mediapipe(url):
    collector = VideoCollector()
    print("다운로드 중...")
    try:
        res = collector.download_video(url)
        path = res["file_path"] if isinstance(res, dict) else res[0]
        if isinstance(path, dict) and "file_path" in path:
            path = path["file_path"]
        print("다운로드 완료:", path)
        
        preprocessor = VideoPreprocessor()
        meta = preprocessor.get_metadata(path)
        fps = meta["fps"]
        
        mp_face_detection = mp.solutions.face_detection
        
        # 저장 폴더
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_faces_mp")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        print("프레임 분석 및 오탐지 캡처 시작...")
        
        FACE_SAMPLE_INTERVAL = 1
        saved_count = 0
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            for frame_idx, frame in preprocessor.get_frames(path, sample_rate=1):
                if frame_idx % FACE_SAMPLE_INTERVAL == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                    
                    if results.detections:
                        # 감지된 부분을 원본 해상도 프레임에 빨간 박스로 그림
                        display_frame = frame.copy()
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = display_frame.shape
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            
                        out_path = os.path.join(debug_dir, f"face_mp_frame_{frame_idx}.jpg")
                        cv2.imwrite(out_path, display_frame)
                        saved_count += 1
                        print(f"  얼굴 감지! 프레임 {frame_idx} 저장 완료: {out_path}")
                        
        print(f"새로운 MediaPipe 분석 완료. 총 {saved_count}개의 프레임 이미지가 저장되었습니다.")
        
        if os.path.exists(path):
            os.remove(path)
            
    except Exception as e:
        print("에러 발생:", e)

if __name__ == "__main__":
    url = "https://www.instagram.com/reels/DVK_k8_E5XZ/"
    debug_face_detection_mediapipe(url)
