import sys
import os

# backend 폴더 내에서 실행되도록 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.collector import VideoCollector
from pipeline.analyzer_visual import extract_visual_features

def test_face_detection(url):
    collector = VideoCollector()
    print("다운로드 중...")
    try:
        # DB에 저장 안 하고 단순히 다운로드만 (다운로드된 경로 반환)
        res = collector.download_video(url)
        path = res["file_path"] if isinstance(res, dict) else res[0]
        if isinstance(path, dict) and "file_path" in path:
            path = path["file_path"]
        print("다운로드 완료:", path)
        print("비주얼 특징 추출 시작...")
        features = extract_visual_features(path)
        print("결과:", features)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print("에러 발생:", e)

if __name__ == "__main__":
    test_face_detection("https://www.instagram.com/reels/DVTIsf1EqA2/")
