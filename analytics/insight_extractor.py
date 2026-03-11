from sqlalchemy.orm import Session
from typing import Dict, Any
from api import models

# 감정 톤 한국어 매핑
TONE_KR = {
    "energetic":   "에너제틱",
    "dramatic":    "드라마틱",
    "calm":        "잔잔함",
    "informative": "정보 전달형",
    "neutral":     "중립적",
}

class InsightExtractor:
    def __init__(self, db: Session):
        self.db = db

    def extract_insights(self) -> Dict[str, Any]:
        """
        고성과 영상들의 공통 패턴을 추출합니다.
        """
        videos = self.db.query(models.Video).join(models.Score).join(models.Feature).all()

        if not videos:
            return {"message": "인사이트를 추출하기에 데이터가 부족합니다."}

        high_performers = []
        low_performers = []

        for v in videos:
            if not v.scores or not v.features:
                continue

            avg_score = (
                (v.scores.hook_score or 0) +
                (v.scores.engagement_score or 0) +
                (v.scores.storytelling_score or 0) +
                (v.scores.product_exposure_score or 0)
            ) / 4.0

            if avg_score > 70:
                high_performers.append(v)
            else:
                low_performers.append(v)

        if not high_performers:
            return {"message": "아직 고성과 영상이 없습니다. 더 많은 영상을 분석하면 패턴이 나타납니다."}

        # 고성과 영상들의 평균 지표 계산
        avg_scene_freq = sum(v.features.scene_frequency or 0 for v in high_performers) / len(high_performers)
        avg_intensity  = sum(v.features.visual_intensity_3s or 0 for v in high_performers) / len(high_performers)
        avg_pacing     = sum(v.features.pacing or 0 for v in high_performers) / len(high_performers)

        # 가장 많이 나온 감정 톤
        tones: Dict[str, int] = {}
        for v in high_performers:
            tone = v.features.emotional_tone or "neutral"
            tones[tone] = tones.get(tone, 0) + 1

        top_tone_en = max(tones, key=tones.get) if tones else "neutral"
        top_tone_kr = TONE_KR.get(top_tone_en, top_tone_en)

        return {
            "total_analyzed": len(videos),
            "high_performers_count": len(high_performers),
            "recommended_patterns": {
                "optimal_scene_frequency": round(avg_scene_freq, 2),
                "optimal_visual_intensity": round(avg_intensity, 2),
                "optimal_pacing": round(avg_pacing, 2),
                "best_performing_tone": top_tone_kr,
            },
            "insights": [
                f"고성과 영상의 평균 페이싱은 {avg_pacing:.1f}배 속도로, 빠른 템포가 시청자 몰입에 효과적입니다.",
                f"가장 반응이 좋은 감정 톤은 '{top_tone_kr}'입니다. 제작 시 해당 분위기 연출을 참고하세요.",
                f"초당 {avg_scene_freq:.1f}회의 화면 전환 빈도가 시청 지속률 유지에 최적화된 수치입니다.",
            ],
        }
