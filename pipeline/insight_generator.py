"""
InsightGenerator — Google Gemini AI를 활용한 영상 분석 인사이트 생성기.

Gemini API가 설정되어 있으면 AI가 풍부한 인사이트를 생성하고,
API 키가 없거나 오류 발생 시 규칙 기반 폴백(fallback)으로 자동 전환됩니다.
"""
import os
from typing import Dict, Any

from dotenv import load_dotenv  # type: ignore

load_dotenv()

# ── Gemini 클라이언트 초기화 ─────────────────────────────────────────────────
_gemini_client = None
_gemini_model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    try:
        from google import genai  # type: ignore
        _gemini_client = genai.Client(api_key=api_key)
        print(f"[InsightGenerator] ✅ Gemini 연결 성공 (모델: {_gemini_model_name})")
    except Exception as e:
        print(f"[InsightGenerator] ⚠️ Gemini 초기화 실패: {e}")
        _gemini_client = None

    return _gemini_client


# ── 사용자 커스터마이징 프롬프트 ────────────────────────────────────────────
# 이 변수들을 수정해서 AI가 어떤 역할로 분석할지, 어떤 형식으로 출력할지 결정합니다.

SYSTEM_ROLE = """당신은 숏폼 콘텐츠 마케팅 전문가이자 SNS 바이럴 분석가입니다.
유튜브 쇼츠, 인스타그램 릴스, 틱톡 영상을 분석하여 
구체적이고 실행 가능한 인사이트를 제공합니다.
항상 한국어로 답변하며, 마케터와 크리에이터가 바로 활용할 수 있는 실용적인 조언을 제공합니다."""

TITLE_PROMPT_TEMPLATE = """
다음은 숏폼 영상 분석 데이터입니다. 영상에서 실제로 무슨 일이 일어나는지를 짧게 묘사하는 제목을 작성하세요.

## 분석 데이터
- 전체 평균: {total_score:.1f}/100 → 등급 기준: 85이상=A, 70이상=B, 50이상=C, 미만=D
- 감정 톤: {emotional_tone}
- 얼굴 등장 여부: {has_faces}
- 감지된 객체: {objects_detected}
- 원본 제목: {original_title}
- 영상 대본 요약: {transcript_summary}

## 규칙
- 점수나 분석 지표(훅, 템포, 참여도 등)는 절대 제목에 포함하지 마세요.
- 영상에 등장하는 사람, 행동, 장소, 사물, 주제를 중심으로 작성하세요.
- 제목은 10자 내외의 짧고 자연스러운 한국어로 작성하세요.

## 출력 형식 (정확히 이 형식 사용):
등급 · 영상 내용 요약 제목

## 예시:
A등급 · 간호사들과의 솔직한 대화
B등급 · 봄동으로 비빔밥 만들기
C등급 · 강아지와 고양이의 첫 만남
D등급 · 새벽 편의점 아르바이트 브이로그

한 줄로만 답변하세요. 앞뒤 설명 없이 제목만 작성하세요.
"""

OPINION_PROMPT_TEMPLATE = """
다음은 숏폼 영상(TikTok, Reels, Shorts) 분석 데이터입니다.
이 데이터를 바탕으로 마케터나 크리에이터가 참고할 만한 '종합 평가 의견'을 3~5개의 리스트 항목(-)으로 작성해주세요.

## 분석 데이터
- 훅 점수: {hook_score}/100 | 참여도: {engagement_score}/100 | 스토리텔링: {storytelling_score}/100 | 제품 노출: {product_exposure_score}/100
- 전체 평균: {total_score:.1f}/100
- 감정 톤: {emotional_tone} | 장면 전환: {scene_frequency:.2f}/초 | 발화 속도: {speech_tempo:.0f} WPM
- 얼굴 등장: {has_faces} | 감지된 객체: {objects_detected}
- 대본 요약: {transcript_summary}

## 규칙
- 첫 번째 항목은 반드시 "영상 내용 요약: "으로 시작하며, 영상에 등장하는 객체, 인물, 대본 등을 종합하여 **현재 어떤 상황이 벌어지고 있는지 전체적인 맥락을 묘사**하세요. (단순히 자막이나 음성을 그대로 나열하지 마세요.)
- 두 번째 항목부터는 영상의 총평, 장점, 보완점 등을 구체적으로 피드백하세요.
- 문장 앞에 이모지(📊, 🌟, 👍, ⚠️ 등)를 절대 사용하지 마세요.
- 반드시 한국어로 작성하며, 구체적이고 실용적인 피드백을 제공하세요.
- 각 항목은 '- ' 로 시작하세요.
"""


class InsightGenerator:
    """
    Google Gemini AI를 활용한 영상 인사이트 생성기.
    API 키가 없으면 규칙 기반 폴백으로 자동 전환됩니다.
    """

    TONE_MAP: Dict[str, str] = {
        "energetic": "에너제틱",
        "dramatic": "드라마틱",
        "calm": "잔잔한",
        "informative": "정보성",
        "neutral": "중립적",
    }

    @staticmethod
    def _calc_total_score(scores: Dict[str, float]) -> float:
        return (
            scores.get("hook_score", 0)
            + scores.get("engagement_score", 0)
            + scores.get("storytelling_score", 0)
            + scores.get("product_exposure_score", 0)
        ) / 4

    def _build_context(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
        original_title: str = "",
    ) -> Dict[str, Any]:
        """프롬프트에 사용할 컨텍스트 딕셔너리를 구성합니다."""
        total_score = self._calc_total_score(scores)

        # 대본 전체 (음성 기반)
        transcript = features.get("transcript", "")
        transcript_summary = transcript or "없음"

        emotional_tone = features.get("emotional_tone", "neutral")
        tone_kr = self.TONE_MAP.get(emotional_tone, emotional_tone)

        return {
            "hook_score": scores.get("hook_score", 0),
            "engagement_score": scores.get("engagement_score", 0),
            "storytelling_score": scores.get("storytelling_score", 0),
            "product_exposure_score": scores.get("product_exposure_score", 0),
            "total_score": total_score,
            "emotional_tone": tone_kr,
            "scene_frequency": features.get("scene_frequency", 0),
            "speech_tempo": features.get("speech_tempo", 0),
            "audio_energy": features.get("audio_energy", 0),
            "has_faces": "있음" if features.get("has_faces") else "없음",
            "objects_detected": features.get("objects_detected", "없음") or "없음",
            "transcript_summary": transcript_summary,
            "original_title": original_title or "없음",
        }

    def _call_gemini(self, prompt: str) -> str | None:
        """Gemini API를 호출합니다. 실패 시 None 반환."""
        client = _get_gemini_client()
        if client is None:
            return None
        try:
            full_prompt = f"{SYSTEM_ROLE}\n\n{prompt}"
            response = client.models.generate_content(
                model=_gemini_model_name,
                contents=full_prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"[InsightGenerator] ⚠️ Gemini API 오류: {e}")
            return None

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def generate_title(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
        original_title: str = "",
    ) -> str:
        """AI가 생성한 한 줄 영상 제목을 반환합니다."""
        ctx = self._build_context(scores, features, original_title)
        prompt = TITLE_PROMPT_TEMPLATE.format(**ctx)
        result = self._call_gemini(prompt)
        if result:
            # Gemini가 여러 줄을 반환할 경우 첫 줄만 사용
            return result.split("\n")[0].strip()
        # 폴백: 규칙 기반
        return self._fallback_title(scores, features, original_title)

    def generate_opinion(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
    ) -> list[str]:
        """AI가 생성한 분석 인사이트 리스트를 반환합니다."""
        ctx = self._build_context(scores, features)
        prompt = OPINION_PROMPT_TEMPLATE.format(**ctx)
        result = self._call_gemini(prompt)
        if result:
            # 마크다운 리스트 파싱 (- 로 시작하는 줄들)
            lines = [
                line.strip()
                for line in result.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return lines if lines else self._fallback_opinion(scores, features)
        # 폴백: 규칙 기반
        return self._fallback_opinion(scores, features)

    def generate_timeline_segments(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
        duration: float,
    ) -> list[Dict[str, Any]]:
        """타임라인 세그먼트 분석 (프론트엔드 타임라인용)."""
        hook_end = min(3.0, duration * 0.15)
        body_end = duration * 0.80
        outro_end = duration

        return [
            {
                "key": "hook",
                "label": "도입부 (Hook)",
                "time_range": f"0s ~ {hook_end:.1f}s",
                "score": scores.get("hook_score", 0),
                "flex": 15,
                "strength": (
                    "시선을 끄는 연출과 오디오가 우수합니다."
                    if scores.get("hook_score", 0) >= 70
                    else None
                ),
                "weakness": (
                    "시각적 자극이 부족해 초반 이탈 우려가 있습니다."
                    if scores.get("hook_score", 0) < 70
                    else None
                ),
                "tip": "첫 장면에 화려한 전환 또는 강한 멘트를 배치하세요.",
                "metrics": {
                    "visual_intensity": round(features.get("visual_intensity_3s", 0), 2),
                    "audio_energy": round(features.get("audio_energy", 0) * 100, 1),
                },
            },
            {
                "key": "body",
                "label": "본론 전개 (Body)",
                "time_range": f"{hook_end:.1f}s ~ {body_end:.1f}s",
                "score": float(round(
                    (scores.get("engagement_score", 0) + scores.get("storytelling_score", 0)) / 2, 1
                )),
                "flex": 65,
                "strength": (
                    "적절한 컷 전환과 페이싱으로 지루함을 방지합니다."
                    if scores.get("engagement_score", 0) >= 60
                    else None
                ),
                "weakness": (
                    "중반부 템포가 느려지거나 내용 전개가 약합니다."
                    if scores.get("engagement_score", 0) < 60
                    else None
                ),
                "tip": "중반부 페이싱을 유지하고 핵심 메시지를 반복 강조하세요.",
                "metrics": {
                    "engagement": scores.get("engagement_score", 0),
                    "storytelling": scores.get("storytelling_score", 0),
                    "pacing_wpm": round(features.get("speech_tempo", 0), 0),
                    "scene_cuts_per_sec": round(features.get("scene_frequency", 0), 2),
                },
            },
            {
                "key": "outro",
                "label": "마무리 & CTA (Outro)",
                "time_range": f"{body_end:.1f}s ~ {outro_end:.1f}s",
                "score": scores.get("product_exposure_score", 0),
                "flex": 20,
                "strength": (
                    "핵심 메시지나 제품이 안정적으로 표출되었습니다."
                    if scores.get("product_exposure_score", 0) >= 50
                    else None
                ),
                "weakness": (
                    "행동 유도(CTA)나 각인 효과가 부족합니다."
                    if scores.get("product_exposure_score", 0) < 50
                    else None
                ),
                "tip": "마지막 3초에 CTA 텍스트(구독/팔로우/링크)를 삽입하세요.",
                "metrics": {
                    "product_exposure": scores.get("product_exposure_score", 0),
                    "objects_detected": features.get("objects_detected", "없음"),
                },
            },
        ]

    # ── 폴백 (규칙 기반, Gemini 없을 때 사용) ────────────────────────────────

    def _fallback_title(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
        original_title: str = "",
    ) -> str:
        total_score = self._calc_total_score(scores)

        if total_score >= 85:
            grade = "A등급"
        elif total_score >= 70:
            grade = "B등급"
        elif total_score >= 50:
            grade = "C등급"
        else:
            grade = "D등급"

        # 1순위: transcript 앞부분으로 내용 요약 (가장 정확한 스토리 힌트)
        transcript = str(features.get("transcript", "") or "").strip()
        if transcript:
            # 첫 문장 또는 앞 20자 추출
            first_sentence = transcript.split(".")[0].split("?")[0].split("!")[0].strip()
            snippet = first_sentence[:20] + ("…" if len(first_sentence) > 20 else "")  # type: ignore[index]
            if snippet:
                return f"{grade} · {snippet}"

        # 2순위: 원본 제목 (yt-dlp source_title)
        if original_title and str(original_title).strip():
            title_str = str(original_title).strip()
            trimmed = title_str[:30] + ("…" if len(title_str) > 30 else "")  # type: ignore[index]
            return f"{grade} · {trimmed}"

        # 3순위: 감지된 객체 기반
        objects = str(features.get("objects_detected", "") or "").strip()
        if objects and objects != "없음":
            first_obj = objects.split(",")[0].strip()
            return f"{grade} · {first_obj} 등장 영상"

        # 최후 폴백: 감정 톤 기반
        tone_adj_map = {
            "energetic": "에너제틱한",
            "dramatic": "드라마틱한",
            "calm": "잔잔한",
            "informative": "정보 전달형",
            "neutral": "숏폼",
        }
        tone_adj = tone_adj_map.get(features.get("emotional_tone", "neutral"), "숏폼")
        return f"{grade} · {tone_adj} 영상"

    def _fallback_opinion(
        self,
        scores: Dict[str, float],
        features: Dict[str, Any],
    ) -> list[str]:
        opinion: list[str] = []
        total_score = self._calc_total_score(scores)

        transcript = str(features.get("transcript", "") or "").strip()
        if transcript:
            opinion.append(f"**영상 내용 요약**: {transcript}")
        else:
            objects = str(features.get("objects_detected", "") or "").strip()
            if objects and objects != "없음":
                opinion.append(f"**영상 내용 요약**: 주로 {objects} 등이 등장하는 영상입니다.")

        if total_score >= 85:
            opinion.append("**탁월한 성과 기대**: 전체 밸런스가 매우 뛰어나며 바이럴 잠재력이 높은 A등급 숏폼 구조입니다.")
        elif total_score >= 70:
            opinion.append("**양호한 콘텐츠**: 1~2가지 요소만 보완하면 성과가 극대화될 수 있습니다.")
        elif total_score >= 50:
            opinion.append("**보통 수준**: 차별화된 훅(Hook)이나 연출이 아쉽습니다.")
        else:
            opinion.append("**개선 필요**: 기획 및 편집 템포 수정이 권장됩니다.")

        if scores.get("hook_score", 0) < 60:
            opinion.append("- **초반 이탈 주의**: 첫 3초 시각적 자극이 부족합니다. 강한 훅 대사나 화면 전환을 배치하세요.")
        else:
            opinion.append("- **성공적인 훅**: 첫 3초 시선을 사로잡는 연출력이 우수합니다.")

        if scores.get("storytelling_score", 0) >= 80:
            opinion.append("- **우수한 맥락 전개**: 내용이 논리적으로 잘 전개되어 몰입도를 높이고 있습니다.")

        pacing = features.get("scene_frequency", 0)
        if pacing < 0.3:
            opinion.append("- **느린 템포**: 컷 전환 빈도가 낮아 루즈하게 느껴질 수 있습니다. 1~2초 단위 편집을 시도해 보세요.")
        elif pacing > 1.0:
            opinion.append("- **빠른 템포**: 화면 전환이 매우 역동적입니다. MZ세대나 틱톡 포맷에 최적화된 스피드입니다.")

        if features.get("emotional_tone") == "energetic":
            opinion.append("- **분위기 & 오디오**: 활기찬 에너지 톤이 감지되었습니다. 제품 리뷰나 일상 브이로그에 잘 어울립니다.")

        return opinion
