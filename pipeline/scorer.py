from typing import Dict, Any

def calculate_scores(visual_features: Dict[str, Any], audio_features: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates composite scores based on extracted features.

    Scores are on a scale of 0 to 100.

    콘텐츠 유형에 따라 점수 기준을 다르게 적용합니다:
      - 인물 등장 콘텐츠: 얼굴 감지 보너스 (감정 연결 요소)
      - 비주얼 중심 콘텐츠(음식/제품/동물 등): 객체 다양성 보너스
    """

    # 객체 목록 — 여러 섹션에서 공유
    objects_str = visual_features.get("objects_detected", "")
    objects_list = [obj.strip() for obj in objects_str.split(",") if obj.strip()]

    # 콘텐츠 유형 판별
    # 얼굴이 없어도 객체가 감지되면 '비주얼 중심 콘텐츠'로 분류
    has_faces = visual_features.get("has_faces", False)
    is_faceless_content = (not has_faces) and (
        len(objects_list) >= 1
        or visual_features.get("visual_intensity_3s", 0) >= 15
        or visual_features.get("scene_frequency", 0) >= 0.3
    )

    # 1. Hook Score (strength of first 3 seconds, scene change rate, emotional spike)
    intensity_score = min(visual_features.get("visual_intensity_3s", 0) / 50.0 * 40, 40)
    scene_rate_score = min(visual_features.get("scene_frequency", 0) / 2.0 * 30, 30)
    emotional_spike_score = min(audio_features.get("audio_energy", 0) * 30, 30)
    hook_score = min(intensity_score + scene_rate_score + emotional_spike_score, 100)

    # 2. Engagement Score (visual activity, speech tempo, subtitle density)
    general_visual_activity = min(visual_features.get("visual_intensity_3s", 0) / 50.0 * 20, 20)

    if has_faces:
        # 인물 등장 콘텐츠: 얼굴 감정 연결 보너스 (최대 15점)
        visual_activity_score = 15 + general_visual_activity
    elif is_faceless_content:
        # 비주얼 중심 콘텐츠: 객체 다양성으로 대체 (최대 15점)
        object_variety_bonus = min(len(objects_list) * 5, 15)
        visual_activity_score = object_variety_bonus + general_visual_activity
    else:
        visual_activity_score = general_visual_activity

    tempo = audio_features.get("speech_tempo", 0)
    tempo_score = 30 if 120 <= tempo <= 180 else (
        min(tempo / 120.0 * 30, 30) if tempo < 120 else max(30 - (tempo - 180) / 5.0, 0)
    )

    density_score = min(visual_features.get("subtitle_density", 0) * 35, 35)
    engagement_score = min(visual_activity_score + tempo_score + density_score, 100)

    # 3. Storytelling Score (narrative progression, pacing stability)
    transcript_len = len(audio_features.get("transcript", ""))
    narrative_score = min(transcript_len / 200.0 * 50, 50)

    pacing = audio_features.get("pacing", 0)
    pacing_stability = 50 - min(abs(pacing - 1.0) * 50, 50)
    storytelling_score = min(narrative_score + pacing_stability, 100)

    # 4. Product Exposure Score (object detection frequency, product visibility duration)
    object_frequency_score = min(len(objects_list) * 10, 50)

    product_keywords = [
        "bottle", "cup", "cell phone", "laptop", "book",
        "handbag", "backpack", "suitcase", "tv", "microwave", "refrigerator",
    ]
    detected_products = [obj for obj in objects_list if obj.lower() in product_keywords]
    product_visibility_score = min(len(detected_products) * 25, 50)

    product_score = min(object_frequency_score + product_visibility_score, 100)

    return {
        "hook_score": round(hook_score, 1),
        "engagement_score": round(engagement_score, 1),
        "storytelling_score": round(storytelling_score, 1),
        "product_exposure_score": round(product_score, 1),
    }
