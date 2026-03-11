import numpy as np  # type: ignore
from typing import List, Tuple
from pipeline.detectors.base import BaseDetector

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class YoloDetector(BaseDetector):
    """
    YOLOv8-based object detector — center-weighted.

    화면 중앙 영역(center zone)에 위치한 객체를 우선시합니다.
    신뢰도 임계값과 중앙 거리 기반으로 정렬하여 가장 의미있는
    객체만 반환합니다.
    """

    # 신뢰도 임계값: 이 값 미만의 감지 결과는 무시
    CONF_THRESHOLD: float = 0.40

    # 중앙 존 비율: 0.5 = 화면의 중앙 50% 영역
    CENTER_ZONE: float = 0.50

    def __init__(self, model_name: str = "yolov8n.pt"):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self.model = YOLO(model_name)

    def _center_weight(self, box_xyxy: list, img_w: int, img_h: int) -> float:
        """
        박스 중심점이 화면 중앙에 얼마나 가까운지를 0.0~1.0으로 반환.
        화면 중앙(0.5, 0.5)에 가까울수록 1.0에 가까움.
        """
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2 / img_w  # 정규화된 x 중심 (0~1)
        cy = (y1 + y2) / 2 / img_h  # 정규화된 y 중심 (0~1)

        # 화면 중심(0.5, 0.5)까지의 유클리드 거리 (최대 ~0.707)
        dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        # 거리를 가중치로 변환, 중앙일수록 높음
        return max(0.0, 1.0 - dist * 2)

    def _is_in_center_zone(self, box_xyxy: list, img_w: int, img_h: int) -> bool:
        """박스 중심이 CENTER_ZONE 영역 내에 있는지 확인."""
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        margin = (1.0 - self.CENTER_ZONE) / 2
        return (margin <= cx <= 1 - margin) and (margin <= cy <= 1 - margin)

    def detect(self, frame: np.ndarray) -> List[str]:
        """
        화면 중앙 기반으로 객체를 탐지합니다.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            중앙 가중치 기준으로 정렬된 감지 객체명 리스트 (최대 5개)
        """
        img_h, img_w = frame.shape[:2]
        results = self.model(frame, verbose=False)

        # (class_name, conf, center_weight, is_center) 리스트로 수집
        candidates: List[Tuple[str, float, float, bool]] = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.CONF_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                class_name: str = self.model.names[class_id]

                xyxy = box.xyxy[0].tolist()
                cw = self._center_weight(xyxy, img_w, img_h)
                in_center = self._is_in_center_zone(xyxy, img_w, img_h)

                candidates.append((class_name, conf, cw, in_center))

        if not candidates:
            return []

        # 중앙 존 내부 객체를 우선, 그 다음 중앙 가중치 * 신뢰도 순 정렬
        candidates.sort(
            key=lambda x: (int(x[3]), x[2] * x[1]),
            reverse=True,
        )

        # 중복 제거 후 최대 5개 반환
        seen: set[str] = set()
        result: List[str] = []
        for name, _conf, _cw, _ic in candidates:
            if name not in seen:
                seen.add(name)
                result.append(name)
            if len(result) >= 5:
                break

        return result

