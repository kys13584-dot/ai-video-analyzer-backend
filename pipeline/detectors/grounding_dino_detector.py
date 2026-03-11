import numpy as np
from typing import List
from pipeline.detectors.base import BaseDetector


class GroundingDinoDetector(BaseDetector):
    """
    Grounding DINO-based open-vocabulary object detector (stub).

    Grounding DINO supports text-prompted detection, making it suitable
    for detecting product categories without retraining.

    To enable:
    1. Install: pip install groundingdino-py
    2. Replace the `detect` body with real inference code.
    3. Pass `prompts: list[str]` to the constructor for open-vocab detection.
    See: https://github.com/IDEA-Research/GroundingDINO
    """

    def __init__(self, prompts: List[str] | None = None):
        raise NotImplementedError(
            "GroundingDinoDetector is not yet implemented.\n"
            "Install groundingdino-py and implement the `detect` method.\n"
            "See: https://github.com/IDEA-Research/GroundingDINO"
        )

    def detect(self, frame: np.ndarray) -> List[str]:
        raise NotImplementedError("Implement with Grounding DINO inference.")
