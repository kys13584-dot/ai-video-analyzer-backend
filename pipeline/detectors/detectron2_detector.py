import numpy as np
from typing import List
from pipeline.detectors.base import BaseDetector


class Detectron2Detector(BaseDetector):
    """
    Detectron2-based object detector (stub).

    To enable:
    1. Install: pip install detectron2 (see https://detectron2.readthedocs.io)
    2. Replace the `detect` body with real Detectron2 inference code.
    """

    def __init__(self):
        raise NotImplementedError(
            "Detectron2Detector is not yet implemented.\n"
            "Install Detectron2 and implement the `detect` method in this file.\n"
            "See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
        )

    def detect(self, frame: np.ndarray) -> List[str]:
        raise NotImplementedError("Implement with Detectron2 inference.")
