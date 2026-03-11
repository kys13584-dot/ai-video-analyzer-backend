from abc import ABC, abstractmethod
import numpy as np
from typing import List


class BaseDetector(ABC):
    """
    Abstract base class for all object detectors.

    To implement a custom detector:
    1. Subclass BaseDetector
    2. Implement the `detect` method
    3. Register it in `detectors/__init__.py` under `get_detector`
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[str]:
        """
        Detect objects in a single video frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3)

        Returns:
            List of detected class name strings, e.g. ["person", "car"]
        """
        raise NotImplementedError
