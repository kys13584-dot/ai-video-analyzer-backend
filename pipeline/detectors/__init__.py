"""
Detector registry for the AI Video Analyzer pipeline.

Adding a new detector:
1. Create a file in `pipeline/detectors/` that subclasses `BaseDetector`.
2. Register it in the `_REGISTRY` dict below.
3. No other files need to be changed.
"""

from pipeline.detectors.base import BaseDetector
from pipeline.detectors.yolo_detector import YoloDetector


_REGISTRY = {
    "yolo": YoloDetector,
    # "detectron2": Detectron2Detector,  # Uncomment after installing
    # "grounding_dino": GroundingDinoDetector,  # Uncomment after installing
}


def get_detector(name: str = "yolo") -> BaseDetector:
    """
    Factory to instantiate a detector by name.

    Args:
        name: One of the registered detector names (e.g. "yolo")

    Returns:
        An instance of a BaseDetector subclass.

    Raises:
        ValueError: If the name is not registered.
    """
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown detector '{name}'. Available: {available}\n"
            f"To add a new detector, see pipeline/detectors/__init__.py."
        )
    return cls()


__all__ = ["BaseDetector", "YoloDetector", "get_detector"]
