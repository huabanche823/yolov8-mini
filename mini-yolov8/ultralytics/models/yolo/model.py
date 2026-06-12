# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLOv8 object detection wrapper.

    This mini package keeps only the detection task entry points while reusing
    the original Ultralytics detection trainer, validator, predictor, losses,
    dataloaders, and model parser.
    """

    def __init__(self, model: str | Path = "yolov8n.pt", task: str | None = "detect", verbose: bool = False):
        """Initialize a YOLOv8 detection model."""
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map the detection task to its model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
