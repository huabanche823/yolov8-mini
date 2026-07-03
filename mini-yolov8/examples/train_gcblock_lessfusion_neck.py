from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    print(ROOT)
    print(f"ultralytics source: {ultralytics.__file__}")
    model = YOLO(ROOT / "ultralytics/cfg/models/v11/yolov11_gcblock_lessfusion_neck.yaml")
    results = model.train(
        data=ROOT / "datasets/WasteSortingv3/data_challenge.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        workers=4,
        device=0,
        project=ROOT / "runs",
        name="WasteSortingv3_yolov11_gcblock_lessfusion_neck",
        exist_ok=True,
        pretrained=False,
        val=True,
        plots=True,
        verbose=True,
        amp=False,
        optimizer="SGD",
        patience=20,
        seed=1,
        deterministic=False,
    )
    print(f"train_ok task={model.task} save_dir={results.save_dir}")


if __name__ == "__main__":
    main()
