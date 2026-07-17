from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    data_cfg = WORKSPACE / "WasteSortingv3_dataset/data_challenge.yaml"
    if not data_cfg.exists():
        data_cfg = ROOT / "datasets/WasteSortingv3/data_challenge.yaml"
    print(ROOT)
    print(f"ultralytics source: {ultralytics.__file__}")
    model = YOLO(ROOT / "ultralytics/cfg/models/v11/yolov11_gcblock4_original_fpsc_p4p5_neck.yaml")
    results = model.train(
        data=data_cfg,
        epochs=100,
        imgsz=640,
        batch=32,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        workers=4,
        device=0,
        project=ROOT / "runs",
        name="WasteSortingv3_yolov11_gcblock4_original_fpsc_p4p5_neck_alpha_ciou",
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
        bbox_loss="alpha_ciou",
        alpha_iou=3.0,
    )
    print(f"train_ok task={model.task} save_dir={results.save_dir}")


if __name__ == "__main__":
    main()
