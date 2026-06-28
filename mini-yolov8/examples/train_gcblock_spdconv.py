from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    model_cfg = ROOT / "ultralytics/cfg/models/v11/yolov11_gcblock_spdconv.yaml"
    data_cfg = WORKSPACE / "data_challenge.yaml"
    if not data_cfg.exists():
        data_cfg = ROOT / "datasets/WasteSortingv3/data_challenge.yaml"

    print(ROOT)
    print(f"ultralytics source: {ultralytics.__file__}")
    print(f"model: {model_cfg}")
    print(f"data: {data_cfg}")

    model = YOLO(model_cfg)
    results = model.train(
        data=data_cfg,
        epochs=100,
        imgsz=640,
        batch=32,
        lr0=0.01,
        momentum=0.937,
        workers=4,
        device=0,
        project=WORKSPACE / "runs",
        name="WasteSortingv3_yolov11_gcblock_spdconv",
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
