from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


EXPERIMENTS = {
    "baseline": {
        "name": "WasteSortingv3_yolov11n_baseline",
        "assigner": "taskaligned",
        "cls_loss": "bce",
    },
    "simota": {
        "name": "WasteSortingv3_yolov11n_simota",
        "assigner": "simota",
        "cls_loss": "bce",
    },
    "focal": {
        "name": "WasteSortingv3_yolov11n_focal",
        "assigner": "taskaligned",
        "cls_loss": "focal",
    },
    "simota_focal": {
        "name": "WasteSortingv3_yolov11n_simota_focal",
        "assigner": "simota",
        "cls_loss": "focal",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS, default="simota_focal")
    return parser.parse_args()


def main():
    args = parse_args()
    exp = EXPERIMENTS[args.exp]
    print(ROOT)
    print(f"ultralytics source: {ultralytics.__file__}")
    print(f"running ablation: {args.exp} -> {exp}")

    model = YOLO(ROOT / "ultralytics/cfg/models/v11/yolov11.yaml")
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
        name=exp["name"],
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
        assigner=exp["assigner"],
        cls_loss=exp["cls_loss"],
        focal_gamma=1.5,
        focal_alpha=0.25,
        simota_cls_weight=1.0,
        simota_iou_weight=3.0,
        simota_center_radius=0.0,
    )
    print(f"train_ok task={model.task} save_dir={results.save_dir}")


if __name__ == "__main__":
    main()
