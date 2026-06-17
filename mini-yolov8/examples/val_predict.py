from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "datasets/S2TLD/data.yaml"
DEFAULT_SOURCE = ROOT / "datasets/S2TLD/images/val"
DEFAULT_MODEL_CFG = ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"


def find_default_weights() -> Path:
    """Return a trained weight if one exists, otherwise fall back to model cfg."""
    preferred = [
        ROOT / "runs/s2tld-yolov8n/weights/best.pt",
        ROOT / "runs/s2tld-yolov8n/weights/last.pt",
        ROOT / "runs/SSGD-yolov8n/weights/best.pt",
        ROOT / "runs/SSGD-yolov8n/weights/last.pt",
    ]
    for weight in preferred:
        if weight.exists():
            return weight

    weights = sorted((ROOT / "runs").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return weights[0] if weights else DEFAULT_MODEL_CFG


def parse_args():
    parser = ArgumentParser(description="Validate YOLOv8 and save inference images on the validation set.")
    parser.add_argument("--weights", type=Path, default=find_default_weights(), help="Path to best.pt/last.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset yaml path.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Images or folder used for prediction.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/validation image size.")
    parser.add_argument("--device", default="cpu", help='Device, e.g. "cpu" or "0".')
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs", help="Directory to save results.")
    parser.add_argument("--name", default="s2tld-val-predict", help="Result folder name.")
    parser.add_argument("--save-txt", action="store_true", help="Also save predictions as YOLO txt labels.")
    parser.add_argument("--save-conf", action="store_true", help="Write confidence scores into txt labels.")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    val_save_dir = args.project / f"{args.name}-metrics"
    predict_save_dir = args.project / f"{args.name}-images"

    print(f"ROOT: {ROOT}")
    print(f"weights: {args.weights}")
    print(f"data: {args.data}")
    print(f"source: {args.source}")

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=f"{args.name}-metrics",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    print(f"val_ok save_dir={val_save_dir}")
    if hasattr(metrics, "box"):
        print(f"mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=f"{args.name}-images",
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=False,
    )
    print(f"predict_ok images={len(results)} save_dir={predict_save_dir}")


if __name__ == "__main__":
    main()
