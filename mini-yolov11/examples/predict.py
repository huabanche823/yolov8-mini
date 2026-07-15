from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main():
    print(ROOT)
    model = YOLO(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml")
    results = model.predict(
        source=ROOT / "ultralytics/assets/bus.jpg",
        imgsz=64,
        device="cpu",
        save=True,
        verbose=False,
    )
    print(f"predict_ok images={len(results)} task={model.task}")


if __name__ == "__main__":
    main()
