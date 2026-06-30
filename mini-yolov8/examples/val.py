from ultralytics import YOLO
import time

model = YOLO("/root/yolov8-mini/mini-yolov8/runs/WasteSortingv3_yolov11_challenge_shapeIoU/weights/best.pt")

results = model.val(
    data="/root/yolov8-mini/mini-yolov8/datasets/WasteSortingv3/data_challenge.yaml",
    split="test",
    imgsz=640,
    batch=1,
    device=0,
    verbose=True
)