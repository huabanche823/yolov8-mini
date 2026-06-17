from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main():
    print(ROOT)
    model = YOLO(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml") # 加载模型配置文件或权重文件，若没有权重文件，会自动下载预训练权重
    results = model.train(
        data=ROOT / "datasets/data.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=960,  # 输入图像大小
        batch=32,   
        lr0=0.001,  # 按缩放规则：0.01 * 32/16 = 0.02
        momentum=0.95,
        workers=4,  # 数据加载线程数，0 表示不使用子进程
        device=0,  # 使用 GPU 进行训练
        project=ROOT / "runs",  # 训练结果保存目录的父目录
        name="WARP-yolov8n",  # 本次训练的实验名称
        exist_ok=True,  # 如果目录已存在，允许覆盖
        pretrained=False,  # 不使用预训练权重
        val=True,  # 训练时进行验证
        plots=True,  # 生成训练图表
        verbose=True, # 关闭详细日志输出
        amp=False,
        optimizer="Adam"
    )
    print(f"train_ok task={model.task} save_dir={results.save_dir}")


if __name__ == "__main__":
    main()
