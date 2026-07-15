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
    # 加载模型配置文件或权重文件，若没有权重文件，会自动下载预训练权重
    # 改进不要加载预训练权重
    model = YOLO(ROOT / "ultralytics/cfg/models/v11/yolov11_lsk.yaml") 
    results = model.train(
        data=ROOT / "datasets/WasteSortingv3/data_challenge.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=32,   
        lr0=0.01,  # 按缩放规则：0.01 * 32/16 = 0.02
        momentum=0.937,
        workers=4,  # 数据加载线程数，0 表示不使用子进程
        device=0,  # 使用 GPU 进行训练
        project=ROOT / "runs",  # 训练结果保存目录的父目录
        name="WasteSortingv3_yolov11_lsk",  # 本次训练的实验名称
        exist_ok=True,  # 如果目录已存在，允许覆盖
        pretrained=False,  # 不使用预训练权重
        val=True,  # 训练时进行验证
        plots=True,  # 生成训练图表
        verbose=True, # 关闭详细日志输出
        amp=False,
        optimizer="SGD",
        patience=20,     # 连续30个epoch验证集mAP50-95无有效提升，则触发早停停止训练
        seed=1,
        deterministic=False,
        # bbox_loss="shape_iou",
        # shape_iou_scale=0.0,
        # mosaic=1.0,
        # close_mosaic=10,
        # mixup=0.5,
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # multi_scale=0.0,
        # cls_pw=0.3,
        # minority_oversample=True
    )
    print(f"train_ok task={model.task} save_dir={results.save_dir}")


if __name__ == "__main__":
    main()


"""

YOLOv11_ddfm_lite summary (fused): 140 layers, 2,573,567 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.1s
                   all        355       1894      0.935      0.838      0.909      0.643
                 Brick        219        399      0.975       0.88      0.935      0.686
              Concrete        217        522      0.967       0.91      0.978      0.741
        Plastic Bottle        129        231      0.925      0.801      0.891      0.569
         Reinforcement        174        312      0.938      0.705      0.831      0.555
                timber        192        430      0.869      0.895       0.91      0.665


"""