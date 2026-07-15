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
    model = YOLO(ROOT / "ultralytics/cfg/models/v11/yolov11_linerefine_p4_neck.yaml") 
    results = model.train(
        data=ROOT / "datasets/WasteSortingv3/data_challenge.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=32,   
        lr0=0.01,  # 按缩放规则：0.01 * 32/16 = 0.02
        momentum=0.937,
        weight_decay=0.0005,
        workers=4,  # 数据加载线程数，0 表示不使用子进程
        device=0,  # 使用 GPU 进行训练
        project=ROOT / "runs",  # 训练结果保存目录的父目录
        name="WasteSortingv3_yolov11_linerefine_p4_neck",  # 本次训练的实验名称
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
        # bbox_loss="mpdiou", #指定IOUloss
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
YOLOv11n
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.7it/s 3.5s
                   all        355       1894      0.857      0.775      0.838      0.592
                 Brick        219        399      0.925      0.822       0.91      0.669
              Concrete        217        522       0.97      0.811      0.933      0.708
        Plastic Bottle        129        231      0.698      0.784      0.739      0.468
         Reinforcement        174        312      0.902      0.646      0.756      0.499
                timber        192        430      0.791      0.814      0.852      0.617
Speed: 0.1ms preprocess, 0.9ms inference, 0.0ms loss, 0.7ms postprocess per image

YOLOv11n_dsconv
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.4it/s 4.2s
                   all        355       1894      0.864      0.775      0.845      0.597
                 Brick        219        399      0.949       0.83       0.92      0.672
              Concrete        217        522      0.975      0.803      0.939      0.704
        Plastic Bottle        129        231       0.72      0.784      0.751      0.493
         Reinforcement        174        312      0.882      0.641      0.761       0.49
                timber        192        430      0.795      0.819      0.853      0.625
Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 0.7ms postprocess per image


YOLOv11_dsconv_lsk summary (fused): 158 layers, 2,757,589 parameters, 0 gradients, 6.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.6it/s 3.9s
                   all        355       1894      0.856       0.79      0.847      0.585
                 Brick        219        399       0.95      0.854      0.916      0.664
              Concrete        217        522      0.958      0.822      0.929       0.68
        Plastic Bottle        129        231       0.75      0.765      0.792      0.497
         Reinforcement        174        312      0.877      0.676      0.757      0.488
                timber        192        430      0.744      0.833      0.843      0.594
Speed: 2.2ms preprocess, 1.6ms inference, 0.0ms loss, 2.6ms postprocess per image

YOLOv11_dsconv_rescbam_p5 summary (fused): 149 layers, 2,680,681 parameters, 0 gradients, 6.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.7it/s 3.6s
                   all        355       1894      0.893       0.83      0.893      0.633
                 Brick        219        399      0.946      0.871      0.937      0.679
              Concrete        217        522      0.977      0.893      0.971      0.739
        Plastic Bottle        129        231      0.772      0.818       0.84      0.557
         Reinforcement        174        312      0.888      0.714      0.815      0.531
                timber        192        430      0.882      0.854      0.904      0.657
Speed: 2.3ms preprocess, 1.4ms inference, 0.0ms loss, 2.1ms postprocess per image

YOLOv11 summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.6s/it 9.8s
                   all        355       1894      0.928      0.844      0.912       0.65
                 Brick        219        399      0.965      0.885      0.937       0.69
              Concrete        217        522      0.977      0.908      0.966      0.748
        Plastic Bottle        129        231      0.911      0.842      0.904      0.586
         Reinforcement        174        312      0.909      0.704      0.833      0.555
                timber        192        430      0.879      0.882      0.917      0.673
Speed: 3.0ms preprocess, 2.0ms inference, 0.0ms loss, 5.1ms postprocess per image
Results saved to /root/yolov8-mini/mini-yolov8/runs/WasteSortingv3_yolov11_challenge



YOLOv11_gcblock summary (fused): 113 layers, 2,599,897 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 3.0it/s 2.0s
                   all        355       1894      0.917      0.856      0.914      0.646
                 Brick        219        399      0.949      0.925      0.961      0.697
              Concrete        217        522       0.98      0.928      0.981      0.757
        Plastic Bottle        129        231      0.878      0.813      0.873      0.556
         Reinforcement        174        312      0.892      0.718      0.837       0.55
                timber        192        430      0.889      0.893       0.92      0.668
Speed: 0.9ms preprocess, 1.0ms inference, 0.0ms loss, 0.7ms postprocess per image


"""