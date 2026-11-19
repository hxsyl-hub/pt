# -*- coding: utf-8 -*-
"""
用 Ultralytics 训练 YOLO11 基线，超参与增强设置与 yolov5 基线保持一致，便于对比。

依赖（云端首次安装）:
    pip install ultralytics

用法（项目根执行）:
    python scripts/train_yolov11.py                      # 默认 yolo11s，50 轮，关闭全部增强
    python scripts/train_yolov11.py --model yolo11n.pt   # 换模型规格
    python scripts/train_yolov11.py --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# 与 configs/hyp_baseline_none.yaml 保持一致：关闭全部在线增强
ENHANCE_OFF = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    erasing=0.0, auto_augment=None, close_mosaic=0,
)


def build_data_yaml(dataset_root, out_path):
    """Ultralytics 对相对 path 的解析依赖全局 datasets_dir，这里写绝对路径最稳妥"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: ng\n"
        "  1: ok\n"
    )
    out_path.write_text(text, encoding='utf-8')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='训练 YOLO11 基线（关闭增强，与 yolov5 基线同条件）')
    parser.add_argument('--model', default='yolo11s.pt', help='Ultralytics 模型或权重')
    parser.add_argument('--data-root', default=str(PROJECT_ROOT / 'label_dataset' / 'dataset_2'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='base_v11s_none')
    parser.add_argument('--project', default=str(PROJECT_ROOT / 'runs'))
    parser.add_argument('--lr0', type=float, default=0.001)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print('未安装 ultralytics，请先在云端执行: pip install ultralytics')
        return 1

    data_root = Path(args.data_root)
    if not (data_root / 'train' / 'images').is_dir():
        print(f'数据集目录不存在: {data_root}')
        return 1

    data_yaml = build_data_yaml(data_root, Path(args.project) / f'{args.name}_data.yaml')
    print(f'数据配置: {data_yaml}')
    print(f'模型: {args.model}，epochs={args.epochs}，batch={args.batch}，imgsz={args.imgsz}，lr0={args.lr0}')
    print('增强: 全部关闭（与 yolov5 基线一致）')

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        optimizer='Adam',
        lr0=args.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        label_smoothing=0.0,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        **ENHANCE_OFF,
    )

    print(f'训练完成，结果目录: {Path(args.project) / args.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
