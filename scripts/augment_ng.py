# -*- coding: utf-8 -*-
"""
对训练集中的缺陷类（默认 class 0 = ng）做缩放增强，模拟小目标和全目标。

做法：
  1. 只处理 train 中指定类别的图片，按 --scales 缩放整图（标签是归一化坐标，无需改动）；
  2. 可选 HSV 颜色扰动；
  3. 生成的文件以 `_aug_s<比例>` 后缀写入同一 train 目录，另存清单便于回滚。

默认只预览（统计 + 抽样拼图），加 --execute 才写文件：
    python scripts/augment_ng.py                       # 预览
    python scripts/augment_ng.py --execute             # 执行
    python scripts/augment_ng.py --remove artifacts/增强_ng_xxx/added_manifest.csv   # 回滚
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def imread_unicode(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(path))
    return ok


def parse_label(path):
    boxes = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            boxes.append((int(float(parts[0])), *map(float, parts[1:])))
        except ValueError:
            continue
    return boxes


def jitter_hsv(img, strength, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10) * strength) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - 0.3 * strength, 1 + 0.3 * strength), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(1 - 0.3 * strength, 1 + 0.3 * strength), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def collect_targets(split_dir, classes):
    image_dir, label_dir = split_dir / 'images', split_dir / 'labels'
    targets = []
    for label_path in sorted(label_dir.glob('*.txt')):
        boxes = parse_label(label_path)
        if not boxes or not any(b[0] in classes for b in boxes):
            continue
        stem = label_path.stem
        image_path = next((image_dir / (stem + ext) for ext in IMG_EXTS
                           if (image_dir / (stem + ext)).exists()), None)
        if image_path is not None:
            targets.append((image_path, label_path, boxes))
    return targets


def make_sheet(items, out_path, cols=4, thumb=400):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * thumb, cols * thumb, 3), 255, np.uint8)
    for i, (name, img) in enumerate(items):
        h, w = img.shape[:2]
        s = min(thumb / w, thumb / h)
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)))
        y, x = (i // cols) * thumb, (i % cols) * thumb
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(canvas, name, (x + 4, y + thumb - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(out_path))
        print(f'  已保存: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='对指定类别做缩放增强，写入 train（带清单可回滚）')
    parser.add_argument('--root', default=str(PROJECT_ROOT / 'label_dataset' / 'dataset_2'))
    parser.add_argument('--split', default='train')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='要增强的类别，默认 0（ng）')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.5, 1.5], help='缩放比例')
    parser.add_argument('--hsv', type=float, default=0.5, help='颜色扰动强度 0~1，0 表示关闭')
    parser.add_argument('--min-box-px', type=float, default=8.0, help='缩小后框边长小于该值则跳过')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sheet', type=int, default=16, help='抽样拼图张数')
    parser.add_argument('--execute', action='store_true', help='真正写文件')
    parser.add_argument('--out', default=None, help='清单与拼图目录，默认 artifacts/增强_ng_时间戳')
    parser.add_argument('--remove', metavar='MANIFEST', help='按清单删除已生成的增强文件')
    args = parser.parse_args()

    if args.remove:
        rows = list(csv.DictReader(open(args.remove, encoding='utf-8-sig')))
        removed = 0
        for row in rows:
            p = Path(row['path'])
            if p.exists():
                p.unlink()
                removed += 1
        print(f'已删除 {removed} 个增强文件')
        return 0

    split_dir = Path(args.root) / args.split
    targets = collect_targets(split_dir, set(args.classes))
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'artifacts' / f'增强_ng_{datetime.now():%Y%m%d_%H%M%S}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'数据: {split_dir}')
    print(f'增强类别: {args.classes}，缩放: {args.scales}，颜色扰动: {args.hsv}')
    print(f'可增强图片: {len(targets)} 张，预计新增 {len(targets) * len(args.scales)} 张')
    print(f'模式: {"执行" if args.execute else "预览（加 --execute 才写文件）"}')

    rng = np.random.default_rng(args.seed)
    created, sheet_items, skipped = [], [], 0

    for image_path, label_path, boxes in targets:
        img = imread_unicode(image_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        for scale in args.scales:
            nw, nh = int(round(w * scale)), int(round(h * scale))
            if nw < 32 or nh < 32:
                skipped += 1
                continue
            if any(b[3] * nw < args.min_box_px or b[4] * nh < args.min_box_px for b in boxes):
                skipped += 1
                continue
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            new_img = cv2.resize(img, (nw, nh), interpolation=interp)
            if args.hsv > 0:
                new_img = jitter_hsv(new_img, args.hsv, rng)
            stem = f'{image_path.stem}_aug_s{int(round(scale * 100))}'
            dst_img = split_dir / 'images' / (stem + '.jpg')
            dst_lab = split_dir / 'labels' / (stem + '.txt')
            if args.execute:
                if dst_img.exists():
                    continue
                imwrite_unicode(dst_img, new_img)
                dst_lab.write_text(label_path.read_text(encoding='utf-8'), encoding='utf-8')
                created.append({'path': str(dst_img)})
                created.append({'path': str(dst_lab)})
                if len(sheet_items) < args.sheet:
                    sheet_items.append((stem, new_img))
            elif len(sheet_items) < args.sheet:
                sheet_items.append((stem, new_img))

    if args.execute and created:
        manifest = out_dir / 'added_manifest.csv'
        with open(manifest, 'w', newline='', encoding='utf-8-sig') as f:
            row = csv.DictWriter(f, fieldnames=['path'])
            row.writeheader()
            row.writerows(created)
        print(f'已新增 {len(created) // 2} 组图片+标签，清单: {manifest}')
        print(f'回滚命令: python scripts/augment_ng.py --remove {manifest}')
    else:
        print(f'预览完成，跳过 {skipped} 个缩放后过小的框')

    make_sheet(sheet_items, out_dir / 'augment_preview.jpg')
    return 0


if __name__ == '__main__':
    sys.exit(main())
