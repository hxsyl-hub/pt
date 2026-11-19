# -*- coding: utf-8 -*-
"""
按类别、样本量比例和框尺寸筛选训练数据：把不需要的图片和标签移动到备份目录（不删除，可回滚）。

默认规则（方案 D）：
  1. 主类别（ng，class 0）全部保留；
  2. 对 --downsample-class（默认 ok，class 1）随机降采样，保留数量 = 主类别数量 × ratio；
  3. 在保留下来的图片里移出框尺寸超限的：letterbox 640 后面积 < 32x32 或 >= 128x128，
     一张图只要有任意一个框超限就整图移出（YOLO 按整图训练，无法只删框）。

默认只预览统计，加 --execute 才真正移动：
    python filter_dataset.py                 # 预览
    python filter_dataset.py --execute       # 执行移动
    python filter_dataset.py --restore artifacts/筛选_xxx/moved_manifest.csv   # 回滚
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def parse_label(label_path):
    """返回 [(cls, xc, yc, w, h)]，脏行跳过并计数"""
    boxes, bad = [], 0
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                bad += 1
                continue
            try:
                boxes.append((int(float(parts[0])), *map(float, parts[1:])))
            except ValueError:
                bad += 1
    return boxes, bad


def find_image(image_dir, stem):
    for ext in IMG_EXTS:
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


def letterbox_scale(w, h, img_size):
    return min(img_size / w, img_size / h)


def scan_split(split_dir, img_size):
    """扫描一个划分，返回 [(stem, boxes, areas, bad, image_path, label_path)]"""
    image_dir, label_dir = split_dir / 'images', split_dir / 'labels'
    records = []
    for label_path in sorted(label_dir.glob('*.txt')):
        image_path = find_image(image_dir, label_path.stem)
        boxes, bad = parse_label(label_path)
        if image_path is None or not boxes:
            records.append((label_path.stem, boxes, [], bad, image_path, label_path))
            continue
        w, h = Image.open(image_path).size
        scale = letterbox_scale(w, h, img_size)
        areas = [(bw * w * scale) * (bh * h * scale) for _, _, _, bw, bh in boxes]
        records.append((label_path.stem, boxes, areas, bad, image_path, label_path))
    return records


def decide(record, move_classes, min_area, max_area, size_classes=None):
    """返回 (reason, 说明)；reason 为 None 表示保留

    size_classes 为 None 时对所有类别应用尺寸规则；否则只检查这些类别的框。
    """
    stem, boxes, areas, bad, image_path, _ = record
    if image_path is None:
        return 'missing_image', '没有对应图片'
    if not boxes:
        return 'empty_or_bad_label', f'空标签或脏行({bad})'
    if any(b[0] in move_classes for b in boxes):
        return 'class', f'含类别 {sorted({b[0] for b in boxes if b[0] in move_classes})}'
    checked = [a for b, a in zip(boxes, areas) if size_classes is None or b[0] in size_classes]
    if not checked:
        return None, ''
    if any(a < min_area for a in checked):
        return 'size_small', f'最小框面积 {min(checked):.0f}'
    if any(a >= max_area for a in checked):
        return 'size_large', f'最大框面积 {max(checked):.0f}'
    return None, ''


def move_pair(record, backup_split_dir, reason):
    stem, _, _, _, image_path, label_path = record
    sub = {'downsample': 'removed_by_downsample',
           'class': 'removed_by_class'}.get(reason, 'removed_by_size')
    dst_img_dir = backup_split_dir / sub / 'images'
    dst_lab_dir = backup_split_dir / sub / 'labels'
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    if image_path is not None:
        dst = dst_img_dir / image_path.name
        if dst.exists():
            dst = dst_img_dir / f'{image_path.stem}_{datetime.now():%H%M%S%f}{image_path.suffix}'
        shutil.move(str(image_path), str(dst))
        moved.append((image_path, dst))
    dst_lab = dst_lab_dir / label_path.name
    if dst_lab.exists():
        dst_lab = dst_lab_dir / f'{label_path.stem}_{datetime.now():%H%M%S%f}.txt'
    shutil.move(str(label_path), str(dst_lab))
    moved.append((label_path, dst_lab))
    return moved


def restore_manifest(manifest_path):
    """按 moved_manifest.csv 把文件搬回原位置"""
    manifest_path = Path(manifest_path)
    with open(manifest_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    restored, skipped, missing = 0, 0, 0
    for row in rows:
        src_img, dst_img = Path(row['src_image']), Path(row['dst_image'])
        src_lab = src_img.parent.parent / 'labels' / (src_img.stem + '.txt')
        dst_lab = dst_img.parent.parent / 'labels' / (dst_img.stem + '.txt')
        if src_img.exists():
            skipped += 1
            continue
        if not dst_img.exists():
            missing += 1
            continue
        src_img.parent.mkdir(parents=True, exist_ok=True)
        src_lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst_img), str(src_img))
        if dst_lab.exists():
            shutil.move(str(dst_lab), str(src_lab))
        restored += 1

    print(f'恢复完成: {restored} 组，已存在跳过 {skipped}，备份缺失 {missing}')
    return restored


def main():
    parser = argparse.ArgumentParser(description='按类别和框尺寸筛选数据（移动而非删除）')
    parser.add_argument('--root', default=str(PROJECT_ROOT / 'label_dataset' / 'dataset_2'))
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--move-classes', nargs='*', type=int, default=[],
                        help='整类移出的类别，默认不移出整类')
    parser.add_argument('--downsample-class', type=int, default=1,
                        help='需要降采样的类别，默认 1（ok）；设为 -1 关闭降采样')
    parser.add_argument('--downsample-ratio', type=float, default=1.0,
                        help='降采样保留数量 = 其它类别数量 × 该比例')
    parser.add_argument('--seed', type=int, default=42, help='降采样随机种子')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--min-area', type=float, default=32 * 32,
                        help='小于该面积(letterbox 后像素²)移出')
    parser.add_argument('--max-area', type=float, default=128 * 128,
                        help='大于等于该面积移出')
    parser.add_argument('--size-classes', nargs='*', type=int, default=None,
                        help='只对这些类别应用尺寸规则，默认全部类别；传空则不筛尺寸')
    parser.add_argument('--out', default=None, help='备份与日志目录，默认 output/筛选_时间戳')
    parser.add_argument('--execute', action='store_true', help='真正移动；不加只预览')
    parser.add_argument('--restore', metavar='MANIFEST', help='按 moved_manifest.csv 恢复数据后退出')
    args = parser.parse_args()

    if args.restore:
        return 0 if restore_manifest(args.restore) >= 0 else 1

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'artifacts' / f'筛选_{datetime.now():%Y%m%d_%H%M%S}'
    out_dir.mkdir(parents=True, exist_ok=True)

    size_cls = '全部' if args.size_classes is None else (args.size_classes or '不筛')
    print(f'数据根目录: {root}')
    print(f'整类移出: {args.move_classes or "无"}，'
          f'降采样类别: {args.downsample_class if args.downsample_class >= 0 else "关闭"}，'
          f'尺寸范围 [{args.min_area:.0f}, {args.max_area:.0f}) 像素²，尺寸规则类别: {size_cls}')
    print(f'模式: {"执行移动" if args.execute else "预览（加 --execute 才移动）"}')
    print(f'备份目录: {out_dir}\n')

    summary, manifest = {}, []
    for split in args.splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f'跳过不存在的划分: {split_dir}')
            continue

        records = scan_split(split_dir, args.img_size)
        class_before = Counter()
        for _, boxes, _, _, _, _ in records:
            for b in boxes:
                class_before[b[0]] += 1

        down_stems = set()
        if args.downsample_class >= 0:
            def has_ds(rec):
                return any(b[0] == args.downsample_class for b in rec[1])
            candidates = [r for r in records if r[1] and has_ds(r)]
            others = [r for r in records if r[1] and not has_ds(r)]
            target = min(int(len(others) * args.downsample_ratio), len(candidates))
            order = list(range(len(candidates)))
            random.Random(args.seed).shuffle(order)
            down_stems = {candidates[i][0] for i in order[target:]}
            print(f'[{split}] 降采样 class {args.downsample_class}: '
                  f'{len(candidates)} -> 保留 {target}（其它类别 {len(others)} 张）')

        plan = []
        reasons = Counter()
        for rec in records:
            if rec[0] in down_stems:
                plan.append((rec, 'downsample', f'降采样 class {args.downsample_class}'))
                reasons['downsample'] += 1
                continue
            reason, detail = decide(rec, args.move_classes, args.min_area, args.max_area,
                                    args.size_classes)
            if reason:
                plan.append((rec, reason, detail))
                reasons[reason] += 1

        kept = len(records) - len(plan)
        print(f'[{split}] 检查 {len(records)} 个标签文件，类别框数 {dict(sorted(class_before.items()))}')
        print(f'        计划移出 {len(plan)}：降采样 {reasons["downsample"]}，按类别 {reasons["class"]}，'
              f'大目标 {reasons["size_large"]}，小目标 {reasons["size_small"]}，'
              f'空标签/脏行 {reasons["empty_or_bad_label"]}，缺图 {reasons["missing_image"]}')
        print(f'        剩余 {kept} 张')
        for rec, reason, detail in plan[:8]:
            print(f'          预览: {rec[0]} -> {reason}（{detail}）')
        if len(plan) > 8:
            print(f'          ... 其余 {len(plan) - 8} 条见日志')

        if args.execute:
            for rec, reason, detail in plan:
                moved = move_pair(rec, out_dir / split, reason)
                manifest.append({
                    'split': split, 'image': rec[0], 'reason': reason, 'detail': detail,
                    'src_image': str(moved[0][0]), 'dst_image': str(moved[0][1]),
                })

        summary[split] = {
            'checked': len(records),
            'class_boxes_before': dict(sorted(class_before.items())),
            'moved_total': len(plan),
            'moved_by_reason': dict(reasons),
            'remaining': kept,
        }
        print()

    if args.execute and manifest:
        with open(out_dir / 'moved_manifest.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0]))
            w.writeheader()
            w.writerows(manifest)
        print(f'移动清单: {out_dir / "moved_manifest.csv"}')

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'root': str(root), 'args': vars(args), 'splits': summary},
                  f, ensure_ascii=False, indent=2)
    print(f'汇总: {out_dir / "summary.json"}')
    if not args.execute:
        print('当前是预览模式，确认无误后加 --execute 执行移动。')


if __name__ == '__main__':
    sys.exit(main())
