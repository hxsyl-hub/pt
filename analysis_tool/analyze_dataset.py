# -*- coding: utf-8 -*-
"""
数据集分布与质量分析：
  类别分布 / 标签分布（类别不平衡检查）
  框大小分布（小/中/大目标）、框宽高比分布、KMeans anchor
  图像质量（亮度/过曝）分析 + 过亮图片候选清单 + 标注核对图

用法（默认分析 label_dataset/dataset_2 的 train、val）：
    python analyze_dataset.py
    python analyze_dataset.py --splits train val test --quality-samples 0
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import warnings
from collections import Counter

import cv2
import numpy as np
import matplotlib
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def imread_unicode(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def list_images(image_dir):
    return sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(IMG_EXTS)
    )


def label_path_of(image_path, image_dir, label_dir):
    name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(label_dir, name + '.txt')


def load_boxes(label_file, stats=None):
    """返回 [(cls:int, xc, yc, w, h)]（归一化坐标）

    stats 传入 Counter/dict 时累计脏行与越界框数量，便于发现标签质量问题。
    """
    boxes = []
    if not os.path.exists(label_file):
        return boxes
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                if stats is not None:
                    stats['malformed'] += 1
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                if stats is not None:
                    stats['malformed'] += 1
                continue
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1.000001 and 0 < h <= 1.000001):
                if stats is not None:
                    stats['out_of_range'] += 1
                continue
            boxes.append((cls, xc, yc, w, h))
    return boxes


def image_size(path):
    """只读图像头拿宽高，不解码像素"""
    with Image.open(path) as im:
        return im.size


def letterbox_area_scale(img_w, img_h, img_size):
    """等比缩放后长边等于 img_size，与 YOLO letterbox 的缩放系数一致"""
    return min(img_size / img_w, img_size / img_h)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {path}')


# ---------------------------------------------------------------- 类别分布
def analyze_class_distribution(split_dir, out_dir, split):
    """类别分布 + 每图框数分布"""
    label_dir = os.path.join(split_dir, 'labels')
    class_counter = Counter()
    boxes_per_image = Counter()
    empty_files = []
    stats = Counter()

    for name in sorted(os.listdir(label_dir)):
        if not name.endswith('.txt'):
            continue
        boxes = load_boxes(os.path.join(label_dir, name), stats)
        boxes_per_image[len(boxes)] += 1
        if not boxes:
            empty_files.append(name)
        for b in boxes:
            class_counter[b[0]] += 1

    total = sum(class_counter.values())
    print(f'  类别框数: {dict(sorted(class_counter.items()))}，总框数 {total}')
    if total:
        for c, n in sorted(class_counter.items()):
            print(f'    class {c}: {n} 框 ({n / total * 100:.2f}%)')
    if empty_files:
        print(f'  空标签文件 {len(empty_files)} 个')
    label_issues = {k: v for k, v in stats.items() if v}
    if label_issues:
        print(f'  标签问题: {label_issues}')

    with open(os.path.join(out_dir, 'class_distribution.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'boxes', 'ratio'])
        for c, n in sorted(class_counter.items()):
            w.writerow([c, n, f'{n / total:.6f}' if total else 0])

    with open(os.path.join(out_dir, 'empty_labels.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(empty_files))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classes = [str(c) for c in sorted(class_counter)]
    counts = [class_counter[int(c)] for c in classes]
    bars = axes[0].bar(classes, counts, color='skyblue')
    for bar, n in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, n, str(n),
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('框数量')
    axes[0].set_title(f'{split} 类别分布')
    axes[0].grid(axis='y', alpha=0.3)

    bp_keys = sorted(boxes_per_image)
    axes[1].bar([str(k) for k in bp_keys], [boxes_per_image[k] for k in bp_keys], color='salmon')
    for i, k in enumerate(bp_keys):
        axes[1].text(i, boxes_per_image[k], str(boxes_per_image[k]),
                     ha='center', va='bottom', fontweight='bold')
    axes[1].set_xlabel('每张图片的框数量')
    axes[1].set_ylabel('图片数量')
    axes[1].set_title(f'{split} 每图框数分布')
    axes[1].grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'class_distribution.png'))

    return {
        'split': split,
        'class_boxes': dict(sorted(class_counter.items())),
        'total_boxes': total,
        'images': sum(boxes_per_image.values()),
        'empty_label_files': len(empty_files),
        'boxes_per_image': dict(sorted(boxes_per_image.items())),
        'label_issues': label_issues,
    }


# ---------------------------------------------------------------- 框尺寸与宽高比
SIZE_BINS = [
    ('<5x5', 0, 5 * 5),
    ('5x5~16x16', 5 * 5, 16 * 16),
    ('16x16~32x32', 16 * 16, 32 * 32),
    ('32x32~64x64', 32 * 32, 64 * 64),
    ('64x64~128x128', 64 * 64, 128 * 128),
    ('>=128x128', 128 * 128, float('inf')),
]


def analyze_box_sizes(split_dir, out_dir, split, img_size=640):
    """框大小分布（letterbox 到 img_size 后的像素尺寸，按面积分档）"""
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels')
    counts = {name: 0 for name, _, _ in SIZE_BINS}
    rows = []
    unreadable = []
    stats = Counter()

    for img_path in list_images(image_dir):
        try:
            w, h = image_size(img_path)
        except (OSError, ValueError):
            unreadable.append(os.path.basename(img_path))
            continue
        scale = letterbox_area_scale(w, h, img_size)
        for cls, xc, yc, bw, bh in load_boxes(label_path_of(img_path, image_dir, label_dir), stats):
            px_w, px_h = bw * w * scale, bh * h * scale
            area = px_w * px_h
            for name, lo, hi in SIZE_BINS:
                if lo <= area < hi:
                    counts[name] += 1
                    break
            rows.append([split, os.path.basename(img_path), cls, f'{w}x{h}',
                         f'{px_w:.2f}', f'{px_h:.2f}', f'{area:.1f}', f'{area ** 0.5:.2f}', name])

    for name, _, _ in SIZE_BINS:
        print(f'    {name}: {counts[name]}')
    label_issues = {k: v for k, v in stats.items() if v}
    if label_issues:
        print(f'    标签问题: {label_issues}')

    header = ['split', 'image', 'class', 'orig_size', 'box_w_px', 'box_h_px',
              'area_px', 'side_px', 'size_bin']
    with open(os.path.join(out_dir, 'box_size_detail.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # 训练时按 32/128 阈值挑出的小目标、超大目标，供下一步移出
    small = [r for r in rows if r[-1] in ('<5x5', '5x5~16x16', '16x16~32x32')]
    large = [r for r in rows if r[-1] == '>=128x128']
    for tag, sel in (('small_boxes', small), ('large_boxes', large)):
        with open(os.path.join(out_dir, f'{tag}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(sel)
    if unreadable:
        with open(os.path.join(out_dir, 'unreadable_images.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(unreadable))
    print(f'    超小目标(<32x32) {len(small)}，超大目标(>=128x128) {len(large)}')

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [n for n, _, _ in SIZE_BINS]
    vals = [counts[n] for n in names]
    bars = ax.bar(names, vals, color='skyblue')
    for bar, n in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, n, str(n), ha='center', va='bottom')
    ax.set_xlabel(f'目标尺寸（letterbox {img_size} 后的像素面积）')
    ax.set_ylabel('框数量')
    ax.set_title(f'{split} 框大小分布')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'box_size_distribution.png'))

    return {'split': split, 'size_counts': counts, 'total_boxes': len(rows),
            'tiny_or_small': len(small), 'huge': len(large)}


def analyze_ratio_distribution(split_dir, out_dir, split, num_clusters=9, img_size=640):
    """宽高比分布 + KMeans anchor（同时给出 letterbox 输入下的像素 anchor）"""
    label_dir = os.path.join(split_dir, 'labels')
    ws, hs = [], []
    label_stats = Counter()
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith('.txt'):
            continue
        for _, _, _, bw, bh in load_boxes(os.path.join(label_dir, name), label_stats):
            ws.append(bw)
            hs.append(bh)

    if not ws:
        print('    没有有效框，跳过')
        return {'split': split, 'boxes': 0}

    ws, hs = np.array(ws), np.array(hs)
    ratios = ws / hs
    stats = {
        'split': split, 'boxes': len(ws),
        'width_range': [float(ws.min()), float(ws.max())],
        'height_range': [float(hs.min()), float(hs.max())],
        'ratio_range': [float(ratios.min()), float(ratios.max())],
        'ratio_mean': float(ratios.mean()),
        'ratio_median': float(np.median(ratios)),
        'label_issues': {k: v for k, v in label_stats.items() if v},
    }
    print(f'    宽度 {ws.min():.4f}~{ws.max():.4f}  高度 {hs.min():.4f}~{hs.max():.4f}')
    print(f'    宽高比 {ratios.min():.2f}~{ratios.max():.2f}  平均 {ratios.mean():.2f}')

    boxes = np.stack([ws, hs], axis=1)
    n_clusters = min(num_clusters, len(boxes))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(boxes)
    anchors = km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0] * km.cluster_centers_[:, 1])]
    anchors_px = anchors * img_size
    stats['anchors_norm'] = [{'w': float(a[0]), 'h': float(a[1]), 'ratio': float(a[0] / a[1])}
                             for a in anchors]
    stats['anchors_px'] = [{'w': float(a[0]), 'h': float(a[1])} for a in anchors_px]
    print(f'    推荐 anchor（{n_clusters} 组，{img_size} 输入下的像素值）:')
    for i, a in enumerate(anchors_px):
        print(f'      Anchor {i + 1}: ({a[0]:.1f}, {a[1]:.1f})  宽高比 {a[0] / a[1]:.2f}')

    with open(os.path.join(out_dir, 'anchors.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{split}: {len(ws)} 个框\n')
        f.write(f'宽度 {ws.min():.6f}~{ws.max():.6f}，高度 {hs.min():.6f}~{hs.max():.6f}\n')
        f.write(f'宽高比 {ratios.min():.6f}~{ratios.max():.6f}，平均 {ratios.mean():.6f}\n\n')
        f.write('归一化 anchor:\n')
        for i, a in enumerate(anchors):
            f.write(f'  Anchor {i + 1}: ({a[0]:.6f}, {a[1]:.6f}), 宽高比 {a[0] / a[1]:.2f}\n')
        f.write(f'\nYOLOv5 anchors（{img_size} 输入下的像素值，按面积从 P3 到 P5 分 3 组）:\n')
        for head, group in zip(('P3', 'P4', 'P5'), np.array_split(anchors_px, 3)):
            vals = ', '.join(f'{a[0]:.0f},{a[1]:.0f}' for a in group)
            f.write(f'  {head}: {vals}\n')
        f.write(f'\nanchors: [{", ".join(f"{a[0]:.0f},{a[1]:.0f}" for a in anchors_px)}]\n')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(ratios, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('宽高比分布')
    axes[0, 0].set_xlabel('宽/高')

    axes[0, 1].hist(ws, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('归一化宽度分布')
    axes[0, 1].set_xlabel('宽度')

    axes[1, 0].hist(hs, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('归一化高度分布')
    axes[1, 0].set_xlabel('高度')

    axes[1, 1].scatter(ws, hs, s=8, alpha=0.4, color='purple')
    axes[1, 1].scatter(anchors[:, 0], anchors[:, 1], s=120, color='red', marker='x', label='anchor')
    axes[1, 1].set_title('宽高散点图 + KMeans anchor')
    axes[1, 1].set_xlabel('宽度')
    axes[1, 1].set_ylabel('高度')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(f'{split} 框比例分布')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'ratio_distribution.png'))

    return stats


# ---------------------------------------------------------------- 图像质量与标注核对
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}


def analyze_image_quality(split_dir, out_dir, split, num_samples=0, contact_sheet=0,
                          over_threshold=0.20):
    """亮度 / 过曝 / 对比度分析。num_samples=0 表示全部图片"""
    image_dir = os.path.join(split_dir, 'images')
    files = list_images(image_dir)
    if num_samples and num_samples < len(files):
        files = random.Random(42).sample(files, num_samples)
        files.sort()

    hist_sum = {'b': np.zeros(256), 'g': np.zeros(256), 'r': np.zeros(256)}
    gray_hist_sum = np.zeros(256)
    rows = []
    for i, img_path in enumerate(files, 1):
        img = imread_unicode(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        over = float((gray >= 240).mean())
        dark = float((gray <= 15).mean())
        for ch, key in zip(cv2.split(img), 'bgr'):
            hist_sum[key] += cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
        gray_hist_sum += cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        rows.append([os.path.basename(img_path), mean, std, over, dark])
        if i % 200 == 0:
            print(f'    已处理 {i}/{len(files)} 张')

    if not rows:
        print('    没有可读图片，跳过')
        return {'split': split, 'images': 0}

    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    overs = np.array([r[3] for r in rows])
    darks = np.array([r[4] for r in rows])
    pct = [10, 25, 50, 75, 90, 99]
    stats = {
        'split': split, 'images': len(rows),
        'brightness_mean': round(float(means.mean()), 2),
        'brightness_percentiles': {f'p{p}': round(float(np.percentile(means, p)), 2) for p in pct},
        'contrast_mean': round(float(stds.mean()), 2),
        'overexposed_ratio_percentiles': {f'p{p}': round(float(np.percentile(overs, p)), 4) for p in pct},
        'dark_ratio_percentiles': {f'p{p}': round(float(np.percentile(darks, p)), 4) for p in pct},
        f'suspected_overexposed_images(>={over_threshold:.0%}_pixels>=240)': int((overs >= over_threshold).sum()),
    }
    print(f'    平均亮度 {means.mean():.1f}（p10 {np.percentile(means, 10):.1f} / '
          f'p50 {np.percentile(means, 50):.1f} / p99 {np.percentile(means, 99):.1f}）')
    print(f'    过曝像素占比 p50 {np.percentile(overs, 50):.1%} / p90 {np.percentile(overs, 90):.1%} / '
          f'p99 {np.percentile(overs, 99):.1%}')
    print(f'    疑似过曝（>=240 像素占比 >= {over_threshold:.0%}）'
          f'{stats[f"suspected_overexposed_images(>={over_threshold:.0%}_pixels>=240)"]} 张')

    rows.sort(key=lambda r: -r[3])
    with open(os.path.join(out_dir, 'image_quality.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'brightness_mean', 'contrast_std', 'overexposed_ratio', 'dark_ratio'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.2f}', f'{r[2]:.2f}', f'{r[3]:.4f}', f'{r[4]:.4f}'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for key, color in (('b', 'blue'), ('g', 'green'), ('r', 'red')):
        h = hist_sum[key] / max(hist_sum[key].sum(), 1)
        axes[0, 0].plot(h, color=color, label=key.upper())
    axes[0, 0].set_title('BGR 累计直方图（归一化）')
    axes[0, 0].set_xlabel('像素值')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    axes[0, 1].plot(gray_hist_sum / max(gray_hist_sum.sum(), 1), color='black')
    axes[0, 1].set_title('灰度累计直方图（看是否偏右=过亮）')
    axes[0, 1].set_xlabel('像素值')
    axes[0, 1].set_yscale('log')

    axes[1, 0].hist(means, bins=50, color='orange', alpha=0.8, edgecolor='black')
    axes[1, 0].axvline(means.mean(), color='red', linestyle='--', label=f'均值 {means.mean():.1f}')
    axes[1, 0].set_title('图片平均亮度分布')
    axes[1, 0].set_xlabel('灰度均值')
    axes[1, 0].legend()

    axes[1, 1].hist(overs * 100, bins=50, color='crimson', alpha=0.8, edgecolor='black')
    axes[1, 1].axvline(over_threshold * 100, color='blue', linestyle='--',
                       label=f'阈值 {over_threshold:.0%}')
    axes[1, 1].set_title('过曝像素占比分布（>=240）')
    axes[1, 1].set_xlabel('过曝像素百分比 (%)')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(f'{split} 图像质量分析（{len(rows)} 张）')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'image_quality.png'))

    if contact_sheet:
        make_contact_sheet(image_dir, out_dir, [r[0] for r in rows[:contact_sheet]],
                           'overexposed_contact_sheet.jpg')
    return stats


def draw_boxes(img, boxes):
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = CLASS_COLORS.get(cls, (255, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(cls), (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def make_label_check_sheet(split_dir, out_dir, split, limit=20, cols=4, thumb=400, seed=0):
    """随机抽 limit 张图画上 GT 框，拼成网格供人工核对标注"""
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels')
    files = list_images(image_dir)
    sel = random.Random(seed).sample(files, min(limit, len(files)))
    selected = []
    for path in sel:
        img = imread_unicode(path)
        if img is None:
            continue
        draw_boxes(img, load_boxes(label_path_of(path, image_dir, label_dir)))
        selected.append((os.path.basename(path), img))
    make_contact_sheet(image_dir, out_dir, [s[0] for s in selected],
                       'label_check_sheet.jpg', cols=cols, thumb=thumb, preload=dict(selected))
    print(f'  标注核对图已保存（{len(selected)} 张）')


def make_contact_sheet(image_dir, out_dir, names, filename, cols=5, thumb=320, preload=None):
    """把指定图片拼成缩略图网格"""
    if not names:
        return
    n_rows = (len(names) + cols - 1) // cols
    canvas = np.full((n_rows * thumb, cols * thumb, 3), 255, np.uint8)
    for i, name in enumerate(names):
        img = preload.get(name) if preload and name in preload else imread_unicode(os.path.join(image_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(thumb / w, thumb / h)
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)))
        y, x = (i // cols) * thumb, (i % cols) * thumb
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(canvas, name, (x + 3, y + thumb - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    out = os.path.join(out_dir, filename)
    ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(out)
        print(f'  已保存: {out}')


def main():
    default_root = os.path.join(SCRIPT_DIR, '..', 'label_dataset', 'dataset_2')
    parser = argparse.ArgumentParser(description='数据集分布与图像质量分析')
    parser.add_argument('--root', default=os.path.normpath(default_root), help='数据集根目录（含 train/val/test）')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='要分析的划分')
    parser.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'output', '数据集分布与质量分析'), help='结果输出目录')
    parser.add_argument('--img-size', type=int, default=640, help='letterbox 目标尺寸')
    parser.add_argument('--clusters', type=int, default=9, help='KMeans anchor 簇数')
    parser.add_argument('--quality-samples', type=int, default=0, help='图像质量分析采样数，0=全部')
    parser.add_argument('--overexposed-threshold', type=float, default=0.20,
                        help='过曝判定阈值：>=240 像素的占比')
    parser.add_argument('--contact-sheet', type=int, default=20, help='生成过曝缩略图网格的张数，0=关闭')
    parser.add_argument('--label-check', type=int, default=20, help='生成 GT 标注核对网格的张数，0=关闭')
    args = parser.parse_args()

    summary = {'dataset_root': os.path.abspath(args.root), 'splits': {}}
    for split in args.splits:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f'跳过不存在的划分: {split_dir}')
            continue
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n===== [{split}] ====')
        print('类别分布')
        r_class = analyze_class_distribution(split_dir, out_dir, split)
        print('框大小分布')
        r_size = analyze_box_sizes(split_dir, out_dir, split, args.img_size)
        print('框宽高比分布 + anchor')
        r_ratio = analyze_ratio_distribution(split_dir, out_dir, split, args.clusters, args.img_size)
        print('图像质量分析')
        r_quality = analyze_image_quality(split_dir, out_dir, split, args.quality_samples,
                                          args.contact_sheet, args.overexposed_threshold)
        print('人工核对标注（GT 框可视化）')
        if args.label_check:
            make_label_check_sheet(split_dir, out_dir, split, args.label_check)
        summary['splits'][split] = {
            'class': r_class, 'size': r_size, 'ratio': r_ratio, 'quality': r_quality,
        }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n汇总已保存: {os.path.join(args.out, "summary.json")}')


if __name__ == '__main__':
    sys.exit(main())
