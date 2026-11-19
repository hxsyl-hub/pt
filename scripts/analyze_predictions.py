# -*- coding: utf-8 -*-
"""
预测结果分析：对比 detect.py 的预测标签与 GT 标签，定位漏检、误检、低分预测和框偏移，
输出分类清单、可视化拼图，并可选用预测框修正 GT 标签。

常用命令（项目根执行）:
    python scripts/analyze_predictions.py                       # 分析验证集预测
    python scripts/analyze_predictions.py --iou 0.5 --low-score 0.5
    python scripts/analyze_predictions.py --apply               # 按建议修正 GT（自动备份原标签）

预期输入目录:
    runs/detect/val_pred/labels          由 scripts/predict_val.sh 生成（含置信度）
    label_dataset/dataset_2/val/labels   GT 标签
    label_dataset/dataset_2/val/images   原始图片
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # 按类别上色，缺省洋红

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def parse_label(path, with_conf=False):
    """返回 [(cls, cx, cy, w, h, conf)]"""
    boxes = []
    path = Path(path)
    if not path.exists():
        return boxes
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if with_conf:
            if len(parts) < 6:
                continue
            try:
                cls, cx, cy, w, h, conf = map(float, parts[:6])
            except ValueError:
                continue
            boxes.append((int(cls), cx, cy, w, h, conf))
        else:
            if len(parts) != 5:
                continue
            try:
                cls, cx, cy, w, h = map(float, parts)
            except ValueError:
                continue
            boxes.append((int(cls), cx, cy, w, h, 1.0))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, gts, iou_thr):
    """按 IoU 降序贪心匹配，类别必须一致。返回 (matched, 未匹配预测下标, 未匹配 GT 下标)"""
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p[0] != g[0]:
                continue
            v = iou(p, g)
            if v > 0:
                candidates.append((v, i, j))
    candidates.sort(reverse=True)

    used_p, used_g, matched = set(), set(), []
    for v, i, j in candidates:
        if v < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, v))
    return (matched,
            [i for i in range(len(preds)) if i not in used_p],
            [j for j in range(len(gts)) if j not in used_g])


def find_image(image_dir, stem):
    for ext in IMG_EXTS:
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


def write_label(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for b in boxes:
            f.write(f'{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n')


def draw(img, gt_boxes, pred_boxes, keep_gt=None, keep_pred=None):
    h, w = img.shape[:2]

    def box(x1y1x2y2, color, text):
        x1, y1, x2, y2 = x1y1x2y2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for k, b in enumerate(gt_boxes):
        if keep_gt is not None and k not in keep_gt:
            continue
        x1, y1 = int((b[1] - b[3] / 2) * w), int((b[2] - b[4] / 2) * h)
        x2, y2 = int((b[1] + b[3] / 2) * w), int((b[2] + b[4] / 2) * h)
        box((x1, y1, x2, y2), (0, 255, 0), f'GT {b[0]}')
    for k, b in enumerate(pred_boxes):
        if keep_pred is not None and k not in keep_pred:
            continue
        x1, y1 = int((b[1] - b[3] / 2) * w), int((b[2] - b[4] / 2) * h)
        x2, y2 = int((b[1] + b[3] / 2) * w), int((b[2] + b[4] / 2) * h)
        box((x1, y1, x2, y2), (0, 0, 255), f'P {b[0]} {b[5]:.2f}')
    return img


def make_sheet(items, out_path, cols=4, thumb=400):
    """items: [(name, img)]"""
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


def apply_fixes(fix_rows, gt_dir, backup_dir):
    """按建议把 GT 框替换为预测框，原标签备份到 backup_dir"""
    by_file = defaultdict(list)
    for row in fix_rows:
        by_file[row['image']].append(row)

    changed = 0
    for name, rows in by_file.items():
        gt_path = gt_dir / (Path(name).stem + '.txt')
        if not gt_path.exists():
            continue
        gts = parse_label(gt_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_path, backup_dir / gt_path.name)

        for row in rows:
            target = (int(row['class']), float(row['gt_cx']), float(row['gt_cy']),
                      float(row['gt_w']), float(row['gt_h']))
            idx = max(range(len(gts)), key=lambda k: iou(gts[k], target)) if gts else None
            if idx is None or iou(gts[idx], target) < 0.99:
                continue
            gts[idx] = (int(row['class']), float(row['new_cx']), float(row['new_cy']),
                        float(row['new_w']), float(row['new_h']), 1.0)
            changed += 1
        write_label(gt_path, gts)
    print(f'已修正 {changed} 个框，原标签备份在 {backup_dir}')


def main():
    parser = argparse.ArgumentParser(description='对比预测与 GT，输出漏检/误检/低分/框偏清单')
    parser.add_argument('--pred-dir', default=str(PROJECT_ROOT / 'runs/detect/val_pred/labels'))
    parser.add_argument('--gt-dir', default=str(PROJECT_ROOT / 'label_dataset/dataset_2/val/labels'))
    parser.add_argument('--image-dir', default=str(PROJECT_ROOT / 'label_dataset/dataset_2/val/images'))
    parser.add_argument('--out', default=str(PROJECT_ROOT / 'artifacts/预测分析/val_pred'))
    parser.add_argument('--iou', type=float, default=0.5, help='匹配用的 IoU 阈值')
    parser.add_argument('--low-score', type=float, default=0.5, help='低于该置信度单独列出')
    parser.add_argument('--fix-min', type=float, default=0.10, help='框偏修正的最小 IoU')
    parser.add_argument('--sheet-limit', type=int, default=20, help='可视化拼图最多张数')
    parser.add_argument('--apply', action='store_true', help='按建议修正 GT，原标签自动备份')
    args = parser.parse_args()

    pred_dir, gt_dir, image_dir = Path(args.pred_dir), Path(args.gt_dir), Path(args.image_dir)
    out = Path(args.out)
    for d in ('false_negatives', 'false_positives', 'low_score'):
        (out / d / 'images').mkdir(parents=True, exist_ok=True)
        (out / d / 'prelabels').mkdir(parents=True, exist_ok=True)
        (out / d / 'gtlabels').mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob('*.txt'))
    if not gt_files:
        print(f'GT 目录为空: {gt_dir}')
        return 1
    if not pred_dir.is_dir():
        print(f'预测目录不存在: {pred_dir}，先运行 bash scripts/predict_val.sh')
        return 1

    stats = Counter()
    per_image = []
    fix_rows = []
    fn_items, fp_items, low_items = [], [], []

    for gt_path in gt_files:
        stem = gt_path.stem
        gts = parse_label(gt_path)
        preds = parse_label(pred_dir / (stem + '.txt'), with_conf=True)
        matched, unmatched_p, unmatched_g = match_boxes(preds, gts, args.iou)

        tp = len(matched)
        fn_idx = list(unmatched_g)
        fp_idx = [i for i in unmatched_p if preds[i][5] >= args.low_score]
        low_idx = [i for i in range(len(preds)) if preds[i][5] < args.low_score]

        # 框偏：未匹配的 GT 找同类预测最高 IoU，落在 [fix_min, iou) 视为标注框不准
        suspicious = []
        for j in fn_idx:
            best = max(((iou(preds[i], gts[j]), i) for i in range(len(preds))
                        if preds[i][0] == gts[j][0]), default=(0.0, -1))
            if args.fix_min <= best[0] < args.iou:
                suspicious.append((j, best[1], best[0]))
                if best[1] >= 0:
                    p = preds[best[1]]
                    fix_rows.append({
                        'image': stem, 'class': gts[j][0],
                        'gt_cx': f'{gts[j][1]:.6f}', 'gt_cy': f'{gts[j][2]:.6f}',
                        'gt_w': f'{gts[j][3]:.6f}', 'gt_h': f'{gts[j][4]:.6f}',
                        'new_cx': f'{p[1]:.6f}', 'new_cy': f'{p[2]:.6f}',
                        'new_w': f'{p[3]:.6f}', 'new_h': f'{p[4]:.6f}',
                        'iou': f'{best[0]:.4f}',
                    })

        stats['gt'] += len(gts)
        stats['pred'] += len(preds)
        stats['tp'] += tp
        stats['fn'] += len(fn_idx)
        stats['fp'] += len(fp_idx)
        stats['low_score'] += len(low_idx)
        stats['suspicious_box'] += len(suspicious)
        per_image.append([stem, len(gts), len(preds), tp, len(fn_idx), len(fp_idx),
                          len(low_idx), len(suspicious)])

        img_path = find_image(image_dir, stem)
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR) if img_path else None

        def dump(kind, sel_g, sel_p):
            if img is None:
                return
            for d in (f'{kind}/images', f'{kind}/prelabels', f'{kind}/gtlabels'):
                (out / d).mkdir(parents=True, exist_ok=True)
            ext = img_path.suffix
            dst = out / kind / 'images' / (stem + ext)
            if not dst.exists():
                shutil.copy2(img_path, dst)
            write_label(out / kind / 'prelabels' / (stem + '.txt'), [preds[i] for i in sel_p])
            write_label(out / kind / 'gtlabels' / (stem + '.txt'), [gts[j] for j in sel_g])
            return img

        if fn_idx:
            vis = dump('false_negatives', fn_idx, unmatched_p)
            if vis is not None:
                fn_items.append((stem, draw(vis.copy(), gts, preds, keep_gt=fn_idx, keep_pred=unmatched_p)))
        if fp_idx:
            vis = dump('false_positives', [], fp_idx + low_idx)
            if vis is not None:
                fp_items.append((stem, draw(vis.copy(), gts, preds, keep_pred=fp_idx + low_idx)))
        if low_idx:
            vis = dump('low_score', [], low_idx)
            if vis is not None:
                low_items.append((stem, draw(vis.copy(), gts, preds, keep_pred=low_idx)))

    summary = dict(stats)
    summary.update({
        'images': len(gt_files),
        'precision_est': round(stats['tp'] / max(stats['tp'] + stats['fp'], 1), 4),
        'recall_est': round(stats['tp'] / max(stats['tp'] + stats['fn'], 1), 4),
        'iou_threshold': args.iou,
        'low_score_threshold': args.low_score,
    })
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    with open(out / 'detail.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'gt', 'pred', 'tp', 'fn', 'fp', 'low_score', 'suspicious_box'])
        w.writerows(per_image)

    if fix_rows:
        with open(out / 'fix_suggestions.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(fix_rows[0]))
            w.writeheader()
            w.writerows(fix_rows)

    print(f'图片 {summary["images"]}，GT {stats["gt"]}，预测 {stats["pred"]}')
    print(f'TP {stats["tp"]}，漏检 FN {stats["fn"]}，误检 FP {stats["fp"]}，'
          f'低分预测 {stats["low_score"]}，疑似框偏 {stats["suspicious_box"]}')
    print(f'估算 P {summary["precision_est"]:.4f} / R {summary["recall_est"]:.4f}')

    make_sheet(fn_items[:args.sheet_limit], out / 'false_negatives_sheet.jpg')
    make_sheet(fp_items[:args.sheet_limit], out / 'false_positives_sheet.jpg')
    make_sheet(low_items[:args.sheet_limit], out / 'low_score_sheet.jpg')
    print(f'结果目录: {out}')

    if args.apply and fix_rows:
        apply_fixes(fix_rows, gt_dir, out / 'labels_backup')
    elif fix_rows:
        print(f'框偏建议 {len(fix_rows)} 条，见 {out / "fix_suggestions.csv"}；'
              f'确认后加 --apply 写回 GT')
    return 0


if __name__ == '__main__':
    sys.exit(main())
