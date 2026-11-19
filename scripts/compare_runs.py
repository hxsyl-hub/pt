# -*- coding: utf-8 -*-
"""
对比多个训练结果的指标（兼容 YOLOv5 与 Ultralytics 的 results.csv 列名）。

用法:
    python scripts/compare_runs.py runs/base_v5s_none runs/base_v11s_none
    python scripts/compare_runs.py runs/*          # 自动展开
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

COLUMN_ALIASES = {
    'epoch': ['epoch'],
    'precision': ['metrics/precision', 'metrics/precision(B)'],
    'recall': ['metrics/recall', 'metrics/recall(B)'],
    'map50': ['metrics/mAP_0.5', 'metrics/mAP50', 'metrics/mAP50(B)'],
    'map5095': ['metrics/mAP_0.5:0.95', 'metrics/mAP50-95', 'metrics/mAP50-95(B)'],
    'train_box': ['train/box_loss'],
    'val_box': ['val/box_loss'],
}


def get(row, keys):
    for key in keys:
        for k, v in row.items():
            if k.strip() == key:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
    return None


def summarize(run_dir):
    csv_path = Path(run_dir) / 'results.csv'
    if not csv_path.exists():
        print(f'跳过（无 results.csv）: {run_dir}')
        return None
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    if not rows:
        print(f'跳过（空文件）: {run_dir}')
        return None

    def metric(row, name):
        return get(row, COLUMN_ALIASES[name])

    best = max(rows, key=lambda r: metric(r, 'map5095') or 0)
    last = rows[-1]
    return {
        'run': Path(run_dir).name,
        'epochs': len(rows),
        'last_epoch': int(metric(last, 'epoch') or len(rows) - 1),
        'best_epoch': int(metric(best, 'epoch') or 0),
        'last_p': metric(last, 'precision'),
        'last_r': metric(last, 'recall'),
        'last_map50': metric(last, 'map50'),
        'last_map5095': metric(last, 'map5095'),
        'best_map5095': metric(best, 'map5095'),
    }


def fmt(v, digits=4):
    return '-' if v is None else f'{v:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description='对比多个训练结果的指标')
    parser.add_argument('runs', nargs='+', help='训练结果目录（包含 results.csv）')
    args = parser.parse_args()

    results = [r for r in (summarize(d) for d in args.runs) if r]
    if not results:
        return 1

    header = ['run', 'epochs', 'best_ep', 'P', 'R', 'mAP50', 'mAP50-95', 'best mAP50-95']
    print(' | '.join(header))
    print(' | '.join('---' for _ in header))
    for r in results:
        print(' | '.join([
            r['run'], str(r['epochs']), str(r['best_epoch']),
            fmt(r['last_p']), fmt(r['last_r']),
            fmt(r['last_map50']), fmt(r['last_map5095']), fmt(r['best_map5095']),
        ]))
    print('\n注：P/R/mAP 为末轮指标，best mAP50-95 为该轮最高值')
    return 0


if __name__ == '__main__':
    sys.exit(main())
