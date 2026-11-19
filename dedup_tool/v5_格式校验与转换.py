# -*- coding: utf-8 -*-
"""
YOLOv5 兼容性校验 / 整理工具
================================================================

先讲清楚一件事（很重要）：
    YOLOv8 的【检测任务】标签格式和 YOLOv5 **完全相同** —— 都是
        class cx cy w h
    每行 5 列，坐标按图片宽高归一化到 0~1，图片和标签同名不同扩展名。
    两者差别只在 API 和 yaml 的写法，不在 txt 内容。

    真正需要处理的"v8 -> v5"问题只有下面这几种（本脚本逐项检查）：
        1. 目录名不是 train/val/test（Roboflow 常导出成 valid）
        2. 标签行多了第 6 列（置信度，v5 不认）
        3. 坐标越界（>1 或 <0）、w/h 非正
        4. 空行、多余空格、类别写成 0.0
        5. 图片和标签配不上对（有图无标签 / 有标签无图）
        6. 缺少 v5 能直接吃的 data.yaml（path / nc / names）

用法
----
    # 只检查，一个字节都不改（默认）
    python v5_格式校验与转换.py

    # 就地修复能安全修复的问题（原文件会备份到 _label_backup/）
    python v5_格式校验与转换.py --fix

    # 顺便生成 v5 版配置文件
    python v5_格式校验与转换.py --yaml D:\\python\\gt\\3dprint_nozzle.yaml

    # 指定数据集目录
    python v5_格式校验与转换.py --src D:\\python\\gt\\label_dataset\\dataset_2

脚本**只改标签 txt 的文本**，不动任何图片像素，不动目录结构，不删文件。
"""

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
V5_SPLITS = ('train', 'val', 'test')
DEFAULT_SRC = r'D:\python\gt\label_dataset\dataset_2'


# ──────────────────────────── 扫描 ────────────────────────────

def find_splits(root: Path) -> list[str]:
    """找出所有形如 <split>/images + <split>/labels 的目录。"""
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / 'images').is_dir() and (d / 'labels').is_dir():
            out.append(d.name)
    return out


def list_images(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}


def list_labels(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in d.iterdir() if p.is_file() and p.suffix == '.txt'}


# ──────────────────────────── 单行校验 ────────────────────────────

def check_line(line: str, nc: int) -> tuple[list[str], list[float] | None, str | None]:
    """返回 (问题列表, 规范化后的数值, 类别)。问题列表为空表示合规。"""
    issues = []
    p = line.split()
    if len(p) < 5:
        return ['列数不足5'], None, None
    if len(p) > 5:
        issues.append('列数%d(多余列)' % len(p))

    # 类别
    try:
        cls_f = float(p[0])
        if cls_f != int(cls_f):
            issues.append('类别非整数')
        cls = int(cls_f)
    except ValueError:
        return ['类别非数字'], None, None
    if cls < 0 or cls >= nc:
        issues.append('类别越界(%d)' % cls)

    # 坐标
    vals = []
    for name, s in zip(('x', 'y', 'w', 'h'), p[1:5]):
        try:
            v = float(s)
        except ValueError:
            return ['坐标非数字'], None, None
        if v < 0 or v > 1:
            issues.append('%s越界' % name)
        vals.append(v)

    if vals[2] <= 0 or vals[3] <= 0:
        issues.append('w/h非正')
    return issues, vals, str(cls)


def normalize_line(line: str) -> str | None:
    """把一行规范成 5 列：class cx cy w h（坐标裁剪到 0~1）。无法修复返回 None。"""
    p = line.split()
    if len(p) < 5:
        return None
    try:
        cls = int(float(p[0]))
        v = [float(x) for x in p[1:5]]
    except ValueError:
        return None
    v = [min(1.0, max(0.0, x)) for x in v]
    if v[2] <= 0 or v[3] <= 0:
        return None
    return '%d %s' % (cls, ' '.join('%.6g' % x for x in v))


# ──────────────────────────── 主流程 ────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description='YOLOv5 兼容性校验 / 整理')
    ap.add_argument('--src', default=DEFAULT_SRC, help='数据集根目录')
    ap.add_argument('--fix', action='store_true', help='就地修复（原文件备份到 _label_backup/）')
    ap.add_argument('--yaml', default=None, help='同时生成 v5 版 data.yaml 到这个路径')
    ap.add_argument('--nc', type=int, default=None, help='类别数，默认取扫描到的最大值+1')
    ap.add_argument('--data-path', default='../label_dataset/dataset_2',
                    help='写进 yaml 的 path（相对 yolov5 仓库根目录）')
    args = ap.parse_args()

    root = Path(args.src)
    if not root.is_dir():
        sys.exit('目录不存在：%s' % root)

    splits = find_splits(root)
    if not splits:
        sys.exit('没找到 <split>/images + <split>/labels 结构：%s' % root)

    print('数据集：%s' % root)
    print('识别到 split：%s' % ', '.join(splits))
    odd = [s for s in splits if s not in V5_SPLITS]
    if odd:
        print('  [提示] 这些目录名不是 train/val/test：%s' % odd)
        print('         YOLOv5 的 yaml 里可以自定义键名，改 yaml 即可，不必改目录名。')
    print()

    # 先扫一遍类别，确定 nc
    cls_all = Counter()
    for s in splits:
        for f in list_labels(root / s / 'labels').values():
            for ln in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                p = ln.split()
                if p:
                    try:
                        cls_all[int(float(p[0]))] += 1
                    except ValueError:
                        pass
    nc = args.nc if args.nc else (max(cls_all) + 1 if cls_all else 0)

    # 逐个 split 检查
    grand = Counter()
    fix_plan: list[tuple[Path, str, str]] = []      # (文件, 原行, 新行)
    per_split_cls = defaultdict(Counter)

    for s in splits:
        imgs = list_images(root / s / 'images')
        lbls = list_labels(root / s / 'labels')
        only_img = sorted(set(imgs) - set(lbls))
        only_lbl = sorted(set(lbls) - set(imgs))

        n_img = n_lbl = n_line = 0
        n_empty = n_ok = 0
        issues_cnt = Counter()

        for stem, lp in lbls.items():
            n_lbl += 1
            raw = lp.read_text(encoding='utf-8', errors='ignore')
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if not lines:
                n_empty += 1
                issues_cnt['空标签文件'] += 1
                continue
            for ln in lines:
                n_line += 1
                issues, vals, cls = check_line(ln, nc)
                if cls is not None:
                    per_split_cls[s][int(cls)] += 1
                if not issues:
                    n_ok += 1
                    continue
                for it in issues:
                    issues_cnt[it.split('(')[0]] += 1
                fixed = normalize_line(ln)
                if fixed:
                    fix_plan.append((lp, ln, fixed))
                else:
                    issues_cnt['无法修复'] += 1

        n_img = len(imgs)
        grand.update(issues_cnt)
        grand['图片'] += n_img
        grand['标签'] += n_lbl
        grand['行'] += n_line
        grand['合规行'] += n_ok
        grand['空标签'] += n_empty
        grand['有图无标签'] += len(only_img)
        grand['有标签无图'] += len(only_lbl)

        print('[%s] 图片 %d  标签 %d  行 %d' % (s, n_img, n_lbl, n_line))
        print('      合规行 %d   空标签文件 %d' % (n_ok, n_empty))
        if only_img:
            print('      有图无标签 %d 张：%s%s'
                  % (len(only_img), ', '.join(only_img[:5]), ' ...' if len(only_img) > 5 else ''))
        if only_lbl:
            print('      有标签无图 %d 个：%s%s'
                  % (len(only_lbl), ', '.join(only_lbl[:5]), ' ...' if len(only_lbl) > 5 else ''))
        if issues_cnt:
            print('      问题：%s' % dict(issues_cnt))
        print()

    # 汇总
    print('=' * 70)
    print('汇总')
    print('=' * 70)
    print('  图片总数      %d' % grand['图片'])
    print('  标签文件      %d' % grand['标签'])
    print('  标注行        %d' % grand['行'])
    print('  合规行        %d' % grand['合规行'])
    print('  空标签文件    %d' % grand['空标签'])
    print('  有图无标签    %d' % grand['有图无标签'])
    print('  有标签无图    %d' % grand['有标签无图'])
    bad = grand['行'] - grand['合规行']
    print('  待修复行      %d' % bad)
    print()
    print('  类别分布      %s   （nc=%d）' % (dict(sorted(cls_all.items())), nc))
    for s in splits:
        print('    %-6s %s' % (s, dict(sorted(per_split_cls[s].items()))))
    print('=' * 70)

    # 写盘
    if args.fix and fix_plan:
        backup = root / '_label_backup'
        print('\n开始修复 %d 行，原文件备份到 %s' % (len(fix_plan), backup))
        touched: dict[Path, list[str]] = {}
        for lp, old, new in fix_plan:
            touched.setdefault(lp, []).append(new)
        for lp, _ in touched.items():
            rel = lp.relative_to(root)
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(lp, dst)
        for lp in touched:
            raw = lp.read_text(encoding='utf-8', errors='ignore')
            out = []
            for ln in raw.splitlines():
                s = ln.strip()
                if not s:
                    continue
                issues, _, _ = check_line(s, nc)
                out.append(s if not issues else (normalize_line(s) or s))
            lp.write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('修复完成，共整理 %d 个标签文件。' % len(touched))
    elif args.fix:
        print('\n没有需要修复的行。')
    else:
        if bad:
            print('\n[只检查] 有 %d 行需要修复，加 --fix 才会真正修改。' % bad)
        else:
            print('\n[结论] 全部合规，无需任何转换，可直接给 YOLOv5 使用。')

    # 生成 yaml
    if args.yaml:
        names = ['ng', 'ok'] if nc == 2 else ['c%d' % i for i in range(nc)]
        out = Path(args.yaml)
        out.parent.mkdir(parents=True, exist_ok=True)
        key = {s: '%s/images' % s for s in splits}
        txt = ['# YOLOv5 数据配置（由 v5_格式校验与转换.py 生成）',
               '# path 相对【yolov5 仓库根目录】解析，本文件一般放在 yolov5/data/ 下',
               '',
               'path: %s' % args.data_path,
               '']
        for k, v in key.items():
            txt.append('%s: %s' % (k, v))
        txt += ['', 'nc: %d' % nc, 'names: %s' % names, '']
        out.write_text('\n'.join(txt), encoding='utf-8')
        print('\n已生成配置文件：%s' % out)


if __name__ == '__main__':
    main()
