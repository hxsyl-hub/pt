# -*- coding: utf-8 -*-
"""
dataset_2 标注一致性审计（默认只读，不改任何文件）

背景：数据集文件名有 7 种状态前缀，但标注只有 0/1 两类。已与业务确认
      0 = 裹头 NG，1 = OK。因此「不出丝」「污染」被标成 1 属于标注错误。
      本脚本把这类疑似错误全部列出来，供人工抽检后批量修正。

用法
----
python check_dataset_audit.py                       # 只体检 + 生成报告
python check_dataset_audit.py --root D:/x/dataset_2  # 指定数据集根目录
python check_dataset_audit.py --apply                # 按规则改类别号（自动备份）

产出
----
<root>/_audit_report.csv     每条不一致记录：split, file, 当前类别, 期望类别, 原因
<root>/labels_backup_<时间>   仅 --apply 时生成，原标签完整备份

检查项
------
1. 图片 / 标签是否一一对应（缺图、缺标签、多余文件）
2. 标签格式：必须 5 列、类别 ∈ {0,1}、坐标 ∈ (0,1]、w/h > 0
3. 空标签（背景图）
4. 文件名前缀与标注类别是否一致（核心：找疑似标错）
5. 完全重复图片（按内容 md5）
"""

import argparse
import csv
import hashlib
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- 规则配置
# 0 = 裹头 NG，1 = OK
NG, OK = 0, 1

EXPECTED_CLASS = {
    "nozzle_tip_wrapped":         NG,   # 裹头，NG
    "nozzle_no_extrusion":        NG,   # 不出丝，NG
    "nozzle_heavy_contamination": NG,   # 重度污染，NG
    "nozzle_slight_contamination": NG,  # 轻微污染，NG
    "camera_occlusion":           NG,   # 摄像头遮挡，NG
    "nozzle_clean":               OK,   # 干净，OK
    "nozzle_extrusion_normal":    OK,   # 正常出丝，OK
}
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def prefix_of(stem: str) -> str:
    """nozzle_clean_0007 -> nozzle_clean"""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def read_label(path: Path):
    """返回 (类别列表, 行列表)；空文件返回 ([], [])"""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    classes = []
    for ln in lines:
        f = ln.split()
        if f:
            classes.append(f[0])
    return classes, lines


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\python\gt\label_dataset\dataset_2",
                    help="数据集根目录（含 train/val/test）")
    ap.add_argument("--apply", action="store_true",
                    help="按前缀规则改写标签类别号（会先整体备份 labels 目录）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[错误] 目录不存在: {root}")

    report = []          # 待写入 csv 的不一致记录
    problems = Counter() # 问题计数
    dup_map = defaultdict(list)
    per_split_expected = {}   # split -> Counter(期望类别)
    per_split_current = {}    # split -> Counter(当前类别)

    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            print(f"[跳过] {split}: 无 images 目录")
            continue

        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}

        print(f"\n===== {split} =====  图片 {len(imgs)}  标签 {len(lbls)}")
        problems["图片无标签"] += len(set(imgs) - set(lbls))
        problems["标签无图片"] += len(set(lbls) - set(imgs))
        for s in sorted(set(imgs) - set(lbls))[:5]:
            print(f"  [缺标签] {s}")
        for s in sorted(set(lbls) - set(imgs))[:5]:
            print(f"  [缺图片] {s}")

        cur_cnt, exp_cnt = Counter(), Counter()
        for stem, lbl in sorted(lbls.items()):
            classes, lines = read_label(lbl)

            if not lines:
                problems["空标签(背景图)"] += 1
                continue

            # ---- 格式检查
            for ln in lines:
                f = ln.split()
                if len(f) != 5:
                    problems["列数不是5"] += 1
                    report.append((split, lbl.name, "?", "?", f"列数={len(f)}"))
                    continue
                try:
                    c = int(float(f[0]))
                    x, y, w, h = (float(v) for v in f[1:])
                except ValueError:
                    problems["非数字字段"] += 1
                    report.append((split, lbl.name, "?", "?", "字段无法转数字"))
                    continue
                if c not in (0, 1):
                    problems["类别号越界"] += 1
                    report.append((split, lbl.name, c, "0/1", "类别号不在{0,1}"))
                if not (0 < w <= 1 and 0 < h <= 1):
                    problems["宽高越界"] += 1
                    report.append((split, lbl.name, c, c, f"w={w} h={h}"))
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    problems["中心点越界"] += 1
                    report.append((split, lbl.name, c, c, f"x={x} y={y}"))

            # ---- 前缀 vs 标注（每张图取第一个框的类别做比对）
            cur = int(float(classes[0]))
            cur_cnt[cur] += 1
            pref = prefix_of(stem)
            exp = EXPECTED_CLASS.get(pref)
            if exp is None:
                problems["未知前缀"] += 1
                report.append((split, lbl.name, cur, "?", f"前缀 {pref} 未在规则表里"))
            else:
                exp_cnt[exp] += 1
                if cur != exp:
                    problems["前缀与标注不一致"] += 1
                    report.append((split, lbl.name, cur, exp, f"{pref} 应为 {'NG' if exp == 0 else 'OK'}"))

        per_split_current[split] = cur_cnt
        per_split_expected[split] = exp_cnt

        print(f"  当前标注: 0(NG)={cur_cnt[0]}  1(OK)={cur_cnt[1]}")
        print(f"  按前缀应为: 0(NG)={exp_cnt[0]}  1(OK)={exp_cnt[1]}")
        print(f"  疑似标错: {sum(1 for r in report if r[0] == split)} 条（含格式问题）")

        # ---- 重复图检测（同 split 内）
        for stem, p in imgs.items():
            dup_map[(split, md5_of(p))].append(stem)
        dup_groups = [v for (s, _), v in dup_map.items() if s == split and len(v) > 1]
        problems["完全重复图片(组)"] += len(dup_groups)
        if dup_groups:
            print(f"  完全重复图片: {len(dup_groups)} 组，例如 {dup_groups[0][:3]}")

    # ------------------------------------------------------------ 汇总
    print("\n" + "=" * 56)
    print("问题汇总")
    for k, v in problems.most_common():
        print(f"  {k:24s} {v}")

    tot_cur = Counter()
    tot_exp = Counter()
    for s in per_split_current:
        tot_cur.update(per_split_current[s])
        tot_exp.update(per_split_expected[s])
    if tot_cur:
        n = sum(tot_cur.values())
        print(f"\n当前标注占比: NG={tot_cur[0]} ({tot_cur[0]/n:.1%})  OK={tot_cur[1]} ({tot_cur[1]/n:.1%})")
        n2 = sum(tot_exp.values())
        print(f"修正后占比  : NG={tot_exp[0]} ({tot_exp[0]/n2:.1%})  OK={tot_exp[1]} ({tot_exp[1]/n2:.1%})")

    # ------------------------------------------------------------ 写报告
    out = root / "_audit_report.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "label_file", "当前类别", "期望类别", "原因"])
        w.writerows(report)
    print(f"\n报告已写入: {out}  （{len(report)} 条）")

    # ------------------------------------------------------------ 可选：改类别
    if args.apply:
        if not report:
            print("无需修正，退出。")
            return
        bak = root / f"labels_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[--apply] 备份 labels 到: {bak}")
        for split in SPLITS:
            src = root / split / "labels"
            if src.is_dir():
                shutil.copytree(src, bak / split / "labels")
        fixed = 0
        for split in SPLITS:
            for p in (root / split / "labels").glob("*.txt"):
                pref = prefix_of(p.stem)
                exp = EXPECTED_CLASS.get(pref)
                if exp is None:
                    continue
                classes, lines = read_label(p)
                if not classes or int(float(classes[0])) == exp:
                    continue
                new_lines = []
                for ln in lines:
                    f = ln.split()
                    f[0] = str(exp)
                    new_lines.append(" ".join(f))
                p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                fixed += 1
        # 改过标签必须删缓存，否则训练读的是旧标签
        for c in root.glob("*/labels.cache"):
            c.unlink()
            print(f"  已删除过期缓存: {c}")
        print(f"[--apply] 共修正 {fixed} 个标签文件。")


if __name__ == "__main__":
    main()
