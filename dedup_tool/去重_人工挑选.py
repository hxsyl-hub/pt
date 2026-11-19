# -*- coding: utf-8 -*-
"""
去重_人工挑选.py —— 重复图筛查 + 人工挑选保留哪张

为什么另写一个，而不是改 my_tools/重复过滤.py
--------------------------------------------
1. 原脚本用 imagehash + skimage 算 pHash / SSIM，本机 gt 环境这两个库都没装；
   且 O(n^2) 两两比 SSIM，3783 张要算 700 万次，跑不动。
   实测本数据集 916 组重复【全部是字节级完全相同】，所以只做 SHA256 精确去重就够，
   既快又不存在误杀。
2. 原脚本 os.walk 递归整个 dataset_2，train/val/test 混成一批，分不清
   "同 split 内重复" 和 "跨 split 泄漏"，而这两类处理逻辑完全不同。
3. 原脚本移动后图文平铺进一个目录、重名还改名，无法恢复。本脚本按
   <split>/images 与 <split>/labels 的原结构暂存，可以一键还原。
4. 原脚本保留规则写死 mtime 最旧；跨 split 时应优先留 train、删 val/test。
5. 原脚本扫完直接移动，没有"先出清单、再执行"的缓冲。

用法（推荐顺序）
----------------
    # 1) 只摸底，不动任何文件
    python 去重_人工挑选.py --scan

    # 2) 生成待审核清单 CSV（仍不动文件）
    python 去重_人工挑选.py --export

    # 3) 弹窗人工挑，每按一次数字键就移走那一张（可撤销）
    python 去重_人工挑选.py --review

    # 4) 不开弹窗：Excel 打开清单，在"保留"列填 0 / 1，再执行
    python 去重_人工挑选.py --apply <清单.csv>

    # 5) 全部按建议执行
    python 去重_人工挑选.py --auto

    # 6) 反悔了，按执行日志整体还原
    python 去重_人工挑选.py --restore <执行日志.csv>

弹窗按键
--------
    1~6     移走当前页的第 N 张（图片和标签一起移，可撤销）
    .       组内翻页：超过 6 张的组按 . 看下一批
    A       采纳本组建议
    回车    本组全部保留（含没翻到页的图，标题会写明）
    Z       撤销上一次移走（并回到那一组）
    Q       退出（已做的决定都已写盘，下次 --review 接着审）

建议规则（只影响"建议"列，最终由你定）
--------------------------------------
    跨 split 重复：保留 train，删 val/test（避免验证集泄漏训练集内容）
    同 split 重复：保留文件名序号最小的那张
    按此处理完，val/test 与 train 之间不会再有同一张图。

相似去重（感知哈希，新增）
--------------------------
    除字节级完全相同外，还会对每个唯一内容算 64bit pHash
    （灰度 32x32 -> DCT -> 8x8 低频按中位数二值化），
    不同精确组之间相似度 > --sim（默认 0.95）的连通成"相似"组，
    同样生成清单、同样进弹窗人工审。

    重要：相似 ≠ 相同。同工位拍的两张正常照片也可能超过阈值，
    所以相似组【永远不参与 --auto】，只有两种途径会移走：
      1) 弹窗里你按数字键 / A 键；
      2) 你自己在清单"保留"列填 0 后 --apply。
    精确组（字节级相同）的行为与原来完全一致。

数据集划分（--split，新增）
--------------------------
    人工筛选结束后，可把 train/val/test 按 --ratio（默认 7 2 1）重新划分。

    划分是【按相似组整体划分】：pHash 相似（以及字节相同）的图片
    整组进同一个 split，杜绝"近重复跨 split"造成的验证虚高；
    组的分配顺序固定 seed，结果可复现。

        python 去重_人工挑选.py --split           # 只预览，不动任何文件
        python 去重_人工挑选.py --split apply    # 确认后执行移动

    执行后生成"划分日志_*.csv"，--restore 可整体还原到划分前。

产物（全部落在本工具目录 dedup_tool/_work 下，不碰数据集、不碰 my_tools）
--------------------------------------------------------------------
    _work/
    ├─ 待审核清单_<时间>.csv     审核清单（"保留"列留空，等你填或弹窗填）
    ├─ 执行日志_<时间>.csv       每张图的原始路径 -> 暂存路径，用于还原
    └─ deleted_<时间>/           暂存区，保持 <split>/images 与 <split>/labels 结构
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 中文字体：Windows 本机有；Linux 上自动降级，只会有字体警告，不影响运行
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────── 配置 ────────────────────────────────

DEFAULT_SRC = r'D:\python\gt\label_dataset\dataset_2'
DEFAULT_OUT = r'D:\python\gt\dedup_tool\_work'   # 产物只写这里

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 跨 split 重复时保留谁的优先级：数字越小越优先保留
SPLIT_ORDER = {'train': 0, 'val': 1, 'test': 2}

MAX_DISPLAY = 6        # 弹窗一组一次并排显示几张（. 键翻页看更多）
MAX_GROUP = 30         # 相似组上限：超过则按"最近邻"拆成小圈，防链式雪崩

KIND_CROSS = '跨split'
KIND_SAME = '同split'
KIND_SIM = '相似'      # pHash 感知相似：只能人工审，--auto 永远不碰
SIM_DEFAULT = 0.95     # 相似度阈值（相似度 = 1 - 汉明距离 / 64）
PHASH_SIZE = 32        # DCT 输入边长：32x32 -> 取 8x8 低频 -> 64bit

ST_WAIT = '待审核'
ST_DONE = '已审核'
ST_RUN = '已执行'

CSV_COLUMNS = ['重复组', '重复类型', 'split', '相对路径', '文件SHA256',
               '建议', '保留', '审核状态', '备注']
LOG_COLUMNS = ['时间', '操作', '原始路径', '暂存路径']


# ──────────────────────────────── 数据结构 ────────────────────────────────

@dataclass
class Item:
    """数据集里的一张图。"""
    split: str
    rel: str            # 相对数据根目录，统一用 / 分隔
    sha: str
    cls: int | None     # 标签第一列：0=裹头NG，1=OK，None=无标签


@dataclass
class DupGroup:
    """一组重复的图。members 已按"该留谁"的优先级排好序。

    kind=KIND_CROSS/KIND_SAME 时为字节级完全相同（sha 即内容哈希）；
    kind=KIND_SIM 时为 pHash 感知相似（sha 恒为 'phash'），
    sim 记录组内最高相似度。
    """
    sha: str
    kind: str
    members: list[Item]
    sim: float = 1.0


# ──────────────────────────────── 路径小工具 ────────────────────────────────

def to_rel(p: Path | str) -> str:
    """统一成 / 分隔，写进 CSV 后各平台都能读。"""
    return str(p).replace('\\', '/')


def join_rel(root: Path | str, rel: str) -> Path:
    """把 'train/images/a.jpg' 拼回真实路径，不依赖当前系统的分隔符。"""
    return Path(root, *str(rel).split('/'))


def split_dir(root: Path | str, split: str, sub: str) -> Path:
    """root/<split>/<sub>；split 为空表示数据直接摊在 root 下。"""
    return Path(root, split, sub) if split else Path(root, sub)


def label_rel_of(rel: str) -> str:
    """train/images/a.jpg -> train/labels/a.txt"""
    p = Path(rel)
    return to_rel(p.parent.parent / 'labels' / (p.stem + '.txt'))


# ──────────────────────────────── 文件读取 ────────────────────────────────

def read_image(path: Path):
    """用 imdecode + fromfile 读图，兼容 Windows 中文路径（cv2.imread 会返回 None）。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, ValueError):
        return None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def read_cls(label_path: Path) -> int | None:
    """标签第一列就是类别号；空文件或读不到返回 None。"""
    if not label_path.is_file():
        return None
    try:
        first = label_path.read_text(encoding='utf-8', errors='ignore').split()
    except OSError:
        return None
    if not first:
        return None
    try:
        return int(float(first[0]))
    except ValueError:
        return None


# ──────────────────────────────── 扫描分组 ────────────────────────────────

def discover_splits(src: Path) -> list[str]:
    """src 下凡是包含 images 子目录的都算一个 split。"""
    splits = [p.name for p in sorted(Path(src).iterdir())
              if p.is_dir() and (p / 'images').is_dir()]
    if not splits and (Path(src) / 'images').is_dir():
        splits = ['']                       # 兜底：src 本身就是 images 的父目录
    return sorted(splits, key=lambda s: (SPLIT_ORDER.get(s, 99), s))


def scan(src: Path, sim_threshold: float = 0.0,
         cache_path: Path | None = None) -> tuple[list[Item], list[DupGroup], list[str]]:
    """按 SHA256 分组；sim_threshold>0 时再做 pHash 相似检测。"""
    src = Path(src)
    splits = discover_splits(src)
    print('扫描目录：%s' % src)
    print('识别到 split：%s' % (', '.join(splits) if splits else '(单目录)'))

    items: list[Item] = []
    by_sha: dict[str, list[Item]] = {}
    for split in splits:
        img_dir = split_dir(src, split, 'images')
        lbl_dir = split_dir(src, split, 'labels')
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
                continue
            it = Item(split=split or 'root',
                      rel=to_rel(p.relative_to(src)),
                      sha=sha256_of(p),
                      cls=read_cls(lbl_dir / (p.stem + '.txt')))
            items.append(it)
            by_sha.setdefault(it.sha, []).append(it)
            if len(items) % 500 == 0:
                print('  已计算 %d 张...' % len(items), flush=True)
    print('共 %d 张图片，唯一内容 %d 个' % (len(items), len(by_sha)), flush=True)

    groups: list[DupGroup] = []
    for sha, members in by_sha.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: (SPLIT_ORDER.get(m.split, 99), m.rel))
        kind = KIND_CROSS if len({m.split for m in members}) > 1 else KIND_SAME
        groups.append(DupGroup(sha=sha, kind=kind, members=members))
    if sim_threshold > 0:
        groups += find_similar_groups(src, by_sha, sim_threshold, cache_path)
    # 排序：跨split 完全重复 -> 同split 完全重复 -> 感知相似
    groups.sort(key=lambda g: (g.kind == KIND_SIM, g.kind != KIND_CROSS, g.members[0].rel))
    return items, groups, splits


def suggest_drop(group: DupGroup) -> set[int]:
    """建议删掉的成员下标。members 已按优先级排序，留第一个即可。

    对相似组同样成立：members[0] 是全组最该留的，
    它所在精确组的其余成员（像素完全相同）以及其余组的代表
    都排在后面，全部建议删。
    """
    return set(range(1, len(group.members)))


# ────────────────────── 感知哈希（pHash）相似检测 ──────────────────────

def phash_of(path: Path) -> int:
    """64bit 感知哈希：灰度缩到 32x32 -> DCT -> 8x8 低频按中位数二值化。"""
    img = read_image(path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(small))
    low = d[:8, :8].flatten()
    bits = low > np.median(low[1:])            # DC 项不参与中位数
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


class _DSU:
    """并查集：把相似对连通成组。"""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def find_similar_groups(src: Path, by_sha: dict[str, list[Item]],
                        threshold: float, cache_path: Path | None = None) -> list[DupGroup]:
    """在"不同精确组"之间找 pHash 相似对，连通成相似组。

    同一精确组内部相似度恒为 100%，不算边；
    只有跨精确组的相似边才会把组连成更大的"相似"组（链式合并）。
    返回的组 members 已按"该留谁"的优先级排好序，sim 为组内最高相似度。

    同时打印各阈值下的相似对数量分布——工业数据集同工位照片
    感知相似度天然偏高，阈值应该看着分布来定，而不是拍脑袋 0.85。
    """
    keys = sorted(by_sha)
    n = len(keys)
    hashes = np.zeros(n, dtype=np.uint64)

    cache: dict[str, int] = {}
    if cache_path and Path(cache_path).is_file():
        try:
            cache = {k: int(v) for k, v in
                     json.loads(Path(cache_path).read_text(encoding='utf-8')).items()}
            print('pHash 缓存命中 %d 条' % len(cache))
        except (OSError, ValueError):
            cache = {}

    todo = [s for s in keys if s not in cache]
    print('计算感知哈希（%d 个唯一内容，缓存命中 %d）...' % (n, n - len(todo)), flush=True)
    for i, s in enumerate(todo):
        first = by_sha[s][0]                   # 同 sha 内容一样，算一次就够
        cache[s] = phash_of(join_rel(src, first.rel)) & 0xFFFFFFFFFFFFFFFF
        if (i + 1) % 300 == 0:
            print('  pHash %d / %d ...' % (i + 1, len(todo)), flush=True)
    if cache_path and todo:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache), encoding='utf-8')
        except OSError as e:
            print('  [提示] pHash 缓存没写上：%s' % e)

    for i, s in enumerate(keys):
        hashes[i] = np.uint64(cache[s])
    bad = hashes == 0                          # 解码失败的图不参与比对
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((hashes[:, None] >> shifts) & np.uint64(1)).astype(bool)

    dsu = _DSU(n)
    edges: list[tuple[int, int, float]] = []
    sims_all: list[np.ndarray] = []            # 上三角相似度，用于分布统计
    print('两两比对相似度（只看跨精确组）...', flush=True)
    step = 256                                 # 分块算距离，控制内存
    for s0 in range(0, n, step):
        c = min(step, n - s0)
        dist = (bits[s0:s0 + c, None, :] != bits[None, :, :]).sum(axis=2)
        sim = 1.0 - dist / 64.0
        upper = np.arange(s0, s0 + c)[:, None] < np.arange(n)[None, :]
        sims_all.append(sim[upper].astype(np.float32))
        keep = (sim > threshold) & upper & ~bad[s0:s0 + c, None] & ~bad[None, :]
        for a, b in zip(*np.nonzero(keep)):
            i0, j0 = s0 + int(a), int(b)
            dsu.union(i0, j0)
            edges.append((i0, j0, float(sim[a, b])))
    print('相似对（> %.2f）%d 对。' % (threshold, len(edges)), flush=True)

    # ---- 相似度分布：选阈值看这个表，不靠猜
    if sims_all:
        sims = np.concatenate(sims_all)
        print('\n相似度分布（%d 万对，跨精确组）' % round(len(sims) / 1e4))
        print('  %-10s %12s' % ('相似度 >', '对数'))
        for t in (0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.88, 0.85, 0.80):
            print('  %-10.2f %12d' % (t, int((sims > t).sum())))

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(dsu.find(i), []).append(i)

    # ---- 链式雪崩控制：分量太小说明正常；太大概率是"A≈B≈C≈…接力相似"
    # 串起来的假组（组内首尾可能毫不相似，人工根本没法审）。
    # 处理：每个成员只保留它最相似的一条边，重新连通 -> 拆成真实的小圈。
    if any(len(m) > MAX_GROUP for m in comps.values()):
        keep: dict[int, tuple[int, float]] = {}
        for i, j, s in edges:
            if i not in keep or s > keep[i][1]:
                keep[i] = (j, s)
            if j not in keep or s > keep[j][1]:
                keep[j] = (i, s)
        dsu = _DSU(n)
        for i, (j, _s) in keep.items():
            dsu.union(i, j)
        comps = {}
        for i in range(n):
            comps.setdefault(dsu.find(i), []).append(i)
        print('检测到超大组，已按"最近邻"拆成小圈（组上限 %d 张）。' % MAX_GROUP)

    groups: list[DupGroup] = []
    group_of_root: dict[int, DupGroup] = {}
    for members_idx in comps.values():
        if len(members_idx) < 2:
            continue
        flat = [it for i in members_idx for it in by_sha[keys[i]]]
        flat.sort(key=lambda m: (SPLIT_ORDER.get(m.split, 99), m.rel))
        g = DupGroup(sha='phash', kind=KIND_SIM, members=flat, sim=0.0)
        groups.append(g)
        group_of_root[dsu.find(members_idx[0])] = g
    for i, _j, s in edges:
        g = group_of_root.get(dsu.find(i))
        if g and s > g.sim:
            g.sim = s

    n_in = sum(len(g.members) for g in groups)
    if n_in > 0.3 * n:
        print('\n[警告] 阈值 %.2f 下有 %d/%d 张图被连进了相似组 —— '
              '同工位照片感知相似度天然偏高，这个阈值会把正常样本也当重复。'
              % (threshold, n_in, n))
        print('       请看上面的分布表，调高 --sim 重试（--scan 不改任何文件）。')
    return groups


# ──────────────────────────────── 统计报告 ────────────────────────────────

def print_report(items: list[Item], groups: list[DupGroup], splits: list[str]) -> None:
    """打印重复情况，以及"按建议执行后会变成什么样"（纯计算，不碰文件）。"""
    drop = {g.members[i].rel for g in groups for i in suggest_drop(g)}

    n_cross = sum(1 for g in groups if g.kind == KIND_CROSS)
    print()
    print('=' * 66)
    print('重复统计')
    print('=' * 66)
    print('  图片总数          %6d' % len(items))
    print('  唯一内容          %6d' % len({it.sha for it in items}))
    print('  重复组            %6d 组（涉及 %d 张）'
          % (len(groups), sum(len(g.members) for g in groups)))
    print('    其中 跨 split（泄漏） %4d 组' % n_cross)
    print('    其中 同 split 内      %4d 组' % (len(groups) - n_cross))

    sim_groups = [g for g in groups if g.kind == KIND_SIM]
    if sim_groups:
        n_img = sum(len(g.members) for g in sim_groups)
        cross = sum(1 for g in sim_groups
                    if len({m.split for m in g.members}) > 1)
        print()
        print('  感知相似组（pHash > 阈值，非完全相同）%d 组，涉及 %d 张'
              % (len(sim_groups), n_img))
        print('    相似度范围 %.1f%% ~ %.1f%%；其中跨 split %d 组'
              % (min(g.sim for g in sim_groups) * 100,
                 max(g.sim for g in sim_groups) * 100, cross))
        print('    注意：--auto 不会执行相似组，必须 --review 人工判断')

    print()
    print('按建议处理后的结果预演')
    print('-' * 66)
    print('  %-8s %8s %8s %8s %8s' % ('split', '原始', '移走', '剩余', '变化'))
    for split in splits:
        sub = [it for it in items if it.split == (split or 'root')]
        total = len(sub)
        removed = sum(1 for it in sub if it.rel in drop)
        pct = removed / total * 100 if total else 0.0
        print('  %-8s %8d %8d %8d %6.1f%%' % (split, total, removed, total - removed, pct))

    print()
    print('类别分布变化（0=裹头NG，1=OK）  左侧=原始，右侧=去重后')
    print('-' * 66)
    for split in splits:
        sub = [it for it in items if it.split == (split or 'root')]
        ng_all = sum(1 for it in sub if it.cls == 0)
        ok_all = sum(1 for it in sub if it.cls == 1)
        ng_keep = sum(1 for it in sub if it.cls == 0 and it.rel not in drop)
        ok_keep = sum(1 for it in sub if it.cls == 1 and it.rel not in drop)
        print('  %-8s NG=%-5d OK=%-5d  ->  NG=%-5d OK=%-5d'
              % (split, ng_all, ok_all, ng_keep, ok_keep))


# ──────────────────────────────── 清单读写 ────────────────────────────────

def export_csv(groups: list[DupGroup], out_dir: Path) -> Path:
    """把重复组写成待审核清单，'建议'列预填 1=留 / 0=删，'保留'列留空等你填。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ('待审核清单_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for gi, g in enumerate(groups, 1):
            prefix = 'sim_' if g.kind == KIND_SIM else 'group_'
            gid = '%s%05d' % (prefix, gi)
            drop = suggest_drop(g)
            note = ('相似度%.0f%%，请人工判断' % (g.sim * 100)) if g.kind == KIND_SIM else ''
            for mi, m in enumerate(g.members):
                w.writerow([gid, g.kind, m.split, m.rel, m.sha,
                            '0' if mi in drop else '1', '', ST_WAIT, note])
    print('待审核清单：%s' % path)
    return path


def load_csv(path: Path | str) -> list[dict]:
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in CSV_COLUMNS if rows and c not in rows[0]]
    if missing:
        sys.exit('清单缺少列 %s，确认这是本脚本生成的清单。' % '、'.join(missing))
    return rows


def save_csv(rows: list[dict], path: Path | str) -> None:
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def group_rows(rows: list[dict]) -> list[tuple[str, list[int]]]:
    """按 重复组 归并成 [(组号, 行下标...)]，保持清单里的原始顺序。"""
    order: list[str] = []
    bucket: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        gid = r['重复组']
        if gid not in bucket:
            bucket[gid] = []
            order.append(gid)
        bucket[gid].append(i)
    return [(gid, bucket[gid]) for gid in order if len(bucket[gid]) >= 2]


# ──────────────────────────────── 移动 / 还原 ────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def move_pair(src_root: Path, rel: str, stage_root: Path, action: str,
              log: list[list[str]]) -> bool:
    """图片和同名标签一起搬进暂存区，保持相对结构。搬成功才算一条日志。"""
    src_img = join_rel(src_root, rel)
    if not src_img.is_file():
        log.append([_now(), '跳过:图片不存在', str(src_img), ''])
        return False

    dst_img = join_rel(stage_root, rel)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src_img), str(dst_img))
    except OSError as e:
        log.append([_now(), '失败:%s' % e, str(src_img), ''])
        return False
    log.append([_now(), action, str(src_img), str(dst_img)])

    lbl_rel = label_rel_of(rel)
    src_lbl = join_rel(src_root, lbl_rel)
    if src_lbl.is_file():
        dst_lbl = join_rel(stage_root, lbl_rel)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src_lbl), str(dst_lbl))
            log.append([_now(), action + '(标签)', str(src_lbl), str(dst_lbl)])
        except OSError as e:
            print('  [警告] 图片已移走但标签没搬成功：%s（%s）' % (src_lbl, e))
    return True


def write_log(log_path: Path | str, log: list[list[str]]) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        w.writerows(log)


def restore(log_path: Path | str, src_root: Path | None = None) -> int:
    """按执行日志把文件搬回原位。日志里已经还原过的会被自动跳过。"""
    with open(log_path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    back = skip = 0
    for r in rows:
        src, dst = r.get('原始路径', ''), r.get('暂存路径', '')
        if not src or not dst or not Path(dst).is_file():
            skip += 1
            continue
        Path(src).parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(dst, src)
            back += 1
        except OSError as e:
            print('  还原失败：%s -> %s（%s）' % (dst, src, e))
            skip += 1

    print('已还原 %d 个文件。' % back)
    if skip:
        print('跳过 %d 条（暂存文件已不存在，或已还原过）。' % skip)
    if src_root and back:
        cleanup_cache(src_root)
    return back


def cleanup_cache(src: Path) -> None:
    """删掉 YOLO 的 labels.cache，否则可能读到旧的文件清单。"""
    n = 0
    for split in discover_splits(src):
        c = split_dir(src, split, 'labels.cache')
        if c.is_file():
            c.unlink()
            n += 1
    if n:
        print('已删除 %d 个 labels.cache（下次训练会自动重建）' % n)


# ──────────────────────────────── 弹窗人工挑选 ────────────────────────────────

class Reviewer:
    """弹窗审核器：一组并排显示，按键决定移走哪张。"""

    def __init__(self, rows: list[dict], csv_path: Path, src_root: Path,
                 stage_root: Path, log_path: Path):
        self.rows = rows
        self.csv_path = Path(csv_path)
        self.src_root = Path(src_root)
        self.stage_root = Path(stage_root)
        self.log_path = Path(log_path)

        self.queue: list[tuple[str, list[int]]] = []   # 待审组
        self.pos = 0                                   # 当前在 queue 里的位置
        self.page = 0                                  # 组内翻页：. 键翻看下一批 6 张
        self.cur: tuple[str, list[int], list[int]] | None = None
        self.undo: list[tuple[int, list[list[str]]]] = []
        self.log: list[list[str]] = []
        self.closed = False

        self.fig, self.axes = plt.subplots(1, MAX_DISPLAY, figsize=(3.4 * MAX_DISPLAY, 5.4))
        try:
            self.fig.canvas.manager.set_window_title('重复图人工挑选')
        except AttributeError:                          # 个别后端没有 manager
            pass
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    # ---------- 数据 ----------

    def abs_path(self, row_idx: int) -> Path:
        return join_rel(self.src_root, self.rows[row_idx]['相对路径'])

    def alive(self, idxs: list[int]) -> list[int]:
        """组里还留在数据集里的那些行（已移走的会消失）。"""
        return [i for i in idxs if self.abs_path(i).is_file()]

    def build_queue(self) -> None:
        self.queue = [(gid, idxs) for gid, idxs in group_rows(self.rows)
                      if len(self.alive(idxs)) >= 2]

    def pos_of(self, row_idx: int) -> int:
        """row_idx 所属组在 queue 里的位置；找不到就回 0。"""
        for i, (_, idxs) in enumerate(self.queue):
            if row_idx in idxs:
                return i
        return 0

    # ---------- 界面 ----------

    def draw(self, gid: str, idxs: list[int], alive: list[int]) -> None:
        for ax in self.axes:
            ax.clear()
            ax.axis('off')

        n_pages = max(1, -(-len(alive) // MAX_DISPLAY))     # 组内总页数
        if self.page >= n_pages:
            self.page = n_pages - 1
        if self.page < 0:
            self.page = 0
        start = self.page * MAX_DISPLAY
        view = alive[start:start + MAX_DISPLAY]

        for n, (ax, i) in enumerate(zip(self.axes, view), 1):
            r = self.rows[i]
            img = read_image(self.abs_path(i))
            if img is None:
                ax.text(0.5, 0.5, '(已移走)', ha='center', va='center')
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest')
            hint = '  <建议删>' if r['建议'] == '0' else ''
            ax.set_title('[%d] %s\n%s%s' % (start + n, r['split'], Path(r['相对路径']).name, hint),
                         fontsize=9)

        kind = self.rows[idxs[0]]['重复类型']
        page_txt = ('  第 %d/%d 页（. 翻页）' % (self.page + 1, n_pages)) if n_pages > 1 else ''
        self.fig.suptitle(
            '%s | %s | 第 %d / %d 组 | 本组剩 %d 张%s\n'
            '数字键=移走当前页该张   A=采纳建议   回车=本组全部保留   .=下一页   Z=撤销   Q=退出'
            % (gid, kind, self.pos + 1, len(self.queue), len(alive), page_txt), fontsize=11)
        self.fig.canvas.draw_idle()

    def show_current(self) -> None:
        """从 pos 开始找第一组还剩 2 张以上的，显示它。"""
        while self.pos < len(self.queue):
            gid, idxs = self.queue[self.pos]
            alive = self.alive(idxs)
            if len(alive) >= 2:
                self.cur = (gid, idxs, alive)
                self.draw(gid, idxs, alive)
                self.print_group(gid, alive)
                return
            self.pos += 1
        self.cur = None
        print('\n全部审完。')
        self.fig.suptitle('审核完成  按 Q 退出', fontsize=13)
        self.fig.canvas.draw_idle()

    def print_group(self, gid: str, alive: list[int]) -> None:
        print('\n[%s] %s  剩 %d 张' % (gid, self.rows[alive[0]]['重复类型'], len(alive)))
        start = self.page * MAX_DISPLAY
        view = alive[start:start + MAX_DISPLAY]
        for n, i in enumerate(view, start + 1):
            r = self.rows[i]
            note = ('  ' + r['备注']) if r['备注'] else ''
            print('   %d: %s/%s  建议=%s%s' % (n, r['split'], Path(r['相对路径']).name,
                                              r['建议'], note))
        if len(alive) > MAX_DISPLAY:
            print('   （第 %d/%d 页，. 翻页）' % (self.page + 1,
                                                 -(-len(alive) // MAX_DISPLAY)))

    def advance(self) -> None:
        self.pos += 1
        self.page = 0                    # 切到新组，回到第 1 页
        self.show_current()

    # ---------- 动作 ----------

    def mark(self, keep: list[int], moved: list[int], note: str, done: bool) -> None:
        """把本次决定写回内存并落盘。done=False 表示这组还没审完，下次还要看。"""
        state = ST_DONE if done else ST_WAIT
        for i in keep:
            self.rows[i]['保留'], self.rows[i]['审核状态'], self.rows[i]['备注'] = '1', state, note
        for i in moved:
            self.rows[i]['保留'], self.rows[i]['审核状态'], self.rows[i]['备注'] = '0', ST_DONE, note
        save_csv(self.rows, self.csv_path)

    def do_move(self, row_idx: int, action: str) -> bool:
        """移走一张（图片+标签），并把这次动作压进撤销栈。"""
        tmp: list[list[str]] = []
        ok = move_pair(self.src_root, self.rows[row_idx]['相对路径'], self.stage_root, action, tmp)
        if ok:
            self.log.extend(tmp)
            self.undo.append((row_idx, tmp))
        return ok

    def undo_last(self) -> None:
        if not self.undo:
            print('  没有可撤销的操作')
            return
        row_idx, tmp = self.undo.pop()
        back = 0
        for row in reversed(tmp):               # [时间, 操作, 原始路径, 暂存路径]
            dst, src = row[3], row[2]
            if not dst or not Path(dst).is_file():
                continue
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(dst, src)
                back += 1
            except OSError as e:
                print('  撤销失败：%s' % e)
        self.log = [r for r in self.log if r not in tmp]

        gid = self.rows[row_idx]['重复组']     # 整组状态复位，等重新审
        for r in self.rows:
            if r['重复组'] == gid:
                r['保留'], r['审核状态'], r['备注'] = '', ST_WAIT, '已撤销'
        save_csv(self.rows, self.csv_path)

        self.pos = self.pos_of(row_idx)         # 画面回到被撤销的那一组
        print('  已撤销 %d 个文件' % back)
        self.show_current()

    # ---------- 事件 ----------

    def _on_close(self, _event) -> None:
        self.closed = True

    def on_key(self, event) -> None:
        if self.closed or self.cur is None:
            return
        gid, idxs, alive = self.cur
        k = event.key

        if k in ('q', 'escape'):
            plt.close(self.fig)
            return

        if k in ('z', 'Z'):
            self.undo_last()
            return

        if k in ('.', ','):                       # 组内翻页：看下一批 6 张
            self.page += 1
            self.draw(gid, idxs, alive)
            return

        if k in ('enter', 'return'):
            self.mark(alive, [], '人工判断：都保留', done=True)
            print('  -> 都保留')
            self.advance()
            return

        if k in ('a', 'A'):
            keep = [i for i in alive if self.rows[i]['建议'] == '1']
            drop = [i for i in alive if self.rows[i]['建议'] == '0']
            if not keep:                        # 建议全删时兜底留第一张
                keep, drop = alive[:1], alive[1:]
            moved = [i for i in drop if self.do_move(i, '按建议移入暂存')]
            left = [i for i in keep if i not in moved]
            self.mark(left, moved, '采纳建议', done=len(left) < 2)
            print('  -> 采纳建议，移走 %d 张' % len(moved))
            self.advance()
            return

        if k and k.isdigit():
            slot = self.page * MAX_DISPLAY + int(k) - 1   # 当前页内第 k 张
            if int(k) > MAX_DISPLAY or slot >= len(alive):
                print('  当前页没有第 %s 张（本页共 %d 张）'
                      % (k, min(MAX_DISPLAY, len(alive) - self.page * MAX_DISPLAY)))
                return
            if len(alive) <= 1:
                print('  不能移走最后一张')
                return
            row_idx = alive[slot]
            if not self.do_move(row_idx, '人工移入暂存'):
                print('  移走失败，看日志')
                return
            rest = [i for i in alive if i != row_idx]
            self.mark(rest, [row_idx], '人工判断：移走第%s张' % k, done=len(rest) < 2)
            print('  -> 已移走：%s（Z 可撤销）' % Path(self.rows[row_idx]['相对路径']).name)
            if len(rest) < 2:
                self.advance()
            else:
                self.cur = (gid, idxs, rest)
                self.draw(gid, idxs, rest)

    # ---------- 主循环 ----------

    def run(self) -> None:
        self.build_queue()
        total = len(group_rows(self.rows))
        print('还有 %d 组待审核（共 %d 组）' % (len(self.queue), total))
        if not self.queue:
            print('没有待审核的组了。')
            plt.close(self.fig)
            return

        self.show_current()
        plt.tight_layout(rect=(0, 0, 1, 0.88))
        plt.show()

        if self.log:
            write_log(self.log_path, self.log)
            print('\n执行日志：%s' % self.log_path)
            print('暂存目录：%s' % self.stage_root)
            print('反悔就还原：--restore "%s"' % self.log_path)
        else:
            print('\n本次没有移动任何文件。')
        print('清单已更新：%s' % self.csv_path)


# ────────────────────── 数据集划分（--ratio，默认 7:2:1） ──────────────────────

def plan_split(items: list[Item], groups: list[DupGroup],
               ratios: list[int], seed: int) -> dict[str, str]:
    """把全部图片按"相似组"整体划分成 train/val/test。

    划分单元：
      1. pHash 相似分量（kind=KIND_SIM 的组）—— 整组进同一个 split；
      2. 不在任何相似组里的图，按内容 sha 归并（同内容必同 split）；
      3. 其余每张图自成单元。

    大组优先，放入"按比例缺口最大"的 split；固定 seed 保证可复现。
    返回 {相对路径: 目标split}。
    """
    sim_groups = [g for g in groups if g.kind == KIND_SIM]
    in_sim: set[str] = set()
    units: list[list[Item]] = []
    for g in sim_groups:
        units.append(list(g.members))
        in_sim.update(m.rel for m in g.members)

    by_sha: dict[str, list[Item]] = {}
    for it in items:
        if it.rel not in in_sim:
            by_sha.setdefault(it.sha, []).append(it)
    units += list(by_sha.values())

    total = len(items)
    rsum = float(sum(ratios))
    target = {'train': total * ratios[0] / rsum,
              'val': total * ratios[1] / rsum,
              'test': total * ratios[2] / rsum}

    rng = random.Random(seed)
    units.sort(key=lambda u: (-len(u), rng.random()))
    counts = {'train': 0, 'val': 0, 'test': 0}
    assign: dict[str, str] = {}
    for u in units:
        s = min(counts, key=lambda s: (counts[s] + len(u)) / max(target[s], 1.0))
        for it in u:
            assign[it.rel] = s
        counts[s] += len(u)
    return assign


def print_split_report(assign: dict[str, str], items: list[Item],
                       ratios: list[int]) -> None:
    cls_of = {it.rel: it.cls for it in items}
    total = len(items)
    print()
    print('=' * 66)
    print('划分方案（%d:%d:%d）' % tuple(ratios))
    print('=' * 66)
    print('  %-6s %8s %8s %8s %8s' % ('split', '张数', '占比', 'NG', 'OK'))
    for s in ('train', 'val', 'test'):
        rels = [r for r, t in assign.items() if t == s]
        ng = sum(1 for r in rels if cls_of.get(r) == 0)
        ok = sum(1 for r in rels if cls_of.get(r) == 1)
        print('  %-6s %8d %7.1f%% %8d %8d' % (s, len(rels), len(rels) / total * 100, ng, ok))
    moved = sum(1 for rel, t in assign.items() if rel.split('/')[0] != t)
    print('  需移动 %d 张，原地不动 %d 张（相似组整体划分，组不会跨 split）'
          % (moved, total - moved))


def apply_split(assign: dict[str, str], src_root: Path,
                out_dir: Path, stamp: str) -> None:
    """按划分方案移动图片和标签。目标重名时自动加 _rN 后缀。

    日志格式与去重执行日志一致（原始路径 -> 新路径），
    因此 --restore 可以把整个数据集还原到划分前。
    """
    log: list[list[str]] = []
    moved = renamed = 0
    for rel in sorted(assign):
        tgt = assign[rel]
        if rel.split('/')[0] == tgt:
            continue
        src_img = join_rel(src_root, rel)
        if not src_img.is_file():
            log.append([_now(), '跳过:图片不存在', str(src_img), ''])
            continue
        relp = Path(rel)
        dst_rel = to_rel(Path(tgt) / 'images' / relp.name)
        n = 0
        while join_rel(src_root, dst_rel).exists():
            n += 1
            dst_rel = to_rel(Path(tgt) / 'images'
                             / ('%s_r%d%s' % (relp.stem, n, relp.suffix)))
        if n:
            renamed += 1
        dst_img = join_rel(src_root, dst_rel)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_img), str(dst_img))
        log.append([_now(), '划分移动', str(src_img), str(dst_img)])

        src_lbl = join_rel(src_root, label_rel_of(rel))
        if src_lbl.is_file():
            dst_lbl = join_rel(src_root, to_rel(Path(tgt) / 'labels'
                                                / (Path(dst_rel).stem + '.txt')))
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_lbl), str(dst_lbl))
            log.append([_now(), '划分移动(标签)', str(src_lbl), str(dst_lbl)])
        moved += 1

    log_path = out_dir / ('划分日志_%s.csv' % stamp)
    write_log(log_path, log)
    print('已移动 %d 张（目标重名改名 %d），标签随图一起走。' % (moved, renamed))
    print('划分日志：%s（想还原：--restore "%s"）' % (log_path, log_path))

    # 搬空的旧 images/labels 目录顺手清掉（rmdir 只会删空目录，安全）
    for split in discover_splits(src_root):
        for sub in ('images', 'labels'):
            try:
                split_dir(src_root, split, sub).rmdir()
            except OSError:
                pass


# ──────────────────────────────── 批量执行 ────────────────────────────────

def apply_rows(rows: list[dict], src_root: Path, stage_root: Path,
               log_path: Path, use_suggest: bool = False) -> int:
    """把要删的行移进暂存区。use_suggest=True 表示未审核的行也按建议列执行。

    相似组（KIND_SIM）永远不参与 --auto：相似不等于相同，必须人工审。
    人工 --apply 时"保留"列填了 0 的相似行照常执行。
    """
    if use_suggest:                     # --auto：保留方也算审过了，免得清单里残留"待审核"
        for r in rows:
            if (r['重复类型'] != KIND_SIM and r['保留'] != '0'
                    and r['建议'] == '1' and r['审核状态'] == ST_WAIT):
                r['审核状态'] = ST_DONE
                r['备注'] = r['备注'] or '保留（按建议）'

    todo = [r for r in rows
            if r['保留'] == '0'
            or (use_suggest and not r['保留'] and r['建议'] == '0'
                and r['重复类型'] != KIND_SIM)]
    if not todo:
        print('清单里没有需要移走的图片。')
        if not use_suggest:
            print('  说明：--apply 只处理"保留"列填了 0 的行。')
            print('  想按建议批量执行请用 --auto；想逐组人工挑请用 --review。')
        return 0

    log: list[list[str]] = []
    moved = 0
    for r in todo:
        if move_pair(src_root, r['相对路径'], stage_root, '批量移入暂存', log):
            moved += 1
            r['保留'] = '0'
            r['审核状态'] = ST_RUN
            r['备注'] = r['备注'] or ('按建议执行' if use_suggest else '按清单执行')

    write_log(log_path, log)
    print('已移走 %d 张（图片及其标签）。' % moved)
    print('执行日志：%s' % log_path)
    print('暂存目录：%s' % stage_root)
    return moved


# ──────────────────────────────── 主流程 ────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='重复图筛查 + 人工挑选保留哪张')
    ap.add_argument('--src', default=DEFAULT_SRC, help='数据集根目录（含 train/val/test）')
    ap.add_argument('--out', default=DEFAULT_OUT, help='清单/日志/暂存目录')
    ap.add_argument('--scan', action='store_true', help='只统计，不动任何文件、不建目录')
    ap.add_argument('--export', action='store_true', help='生成待审核清单 CSV（不动文件）')
    ap.add_argument('--sim', type=float, default=SIM_DEFAULT,
                    help='感知相似度阈值 0~1（默认 %.2f），设 0 可关闭相似检测' % SIM_DEFAULT)
    ap.add_argument('--review', action='store_true', help='弹窗逐组人工挑选')
    ap.add_argument('--resume', help='指定要接着审的清单 CSV')
    ap.add_argument('--apply', help='按清单 CSV 执行（保留=0 的移走）')
    ap.add_argument('--auto', action='store_true', help='全部按建议执行（会先确认一次）')
    ap.add_argument('--yes', '-y', action='store_true', help='配合 --auto 跳过确认')
    ap.add_argument('--restore', help='按执行日志还原')
    ap.add_argument('--split', nargs='?', const='plan', default=None, metavar='apply',
                    help='按比例重新划分 train/val/test：不带值=预览，--split apply=执行')
    ap.add_argument('--ratio', nargs=3, type=int, default=[7, 2, 1],
                    metavar=('TRAIN', 'VAL', 'TEST'), help='划分比例，默认 7 2 1')
    ap.add_argument('--seed', type=int, default=0, help='划分随机种子，保证结果可复现')
    return ap


def pick_csv(args, out_dir: Path) -> Path:
    """审核模式用哪份清单：--resume / --apply 指定 > _work 里最新的那份。"""
    if args.resume or args.apply:
        return Path(args.resume or args.apply)
    cand = sorted(out_dir.glob('待审核清单_*.csv'))
    if not cand:
        sys.exit('没找到清单，请先运行 --export')
    print('使用最新清单：%s' % cand[-1])
    return cand[-1]


def main() -> None:
    args = build_parser().parse_args()

    if args.restore:
        src_root = Path(args.src)
        restore(args.restore, src_root if src_root.is_dir() else None)
        return

    if args.review and args.apply:
        sys.exit('--review 与 --apply 不能同时使用')

    src_root = Path(args.src).resolve()
    if not src_root.is_dir():
        sys.exit('目录不存在：%s' % src_root)

    out_dir = Path(args.out)
    if not args.scan:                       # --scan 纯只读，连目录都不建
        out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ---- 划分数据集（独立于去重流程，--split 时不需要清单）
    if args.split:
        items, groups, _ = scan(src_root, args.sim, out_dir / 'phash_cache.json')
        assign = plan_split(items, groups, args.ratio, args.seed)
        print_split_report(assign, items, args.ratio)
        if args.split == 'apply':
            if input('\n确认按上述方案移动文件吗？输入 yes 继续：').strip().lower() != 'yes':
                print('已取消，未做任何改动。')
                return
            apply_split(assign, src_root, out_dir, stamp)
            cleanup_cache(src_root)
        else:
            print('\n（预览模式，未移动任何文件。确认方案后执行：--split apply）')
        return

    # ---- 审核 / 执行：只读清单，不重新扫描
    if args.review or args.apply:
        csv_path = pick_csv(args, out_dir)
        rows = load_csv(csv_path)
        stage_root = out_dir / ('deleted_%s' % stamp)
        log_path = out_dir / ('执行日志_%s.csv' % stamp)

        if args.apply:
            apply_rows(rows, src_root, stage_root, log_path)
            save_csv(rows, csv_path)
        else:
            Reviewer(rows, csv_path, src_root, stage_root, log_path).run()

        cleanup_cache(src_root)
        return

    # ---- 扫描
    items, groups, splits = scan(src_root, args.sim, out_dir / 'phash_cache.json')
    print_report(items, groups, splits)
    if args.scan:
        print('\n（--scan 只统计，未生成清单、未改动任何文件）')
        return

    csv_path = export_csv(groups, out_dir)

    if args.auto:
        if not args.yes:
            if input('\n确认按建议执行吗？输入 yes 继续：').strip().lower() != 'yes':
                print('已取消，未做任何改动。')
                return
        rows = load_csv(csv_path)
        stage_root = out_dir / ('deleted_%s' % stamp)
        log_path = out_dir / ('执行日志_%s.csv' % stamp)
        apply_rows(rows, src_root, stage_root, log_path, use_suggest=True)
        save_csv(rows, csv_path)
        cleanup_cache(src_root)

        remain = [r for r in rows if r['审核状态'] == ST_WAIT]
        if remain:
            n_groups = len({r['重复组'] for r in remain})
            print('\n注意：相似组 %d 组（%d 行）未执行 —— --auto 只处理完全重复。' % (n_groups, len(remain)))
            print('这些必须人工判断：--review（弹窗里按数字键/A/回车）')
        return

    print()
    print('下一步三选一：')
    print('  1) 弹窗人工挑：--review          （相似组必须走这一步）')
    print('  2) Excel 打开清单填"保留"列，再 --apply <清单>')
    print('  3) 全按建议批量执行：--auto      （只会执行完全重复的组）')


if __name__ == '__main__':
    main()
