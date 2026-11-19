# -*- coding: utf-8 -*-
"""
生成上传算力云的两个压缩包（分开打包，改代码后只需重传 code.zip）：

    python scripts/package_upload.py              # 两个包都生成
    python scripts/package_upload.py --only code  # 只生成代码包

包内路径都相对项目根，在云上项目目录解压即可：
    code.zip       -> configs/ scripts/ analysis_tool/ my_tools/ dedup_tool/ yolov5-7.0/
    dataset_2.zip  -> label_dataset/dataset_2/{train,val,test}

云端无法访问 GitHub 时 yolov5 随 code.zip 上传；权重（yolov5s.pt / yolo11s.pt）不打包，运行时自动下载。
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {'__pycache__', '.git', '.idea'}
CODE_DIRS = ['analysis_tool', 'my_tools']
CODE_SKIP_DIRS = {'output', '__pycache__'}
CODE_EXTS = {'.py', '.yaml', '.yml', '.json', '.txt', '.md', '.sh'}
DATASET_SPLITS = ['train', 'val', 'test']

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def collect_files(paths, skip_dirs=SKIP_DIRS):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                files.extend(Path(dirpath) / f for f in filenames)
    return sorted(files)


def write_zip(out_path, files, compression):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(out_path, 'w', compression=compression, allowZip64=True) as zf:
        for f in files:
            zf.write(f, arcname=f.relative_to(ROOT).as_posix())
            total += f.stat().st_size
    print(f'  {out_path.name}: {len(files)} 个文件, 原始 {total / 1024 / 1024:.1f} MB, '
          f'包体 {out_path.stat().st_size / 1024 / 1024:.1f} MB')
    return out_path


def build_code(out_dir):
    print('代码包:')
    files = collect_files([ROOT / 'configs', ROOT / 'scripts'])
    for d in CODE_DIRS:
        files += collect_files([ROOT / d], skip_dirs=SKIP_DIRS | CODE_SKIP_DIRS)
    files += collect_files(list((ROOT / 'dedup_tool').glob('*.py')))
    files = [f for f in files if f.name != Path(__file__).name and f.suffix.lower() in CODE_EXTS]

    # 云端无法 clone 时一并上传 yolov5（已含 Python 3.12 / numpy 2 / Pillow 10 兼容补丁）
    yolo_dir = ROOT / 'yolov5-7.0'
    if yolo_dir.is_dir():
        files += [f for f in collect_files([yolo_dir]) if f.suffix.lower() != '.pyc']
    return write_zip(out_dir / 'code.zip', files, zipfile.ZIP_DEFLATED)


def build_dataset(out_dir):
    print('数据集包:')
    base = ROOT / 'label_dataset' / 'dataset_2'
    missing = [s for s in DATASET_SPLITS if not (base / s).is_dir()]
    if missing:
        print(f'  警告: 缺少划分目录 {missing}')
    files = collect_files([base / s for s in DATASET_SPLITS if (base / s).is_dir()])
    # jpg 本身已压缩，stored 打包更快且体积几乎不变
    return write_zip(out_dir / 'dataset_2.zip', files, zipfile.ZIP_STORED)


def main():
    parser = argparse.ArgumentParser(description='生成上传算力云的压缩包')
    parser.add_argument('--only', choices=['code', 'dataset', 'all'], default='all')
    parser.add_argument('--out', default=str(ROOT / 'artifacts' / 'dist'), help='压缩包输出目录')
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.only in ('code', 'all'):
        build_code(out_dir)
    if args.only in ('dataset', 'all'):
        build_dataset(out_dir)
    print(f'完成，输出目录: {out_dir}')


if __name__ == '__main__':
    sys.exit(main())
