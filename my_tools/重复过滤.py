import os
import hashlib
import shutil
import uuid
import time
import random
import multiprocessing as mp
from collections import defaultdict

from PIL import Image
import imagehash
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 配置参数
SIMILARITY_THRESHOLD = 5  # 汉明距离阈值（pHash）
SSIM_THRESHOLD = 0.85  # SSIM相似度阈值（固定机位工业图背景高度相似，建议 >= 0.85，0.5 会误杀正常样本）
EXACT_DUP_DIR = "exact_duplicates"  # 完全重复文件目录
SIMILAR_DUP_DIR = "similar_duplicates"  # 视觉相似文件目录
USE_MULTIPROCESSING = True  # 是否使用多进程加速
NUM_PROCESSES = 4  # 进程数
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')


def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_phash(file_path):
    """生成感知哈希（pHash）"""
    try:
        img = Image.open(file_path)
        return imagehash.phash(img)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def ssim_check(img1_path, img2_path):
    """计算结构相似性（SSIM）"""
    try:
        # 用 imdecode 读取，兼容中文路径（cv2.imread 在 Windows 中文路径下返回 None）
        img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            return 0

        # 调整图像大小以提高计算速度
        img1 = cv2.resize(img1, (128, 128))
        img2 = cv2.resize(img2, (128, 128))

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算 SSIM（显式给出 uint8 的动态范围，避免 skimage 版本差异告警）
        ssim_value = ssim(gray1, gray2, data_range=255)
        return ssim_value
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return 0


def generate_unique_path(dest_dir, filename):
    """生成唯一文件名，避免覆盖"""
    # 如果目标目录不存在，直接返回路径
    if not os.path.exists(dest_dir):
        return os.path.join(dest_dir, filename)

    # 获取文件名和扩展名
    base, ext = os.path.splitext(filename)

    # 检查原始文件名是否可用
    original_path = os.path.join(dest_dir, filename)
    if not os.path.exists(original_path):
        return original_path

    # 使用UUID生成唯一标识符，避免循环
    unique_id = str(uuid.uuid4())[:8]  # 取前8位，通常足够唯一
    new_filename = f"{base}_{unique_id}{ext}"
    new_path = os.path.join(dest_dir, new_filename)

    # 确保新路径也不存在（极小概率事件）
    if os.path.exists(new_path):
        # 如果仍然存在，使用时间戳+随机数
        timestamp = int(time.time() * 1000)
        random_num = random.randint(0, 9999)
        new_filename = f"{base}_{timestamp}_{random_num}{ext}"
        new_path = os.path.join(dest_dir, new_filename)

    return new_path


def _safe_mtime(path):
    """取 mtime，文件消失时返回 inf（排序时不会被选为保留者）"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return float('inf')


def list_image_files(directory):
    """收集目录下的图像文件。

    关键：剪掉输出目录（exact_duplicates / similar_duplicates），
    防止上次移动进归档的文件被再次扫描——否则保留的原图会和自己的副本
    组成"相似对"而被误移，归档目录也会被反复搬空。
    """
    exclude_dirs = {EXACT_DUP_DIR, SIMILAR_DUP_DIR}
    files = []
    for root, dirs, names in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]  # 剪枝
        for f in names:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                files.append(os.path.join(root, f))
    return files


def find_yolo_label_for_image(image_path):
    """按 YOLO 目录约定查找同名标签（.../images/.../x.jpg -> .../labels/.../x.txt）"""
    parts = os.path.abspath(image_path).split(os.sep)
    if 'images' not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index('images')  # 最后一个 images 段
    label_path = os.sep.join(parts[:i] + ['labels'] + parts[i + 1:])
    label_path = os.path.splitext(label_path)[0] + '.txt'
    return label_path if os.path.exists(label_path) else None


def move_files(files, dest_dir, exclude_files=None):
    """移动文件到目标目录，并随同移动同名 YOLO 标签（保持图文配对）。

    返回 [(源路径, 目标路径), ...]，供移动清单/回滚使用。
    """
    exclude_files = exclude_files or set()
    moved = []
    for file in files:
        if file in exclude_files or not os.path.exists(file):
            continue
        filename = os.path.basename(file)
        dest_path = generate_unique_path(dest_dir, filename)
        try:
            shutil.move(file, dest_path)
            moved.append((file, dest_path))
            print(f"Moved: {file} -> {dest_path}")
        except Exception as e:
            print(f"Failed to move {file}: {e}")
            continue

        # 同步移动对应标签，标签名跟随图片的新名字
        label_path = find_yolo_label_for_image(file)
        if label_path:
            label_dest_name = os.path.splitext(os.path.basename(dest_path))[0] + '.txt'
            label_dest = generate_unique_path(dest_dir, label_dest_name)
            try:
                shutil.move(label_path, label_dest)
                print(f"Moved label: {label_path} -> {label_dest}")
            except Exception as e:
                print(f"Failed to move label {label_path}: {e}")
    return moved


def find_exact_duplicates(directory):
    """查找完全相同的文件（MD5哈希）"""
    md5_dict = defaultdict(list)
    for path in list_image_files(directory):
        file_hash = get_file_hash(path)
        md5_dict[file_hash].append(path)

    # 返回有重复的文件组
    return [files for files in md5_dict.values() if len(files) > 1]


def find_similar_duplicates(directory):
    """查找视觉相似文件（pHash + SSIM）。

    只把昂贵的 pHash 计算放入多进程；两两汉明比较是纯整数运算，
    在主进程全量进行——分块比较会漏掉跨块的相似对。
    """
    image_files = list_image_files(directory)

    if USE_MULTIPROCESSING and len(image_files) > 100:
        print(f"使用多进程计算 {len(image_files)} 个文件的 pHash...")
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(get_phash, image_files)
    else:
        results = [get_phash(p) for p in image_files]

    hashes = {}
    for path, h in zip(image_files, results):
        if h is not None:
            hashes[path] = h

    paths = list(hashes.keys())
    similar_pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            hamming_distance = hashes[paths[i]] - hashes[paths[j]]
            if hamming_distance <= SIMILARITY_THRESHOLD:
                # 二次验证：SSIM
                ssim_value = ssim_check(paths[i], paths[j])
                if ssim_value >= SSIM_THRESHOLD:
                    similar_pairs.append((paths[i], paths[j], hamming_distance, ssim_value))

    return similar_pairs


def move_similar_clusters(similar_pairs, dest_dir, exclude_files=None):
    """把相似对合并成簇（并查集），每簇只保留一张（mtime 最旧），其余移走。

    旧逻辑把每对的两个端点都移走，会导致相似簇"全员消失"。
    """
    members = set()
    for pair in similar_pairs:
        members.add(pair[0])
        members.add(pair[1])
    if not members:
        return []

    parent = {f: f for f in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in similar_pairs:
        ra, rb = find(pair[0]), find(pair[1])
        if ra != rb:
            parent[ra] = rb

    clusters = defaultdict(list)
    for f in members:
        clusters[find(f)].append(f)

    moved = []
    for group in clusters.values():
        keep = min(group, key=lambda p: (_safe_mtime(p), p))  # 保留最旧一张
        to_move = [p for p in group if p != keep]
        print(f"相似簇共 {len(group)} 张，保留: {keep}")
        moved.extend(move_files(to_move, dest_dir, exclude_files))
    return moved


def find_and_move_duplicates(target_dir, mode='hash_only'):
    """
    查找并移动重复文件

    参数:
    target_dir: 目标目录路径
    mode: 去重模式，可选 'hash_only'（仅哈希去重）或 'both'（哈希+视觉相似去重）
    """
    exact_dest = os.path.join(target_dir, EXACT_DUP_DIR)
    similar_dest = os.path.join(target_dir, SIMILAR_DUP_DIR)
    os.makedirs(exact_dest, exist_ok=True)

    # 只有当选择两种模式时才创建视觉相似目录
    if mode == 'both':
        os.makedirs(similar_dest, exist_ok=True)

    print("查找完全重复文件...")
    start_time = time.time()
    exact_dups = find_exact_duplicates(target_dir)
    exact_time = time.time() - start_time
    print(f"找到 {len(exact_dups)} 组完全重复文件，耗时: {exact_time:.2f}秒")

    # 移动完全重复文件（每组保留 mtime 最旧的一张，结果可复现）
    exact_files_to_move = []
    for group in exact_dups:
        keep = min(group, key=lambda p: (_safe_mtime(p), p))
        exact_files_to_move.extend(p for p in group if p != keep)

    moved_exact_pairs = move_files(exact_files_to_move, exact_dest)
    moved_exact = [src for src, _ in moved_exact_pairs]

    all_moved_pairs = list(moved_exact_pairs)
    moved_similar = []

    # 只有当选择两种模式时才进行视觉相似检测
    if mode == 'both':
        print("查找视觉相似文件...")
        start_time = time.time()
        similar_dups = find_similar_duplicates(target_dir)
        similar_time = time.time() - start_time
        print(f"找到 {len(similar_dups)} 对视觉相似文件，耗时: {similar_time:.2f}秒")

        moved_similar_pairs = move_similar_clusters(similar_dups, similar_dest)
        moved_similar = [src for src, _ in moved_similar_pairs]
        all_moved_pairs.extend(moved_similar_pairs)

    # 写移动清单（源路径 -> 目标路径），便于审计与回滚
    if all_moved_pairs:
        log_path = os.path.join(
            target_dir, f"dedup_moved_{time.strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            for src, dst in all_moved_pairs:
                f.write(f"{src}\t{dst}\n")
        print(f"移动清单已写入: {log_path}")

    return moved_exact, moved_similar


if __name__ == "__main__":
    target_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0"

    moved_exact, moved_similar = find_and_move_duplicates(target_dir, mode='both')

    print(f"\n移动完成！"
          f"\n- 完全重复文件：{len(moved_exact)} 个"
          f"\n- 视觉相似文件：{len(moved_similar)} 个"
          f"\n查看目录："
          f"\n- {os.path.join(target_dir, EXACT_DUP_DIR)}"
          f"\n- {os.path.join(target_dir, SIMILAR_DUP_DIR)}")
