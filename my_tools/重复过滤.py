import os
import hashlib
import shutil
from PIL import Image
import imagehash
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import time

# 配置参数
SIMILARITY_THRESHOLD = 5  # 汉明距离阈值（pHash）
SSIM_THRESHOLD = 0.5  # SSIM相似度阈值
EXACT_DUP_DIR = "exact_duplicates"  # 完全重复文件目录
SIMILAR_DUP_DIR = "similar_duplicates"  # 视觉相似文件目录
USE_MULTIPROCESSING = True  # 是否使用多进程加速
NUM_PROCESSES = 4  # 进程数


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
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return 0

        # 调整图像大小以提高计算速度
        img1 = cv2.resize(img1, (128, 128))
        img2 = cv2.resize(img2, (128, 128))

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算 SSIM
        ssim_value = ssim(gray1, gray2)
        return ssim_value
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return 0


import os
import uuid


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
        import time
        import random
        timestamp = int(time.time() * 1000)
        random_num = random.randint(0, 9999)
        new_filename = f"{base}_{timestamp}_{random_num}{ext}"
        new_path = os.path.join(dest_dir, new_filename)

    return new_path


def move_files(files, dest_dir, exclude_files=None):
    """移动文件到目标目录，排除已处理的文件"""
    exclude_files = exclude_files or set()
    moved = []
    for file in files:
        if file in exclude_files or not os.path.exists(file):
            continue
        filename = os.path.basename(file)
        dest_path = generate_unique_path(dest_dir, filename)
        try:
            shutil.move(file, dest_path)
            moved.append(file)
            print(f"Moved: {file} -> {dest_path}")
        except Exception as e:
            print(f"Failed to move {file}: {str(e)}")
    return moved


def find_exact_duplicates(directory):
    """查找完全相同的文件（MD5哈希）"""
    md5_dict = defaultdict(list)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                path = os.path.join(root, file)
                file_hash = get_file_hash(path)
                md5_dict[file_hash].append(path)

    # 返回有重复的文件组
    return [files for files in md5_dict.values() if len(files) > 1]


def find_similar_duplicates_worker(file_paths, threshold=SIMILARITY_THRESHOLD):
    """工作进程函数：查找视觉相似文件"""
    similar_pairs = []

    # 为所有文件计算phash
    phash_dict = {}
    for path in file_paths:
        phash = get_phash(path)
        if phash is not None:
            phash_dict[path] = phash

    # 检查相似性
    checked = set()
    for path1, hash1 in phash_dict.items():
        for path2, hash2 in phash_dict.items():
            if path1 != path2 and path2 not in checked:
                hamming_distance = hash1 - hash2
                if hamming_distance <= threshold:
                    # 二次验证：SSIM
                    ssim_value = ssim_check(path1, path2)
                    if ssim_value >= SSIM_THRESHOLD:
                        similar_pairs.append((path1, path2, hamming_distance, ssim_value))
        checked.add(path1)

    return similar_pairs


def find_similar_duplicates(directory):
    """查找视觉相似文件（pHash + SSIM）"""
    # 收集所有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                path = os.path.join(root, file)
                image_files.append(path)

    # 如果使用多进程，分割任务
    if USE_MULTIPROCESSING and len(image_files) > 100:
        print(f"使用多进程处理 {len(image_files)} 个文件...")
        chunk_size = max(1, len(image_files) // NUM_PROCESSES)
        chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(find_similar_duplicates_worker, chunks)

        # 合并结果
        similar_pairs = []
        for result in results:
            similar_pairs.extend(result)

        return similar_pairs
    else:
        # 单进程处理
        return find_similar_duplicates_worker(image_files)


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

    # 移动完全重复文件（每组保留第一个）
    exact_files_to_move = []
    for group in exact_dups:
        exact_files_to_move.extend(group[1:])  # 保留第一个，移动其余

    moved_exact = move_files(exact_files_to_move, exact_dest)

    moved_similar = []

    # 只有当选择两种模式时才进行视觉相似检测
    if mode == 'both':
        print("查找视觉相似文件...")
        start_time = time.time()
        similar_dups = find_similar_duplicates(target_dir)
        similar_time = time.time() - start_time
        print(f"找到 {len(similar_dups)} 对视觉相似文件，耗时: {similar_time:.2f}秒")

        # 移动视觉相似文件（排除已处理的完全重复文件）
        similar_files_to_move = set()
        for pair in similar_dups:
            similar_files_to_move.add(pair[0])
            similar_files_to_move.add(pair[1])

        # 排除已经移动的完全重复文件
        similar_files_to_move = similar_files_to_move - set(moved_exact)
        moved_similar = move_files(list(similar_files_to_move), similar_dest)

    return moved_exact, moved_similar


if __name__ == "__main__":
    target_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0"


    moved_exact, moved_similar = find_and_move_duplicates(target_dir,mode='both')

    print(f"\n移动完成！"
          f"\n- 完全重复文件：{len(moved_exact)} 个"
          f"\n- 视觉相似文件：{len(moved_similar)} 个"
          f"\n查看目录："
          f"\n- {os.path.join(target_dir, EXACT_DUP_DIR)}"
          f"\n- {os.path.join(target_dir, SIMILAR_DUP_DIR)}")