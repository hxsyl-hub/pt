import cv2
import numpy as np
import os
import random


def random_sharpen(image, boxes, alpha, ksize=(35, 35), gamma=0):
    """
    随机选择中值、均值或高斯模糊进行图像锐化

    参数:
    image: 输入图像
    boxes: 边界框（保持不变）
    alpha: 锐化强度（>0 增强细节；输出 = (1+alpha)*原图 - alpha*模糊图）
    ksize: 模糊核大小
    gamma: 添加到加权和的标量

    返回:
    锐化后的图像和原始边界框
    """
    # 随机选择模糊方法
    blur_method = random.choice(['mean', 'median', 'gaussian'])

    if blur_method == 'mean':
        # 均值模糊
        blurred = cv2.blur(image, ksize)
    elif blur_method == 'median':
        # 中值模糊 - ksize需要是奇数
        ksize_single = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
        blurred = cv2.medianBlur(image, ksize_single)
    elif blur_method == 'gaussian':
        # 高斯模糊 - ksize需要是奇数
        ksize_single = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
        blurred = cv2.GaussianBlur(image, (ksize_single, ksize_single), 0)

    # USM 锐化：out = (1 + alpha) * 原图 - alpha * 模糊图
    # （旧公式 addWeighted(blurred, alpha, image, 1-alpha) 在 alpha=2.5 时等价于
    #   2.5*模糊 - 1.5*原图，高频细节被反向叠加，输出是"细节反转"的图而不是锐化）
    img_sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, gamma)

    print(f"使用 {blur_method} 模糊方法，核大小: {ksize}")

    return img_sharpened, boxes


def process_directory_images(input_dir, output_dir, alpha=1.5, ksize=(35, 35), gamma=0):
    """
    处理目录中的所有图像

    参数:
    input_dir: 输入图像目录
    output_dir: 输出图像目录
    alpha: 锐化权重
    ksize: 模糊核大小
    gamma: 添加到加权和的标量
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    for i, filename in enumerate(image_files):
        # 读取图像
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"无法读取图像: {filename}")
            continue

        # 应用随机锐化（boxes参数设为空列表，因为这里不需要）
        sharpened_image, _ = random_sharpen(image, [], alpha, ksize, gamma)

        # 保存处理后的图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, sharpened_image)

        print(f"处理完成 ({i + 1}/{len(image_files)}): {filename}")

    print("所有图像处理完成！")


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\train_ng_sys_0911\false_negatives\images"  # 替换为你的输入目录
    output_directory = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\train_ng_sys_0911\false_negatives\images2"  # 替换为你的输出目录

    # 处理参数
    alpha_value = 1.5  # 锐化强度（1.0~1.5 常用；过大会出现光晕/振铃）
    kernel_size = (65, 65)  # 模糊核大小
    gamma_value = 0  # gamma校正值

    # 处理目录中的所有图像
    process_directory_images(input_directory, output_directory,
                             alpha_value, kernel_size, gamma_value)