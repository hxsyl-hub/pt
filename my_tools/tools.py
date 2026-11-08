import os
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import numpy as np
import random

plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体


def analyze_label_distribution(label_dir, output_dir=None, plot_title='Label Distribution', save_plot=False):
    """
    分析标签目录中所有标签的分布情况

    参数:
    label_dir (str): 标签文件所在的目录路径
    output_dir (str): 输出结果的目录（可选）
    plot_title (str): 图表标题
    save_plot (bool): 是否保存图表

    返回:
    list: 排序后的标签分布统计结果
    """
    # 检查目录是否存在
    if not os.path.exists(label_dir):
        raise ValueError(f"目录不存在: {label_dir}")

    all_labels = []

    # 遍历目录中的所有文件
    for filename in os.listdir(label_dir):
        # 跳过classes文件
        if 'classes' in filename:
            continue

        file_path = os.path.join(label_dir, filename)

        # 确保是文件而不是目录
        if os.path.isfile(file_path):
            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    content = f.readlines()
                    for row in content:
                        # 提取标签（每行的第一个元素）
                        label = row.strip().split(' ')[0]
                        all_labels.append(label)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # 使用Counter统计标签出现次数
    label_counter = Counter(all_labels)

    # 按出现次数排序
    sorted_results = sorted(label_counter.items(), key=lambda x: x[1])

    # 打印统计结果
    print("标签分布统计:")
    for label, count in sorted_results:
        print(f"标签 {label}: {count} 次")

    # 绘制图表
    labels = [item[0] for item in sorted_results]
    counts = [item[1] for item in sorted_results]

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(plot_title)
    plt.grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count),
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # 保存图表
    if save_plot:
        if output_dir:
            output_path = os.path.join(output_dir, "label_distribution.png")
        else:
            output_path = "label_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")

    plt.show()

    return sorted_results


def letterbox(img, new_shape=640, color=(0, 0, 0)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    target_w, target_h = new_shape
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    img_padded = cv2.copyMakeBorder(img_resized, dh, target_h - new_h - dh, dw, target_w - new_w - dw,
                                    cv2.BORDER_CONSTANT, value=color)

    return img_padded, scale, dw, dh


def process_labels(label_file, scale, dw, dh, img_width, img_height):
    """
    Process the label file to adjust the bounding boxes for resized image.
    Args:
        label_file (str): The path to the label file.
        scale (float): The scaling factor applied to the image.
        dw (int): Padding width added to the image.
        dh (int): Padding height added to the image.
        img_width (int): The original width of the image.
        img_height (int): The original height of the image.
    Returns:
        list: A list of adjusted bounding boxes.
    """
    boxes = []
    with open(label_file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for row in lines:
            try:
                label = row.split(' ')
                x_center = float(label[1]) * img_width
                y_center = float(label[2]) * img_height
                width = float(label[3]) * img_width
                height = float(label[4]) * img_height

                x_center = (x_center * scale) + dw
                y_center = (y_center * scale) + dh
                width = width * scale
                height = height * scale

                boxes.append([label[0], x_center, y_center, width, height])
            except Exception as e:
                print(e)

    return boxes


def analyze_object_size_distribution(label_dir, jpg_dir, target_size=640):
    """
    分析目标检测数据集中对象尺寸的分布情况

    参数:
    label_dir: 标签文件目录路径
    jpg_dir: 图像文件目录路径
    target_size: 目标图像尺寸

    返回:
    dict: 包含各尺寸区间统计结果的字典
    """
    # 用于统计目标尺寸分布
    less_5 = 0
    less_5x5_16x16 = 0
    less_16x16_32x32 = 0
    less_32x32_64x64 = 0
    less_64x64_128x128 = 0
    less_128x128 = 0

    for i, p in enumerate(os.listdir(label_dir)):
        print(i)
        label_file = os.path.join(label_dir, p)
        jpg_file = os.path.join(jpg_dir, p[:-4] + '.jpg')

        if os.path.exists(jpg_file):
            try:
                img0 = cv2.imread(jpg_file)
                img_height, img_width = img0.shape[:2]
            except Exception as e:
                print(e)
                print(jpg_file)
                continue

            img_resized, scale, dw, dh = letterbox(img0, new_shape=target_size)

            # 处理 label 文件，获取调整后的 bounding boxes
            adjusted_boxes = process_labels(label_file, scale, dw, dh, img_width, img_height)

            # 统计尺寸分布
            for box in adjusted_boxes:
                _, x_center, y_center, width, height = box

                if width < 5 or height < 5:
                    less_5 += 1
                elif width * height >= 5 * 5 and width * height < 16 * 16:
                    less_5x5_16x16 += 1
                elif width * height >= 16 * 16 and width * height < 32 * 32:
                    less_16x16_32x32 += 1
                elif width * height >= 32 * 32 and width * height < 64 * 64:
                    less_32x32_64x64 += 1
                elif width * height >= 64 * 64 and width * height < 128 * 128:
                    less_64x64_128x128 += 1
                elif width * height >= 128 * 128:
                    less_128x128 += 1

    # 返回统计结果字典
    result = {
        'less_5': less_5,
        'less_5x5_16x16': less_5x5_16x16,
        'less_16x16_32x32': less_16x16_32x32,
        'less_32x32_64x64': less_32x32_64x64,
        'less_64x64_128x128': less_64x64_128x128,
        'less_128x128': less_128x128
    }

    # 绘制图表
    categories = list(result.keys())
    values = list(result.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color='skyblue')
    plt.xlabel('Object Size Categories')
    plt.ylabel('Frequency')
    plt.title('Object Size Distribution')
    plt.xticks(rotation=45, ha='right')

    # 在条形上显示每个类别的数量
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return result





def analyze_image_histogram(jpg_dir, num_samples=500):
    """
    分析目录下图像的颜色直方图分布

    参数:
    jpg_dir: 图像文件目录路径
    num_samples: 随机选择的图像数量，默认500张

    返回:
    hist_data: 各通道（BGR）颜色直方图的累计数据
    """
    # 获取目录下所有图像文件
    jpg_files = [f for f in os.listdir(jpg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 随机选择num_samples个文件
    selected_files = random.sample(jpg_files, min(num_samples, len(jpg_files)))

    # 用于累加图像的直方图数据
    hist_data = {'blue': np.zeros(256), 'green': np.zeros(256), 'red': np.zeros(256)}

    # 遍历随机选择的图像，计算其直方图并累加
    for file_name in selected_files:
        img_path = os.path.join(jpg_dir, file_name)

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue


        # 计算每个通道的直方图
        hist_b = cv2.calcHist([img], [0], None, [256], [10, 256])  # Blue
        hist_g = cv2.calcHist([img], [1], None, [256], [10, 256])  # Green
        hist_r = cv2.calcHist([img], [2], None, [256], [10, 256])  # Red

        # 累加各通道的直方图数据
        hist_data['blue'] += hist_b.flatten()
        hist_data['green'] += hist_g.flatten()
        hist_data['red'] += hist_r.flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))

    plt.plot(hist_data['blue'], color='blue', label='Blue Channel')
    plt.plot(hist_data['green'], color='green', label='Green Channel')
    plt.plot(hist_data['red'], color='red', label='Red Channel')

    plt.title('Color Histogram Distribution (Random 500 Images)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    # 显示图像
    plt.show()

    return hist_data


# 使用示例
if __name__ == "__main__":
    # 基本用法
    # label_dir = r"F:\data_source\3d_print_guotou\dataset_2\val\labels"
    # results = analyze_label_distribution(label_dir, plot_title='YOLO 标签分布', save_plot=True)
    jpg_path = r"F:\data_source\3d_print_guotou\dataset_2\val\images"
    txt_path = r"F:\data_source\3d_print_guotou\dataset_2\train\labels"
    # result = analyze_object_size_distribution(txt_path, jpg_path)
    # print(result)

    analyze_image_histogram(jpg_path)
