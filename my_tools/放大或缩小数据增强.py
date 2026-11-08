import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def yolo2voc(yolo_boxes, ih, iw):
    """
    将YOLO格式的边界框转换为VOC格式
    :param yolo_boxes: YOLO格式的边界框 [class_id, cx, cy, w, h]
    :param ih: 图像高度
    :param iw: 图像宽度
    :return: VOC格式的边界框 [class_id, xmin, ymin, xmax, ymax]
    """
    if isinstance(yolo_boxes, list):
        yolo_boxes = np.array(yolo_boxes)
    voc_boxes = yolo_boxes.copy()
    voc_boxes[..., 1] = yolo_boxes[..., 1] * iw - yolo_boxes[..., 3] * iw / 2
    voc_boxes[..., 2] = yolo_boxes[..., 2] * ih - yolo_boxes[..., 4] * ih / 2
    voc_boxes[..., 3] = yolo_boxes[..., 1] * iw + yolo_boxes[..., 3] * iw / 2
    voc_boxes[..., 4] = yolo_boxes[..., 2] * ih + yolo_boxes[..., 4] * ih / 2
    # 将坐标转换为整数
    voc_boxes = voc_boxes.astype(int)
    return voc_boxes


def voc2yolo(voc_boxes, ih, iw):
    """
    将VOC格式的边界框转换为YOLO格式
    :param voc_boxes: VOC格式的边界框 [class_id, xmin, ymin, xmax, ymax]
    :param ih: 图像高度
    :param iw: 图像宽度
    :return: YOLO格式的边界框 [class_id, cx, cy, w, h]
    """
    if isinstance(voc_boxes, list):
        voc_boxes = np.array(voc_boxes)
    yolo_boxes = voc_boxes.astype(float).copy()
    yolo_boxes[..., 1] = (voc_boxes[..., 1] + voc_boxes[..., 3]) / 2 / iw
    yolo_boxes[..., 2] = (voc_boxes[..., 2] + voc_boxes[..., 4]) / 2 / ih
    yolo_boxes[..., 3] = (voc_boxes[..., 3] - voc_boxes[..., 1]) / iw
    yolo_boxes[..., 4] = (voc_boxes[..., 4] - voc_boxes[..., 2]) / ih
    # 将结果转为浮点数，保留6位小数
    yolo_boxes = np.around(yolo_boxes, decimals=6)
    return yolo_boxes


def letterbox(image, size=(2048, 2048), boxes=None):
    """
    将图像缩放到指定大小，保持宽高比，并用灰色填充
    :param image: 输入图像
    :param size: 目标大小 (width, height)
    :param boxes: YOLO格式的边界框
    :return: 处理后的图像和边界框
    """
    ih, iw = image.shape[:2]
    w, h = size

    if ih == h and iw == w:
        return image, boxes, 1, 0, 0, 0, 0  # image, boxes, ratio, padT, padB, padL, padR

    ratio = min(w / iw, h / ih)

    new_w = int(iw * ratio)
    new_h = int(ih * ratio)

    dw = (w - new_w) // 2
    dh = (h - new_h) // 2

    scale_img = cv2.resize(image, [new_w, new_h])

    padded_image = cv2.copyMakeBorder(scale_img, dh, h - dh - new_h, dw, w - dw - new_w, cv2.BORDER_CONSTANT,
                                      value=[114, 114, 114])

    padded_boxes = None
    if boxes is not None:
        voc_boxes = yolo2voc(boxes, ih, iw)
        padded_boxes = voc_boxes.copy()
        padded_boxes[:, 1] = voc_boxes[:, 1] * ratio + dw
        padded_boxes[:, 2] = voc_boxes[:, 2] * ratio + dh
        padded_boxes[:, 3] = voc_boxes[:, 3] * ratio + dw
        padded_boxes[:, 4] = voc_boxes[:, 4] * ratio + dh
        padded_boxes = voc2yolo(padded_boxes, h, w)

    return padded_image, padded_boxes, ratio, dh, h - dh - new_h, dw, w - dw - new_w


def cut_box(img, boxes, expand_ratio=1.0):
    """
    从图像中裁剪出包含边界框的区域，并扩大边界
    :param img: 输入图像
    :param boxes: YOLO格式的边界框
    :param expand_ratio: 边界扩大比例
    :return: 裁剪后的图像和边界框列表
    """
    height, width = img.shape[:2]
    cropped_images_and_boxes = []

    if boxes is None or len(boxes) == 0:
        return cropped_images_and_boxes

    boxes = yolo2voc(boxes, height, width)

    for box in boxes:
        # 计算边界框的宽度和高度
        box_w = box[3] - box[1]
        box_h = box[4] - box[2]

        # 计算扩大的边界
        expand_w = int(box_w * expand_ratio / 2)
        expand_h = int(box_h * expand_ratio / 2)

        # 计算新的边界
        new_x_min = max(0, box[1] - expand_w)
        new_y_min = max(0, box[2] - expand_h)
        new_x_max = min(width, box[3] + expand_w)
        new_y_max = min(height, box[4] + expand_h)

        # 裁剪图像
        cropped_img = img[new_y_min:new_y_max, new_x_min:new_x_max, :]

        # 调整边界框坐标
        box_x_min = box[1] - new_x_min
        box_y_min = box[2] - new_y_min
        box_x_max = box[3] - new_x_min
        box_y_max = box[4] - new_y_min

        # 转换为YOLO格式
        new_voc_box = np.array([box[0], box_x_min, box_y_min, box_x_max, box_y_max])
        new_yolo_box = voc2yolo(new_voc_box, cropped_img.shape[0], cropped_img.shape[1])

        cropped_images_and_boxes.append((cropped_img, new_yolo_box))

    return cropped_images_and_boxes


def parse_yolo_label(label_path):
    """
    解析YOLO标签文件
    :param label_path: 标签文件路径
    :return: 包含边界框信息的列表 [class_id, cx, cy, w, h]
    """
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:
                    cls_id, cx, cy, w, h = map(float, data)
                    boxes.append([int(cls_id), cx, cy, w, h])
    except FileNotFoundError:
        print(f"警告: 标签文件 {label_path} 不存在")
    return boxes


def save_yolo_label(label_path, boxes):
    """
    保存YOLO格式的标签文件
    :param label_path: 标签文件路径
    :param boxes: 边界框列表
    """
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def random_color_augmentation(image):
    """
    对图像进行随机颜色变化
    :param image: 输入图像
    :return: 颜色变化后的图像
    """
    # 随机亮度变化
    brightness = np.random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # 随机对比度变化
    contrast = np.random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    # 随机饱和度变化 (转换为HSV空间处理)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.random.uniform(0.7, 1.3)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 随机色调变化
    hue_shift = np.random.randint(-10, 10)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image

def process_images(input_images_dir, input_labels_dir, output_dir, target_size=(2048, 2048), expand_ratio=1.0):
    """
    处理图像和标签
    :param input_images_dir: 输入图像目录
    :param input_labels_dir: 输入标签目录
    :param output_dir: 输出目录
    :param target_size: 目标大小
    :param expand_ratio: 边界扩大比例
    """
    # 创建输出目录
    letterbox_images_dir = os.path.join(output_dir, "letterbox_images")
    letterbox_labels_dir = os.path.join(output_dir, "letterbox_labels")
    cropped_images_dir = os.path.join(output_dir, "cropped_images")
    cropped_labels_dir = os.path.join(output_dir, "cropped_labels")

    for dir_path in [letterbox_images_dir, letterbox_labels_dir, cropped_images_dir, cropped_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_images_dir) if f.lower().endswith(ext)])

    print(f"找到 {len(image_files)} 张图像")

    # 处理每张图像
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(input_images_dir, img_file)
        label_path = os.path.join(input_labels_dir, base_name + ".txt")

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
        image = random_color_augmentation(image)
        # 读取标签
        boxes = parse_yolo_label(label_path)

        # 1. 应用letterbox
        letterbox_img, letterbox_boxes, _, _, _, _, _ = letterbox(image, target_size, boxes)

        # 保存letterbox结果
        cv2.imwrite(os.path.join(letterbox_images_dir, img_file), letterbox_img)
        if letterbox_boxes is not None:
            save_yolo_label(os.path.join(letterbox_labels_dir, base_name + ".txt"), letterbox_boxes)

        # 2. 应用cut_box
        cropped_results = cut_box(image, boxes, expand_ratio)

        # 保存cut_box结果
        for i, (cropped_img, cropped_box) in enumerate(cropped_results):
            cropped_img_name = f"{base_name}_{i}{os.path.splitext(img_file)[1]}"
            cropped_label_name = f"{base_name}_{i}.txt"

            cv2.imwrite(os.path.join(cropped_images_dir, cropped_img_name), cropped_img)
            save_yolo_label(os.path.join(cropped_labels_dir, cropped_label_name), [cropped_box])

        print(f"处理完成: {img_file}")

    print("所有图像处理完成!")


def main():
    parser = argparse.ArgumentParser(description="处理YOLO格式的图像和标签")
    parser.add_argument("--input_images_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--input_labels_dir", type=str, required=True, help="输入标签目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target_size", type=int, nargs=2, default=[2048, 2048],
                        help="目标大小，默认为 2048 2048")
    parser.add_argument("--expand_ratio", type=float, default=3.0,
                        help="边界扩大比例，默认为 1.0")

    args = parser.parse_args()

    print(f"开始处理图像...")
    print(f"输入图像目录: {args.input_images_dir}")
    print(f"输入标签目录: {args.input_labels_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标大小: {args.target_size}")
    print(f"边界扩大比例: {args.expand_ratio}")

    process_images(
        args.input_images_dir,
        args.input_labels_dir,
        args.output_dir,
        tuple(args.target_size),
        args.expand_ratio
    )


if __name__ == "__main__":
    # 示例使用（如果直接运行脚本而不是通过命令行）
    input_images_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\moved_val_images"
    input_labels_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\moved_val_labels"
    output_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\output"

    # 如果通过命令行运行，则使用参数解析
    # 如果直接运行脚本，则使用上面的默认值
    if True == 1:
        # 使用默认值
        process_images(
            input_images_dir,
            input_labels_dir,
            output_dir,
            (2048, 2048),  # 目标大小
            1.0  # 边界扩大比例
        )
    else:
        main()