import os
import cv2
import numpy as np
import argparse
import random


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


def paste_A_boxes_to_B(img1, boxes1, img2, boxes2):
    """
    将img1中的边界框粘贴到img2中，覆盖img2中原有的框
    :param img1: 源图像
    :param boxes1: 源图像的YOLO格式边界框
    :param img2: 目标图像
    :param boxes2: 目标图像的YOLO格式边界框（将被忽略）
    :return: 处理后的图像和边界框
    """
    img2_h, img2_w = img2.shape[:2]
    # 只保存新添加的框，忽略原有的boxes2
    img2_boxes = []

    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        img2_h, img2_w = img2.shape[:2]

    boxes1 = yolo2voc(boxes1, *img1.shape[:2])

    # 截取图1的标签框并覆盖到图2的相同位置
    for box in boxes1:
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]

        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img2_w, x2), min(img2_h, y2)

        # 检查边界框是否有效
        if x2 <= x1 or y2 <= y1:
            continue

        # 从图1中提取边界框区域
        label1_img = img1[y1:y2, x1:x2]

        # 确保目标区域大小与源图像块匹配
        target_region = img2[y1:y2, x1:x1 + (x2 - x1)]

        # 如果源图像块和目标区域大小不匹配，调整源图像块大小
        if target_region.shape[:2] != label1_img.shape[:2]:
            label1_img_resized = cv2.resize(label1_img, (target_region.shape[1], target_region.shape[0]))
        else:
            label1_img_resized = label1_img

        # 图1框粘贴至图2，直接覆盖目标区域
        img2[y1:y2, x1:x1 + (x2 - x1)] = label1_img_resized

        # 将覆盖的框添加至图2的标签里
        img2_boxes.append([box[0], x1, y1, x1 + (x2 - x1), y1 + (y2 - y1)])

    yolo_2_boxes = voc2yolo(img2_boxes, img2_h, img2_w)
    return img2, yolo_2_boxes


def copy_A_boxes_to_B(A_images_dir, A_labels_dir, B_images_dir, B_labels_dir, output_dir, class_ids=None):
    """
    将A目录中的指定类别边界框复制到B目录中的图像上，覆盖B中原有的框
    :param A_images_dir: A图像目录
    :param A_labels_dir: A标签目录
    :param B_images_dir: B图像目录
    :param B_labels_dir: B标签目录（将被忽略）
    :param output_dir: 输出目录
    :param class_ids: 要复制的类别ID列表，如果为None则复制所有类别
    """
    # 创建输出目录
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 获取B目录中的所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    B_image_files = []
    for ext in image_extensions:
        B_image_files.extend([f for f in os.listdir(B_images_dir) if f.lower().endswith(ext)])

    # 获取A目录中的所有图像文件
    A_image_files = []
    for ext in image_extensions:
        A_image_files.extend([f for f in os.listdir(A_images_dir) if f.lower().endswith(ext)])

    print(f"找到 {len(B_image_files)} 张B图像和 {len(A_image_files)} 张A图像")

    # 处理每张B图像
    for img_file in B_image_files:
        base_name = os.path.splitext(img_file)[0]
        B_img_path = os.path.join(B_images_dir, img_file)

        # 读取B图像
        B_image = cv2.imread(B_img_path)
        if B_image is None:
            print(f"警告: 无法读取B图像 {B_img_path}")
            continue

        # 随机选择一张A图像
        A_img_file = random.choice(A_image_files)
        A_base_name = os.path.splitext(A_img_file)[0]
        A_img_path = os.path.join(A_images_dir, A_img_file)
        A_label_path = os.path.join(A_labels_dir, A_base_name + ".txt")

        # 检查A图像和标签是否存在
        if not os.path.exists(A_img_path):
            print(f"警告: A图像不存在: {A_img_path}")
            continue
        if not os.path.exists(A_label_path):
            print(f"警告: A标签不存在: {A_label_path}")
            continue

        # 读取A图像和标签
        A_image = cv2.imread(A_img_path)
        if A_image is None:
            print(f"警告: 无法读取A图像 {A_img_path}")
            continue

        A_boxes = []
        try:
            with open(A_label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:
                        cls_id, cx, cy, w, h = map(float, data)
                        A_boxes.append([int(cls_id), cx, cy, w, h])
        except FileNotFoundError:
            print(f"警告: A标签文件 {A_label_path} 不存在")
            continue

        # 从A中提取指定类别的边界框
        if class_ids is None:
            # 如果没有指定类别，则复制所有类别
            A_class_boxes = A_boxes
        else:
            A_class_boxes = [box for box in A_boxes if box[0] in class_ids]

        if not A_class_boxes:
            print(f"警告: A图像 {A_img_file} 中没有指定类别的边界框")
            continue

        # 应用粘贴操作
        result_image, result_boxes = paste_A_boxes_to_B(A_image, A_class_boxes, B_image.copy(), [])

        # 保存结果
        cv2.imwrite(os.path.join(output_images_dir, img_file), result_image)

        # 保存标签文件（只包含新添加的框）
        with open(os.path.join(output_labels_dir, base_name + ".txt"), 'w') as f:
            for box in result_boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        print(f"处理完成: {img_file} (使用A图像: {A_img_file})")


def main():
    parser = argparse.ArgumentParser(description="YOLO数据增强工具")

    A_images_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_0"
    A_labels_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_0_labels"

    B_images_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_1"
    B_labels_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_1_labels"

    output_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\copy_paste\a2b"

    parser.add_argument("--A_images_dir", type=str, default=A_images_dir, help="A图像目录")
    parser.add_argument("--A_labels_dir", type=str, default=A_labels_dir, help="A标签目录")
    parser.add_argument("--B_images_dir", type=str, default=B_images_dir, help="B图像目录")
    parser.add_argument("--B_labels_dir", type=str, default=B_labels_dir, help="B标签目录（将被忽略）")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="输出目录")
    parser.add_argument("--class_ids", type=int, nargs='+', default=None, help="要复制的类别ID列表，多个用空格分隔，如果不指定则复制所有类别")

    args = parser.parse_args()

    print(f"开始处理...")
    print(f"A图像目录: {args.A_images_dir}")
    print(f"A标签目录: {args.A_labels_dir}")
    print(f"B图像目录: {args.B_images_dir}")
    print(f"B标签目录: {args.B_labels_dir}（将被忽略）")
    print(f"输出目录: {args.output_dir}")
    print(f"类别ID: {args.class_ids}")

    copy_A_boxes_to_B(
        args.A_images_dir,
        args.A_labels_dir,
        args.B_images_dir,
        args.B_labels_dir,
        args.output_dir,
        args.class_ids
    )

    print("处理完成!")


if __name__ == "__main__":
    main()