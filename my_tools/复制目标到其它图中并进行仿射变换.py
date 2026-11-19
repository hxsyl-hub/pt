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
    yolo_boxes = np.asarray(yolo_boxes, dtype=float)
    if yolo_boxes.size == 0:
        # 空框列表直接返回空数组（旧版此处 IndexError）
        return np.zeros((0, 5), dtype=int)
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
    voc_boxes = np.asarray(voc_boxes, dtype=float)
    if voc_boxes.size == 0:
        # 空框列表直接返回空数组（旧版此处 IndexError）
        return np.zeros((0, 5), dtype=float)
    yolo_boxes = voc_boxes.astype(float).copy()
    yolo_boxes[..., 1] = (voc_boxes[..., 1] + voc_boxes[..., 3]) / 2 / iw
    yolo_boxes[..., 2] = (voc_boxes[..., 2] + voc_boxes[..., 4]) / 2 / ih
    yolo_boxes[..., 3] = (voc_boxes[..., 3] - voc_boxes[..., 1]) / iw
    yolo_boxes[..., 4] = (voc_boxes[..., 4] - voc_boxes[..., 2]) / ih
    # 将结果转为浮点数，保留6位小数
    yolo_boxes = np.around(yolo_boxes, decimals=6)
    return yolo_boxes


def random_affine_transform(image, strength=0.5):
    """
    对图像进行随机仿射变换
    :param image: 输入图像
    :param strength: 变换强度
    :return: 变换后的图像
    """
    h, w = image.shape[:2]

    # 生成随机变换矩阵
    # 随机平移
    tx = random.uniform(-strength * w, strength * w)
    ty = random.uniform(-strength * h, strength * h)

    # 随机旋转
    angle = random.uniform(-5, 5)  # 角度范围

    # 随机缩放
    scale = random.uniform(0.9, 1.1)  # 缩放范围

    # 随机剪切
    shear_x = random.uniform(-0.1, 0.1)
    shear_y = random.uniform(-0.1, 0.1)

    # 构建变换矩阵
    M = np.array([[scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)) + shear_x, tx],
                  [scale * np.sin(np.radians(angle)) + shear_y, scale * np.cos(np.radians(angle)), ty]])

    # 应用仿射变换
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return transformed, M


def affine_transformed_rect(M, w, h):
    """
    根据仿射矩阵计算目标变换后在图块内的实际可见外接框。

    warpAffine 会让目标在图块内平移/旋转/剪切，飞出画布的部分被裁掉、
    由边界填充补上。标签必须跟着实际可见像素走，否则就是错位标注。

    :param M: 2x3 仿射矩阵
    :param w: 图块宽
    :param h: 图块高
    :return: (x1, y1, x2, y2)，已 clamp 到图块范围内
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    pts = np.hstack([corners, np.ones((4, 1))]) @ M.T  # (4, 2)
    x1 = max(0, int(np.floor(pts[:, 0].min())))
    y1 = max(0, int(np.floor(pts[:, 1].min())))
    x2 = min(w, int(np.ceil(pts[:, 0].max())))
    y2 = min(h, int(np.ceil(pts[:, 1].max())))
    return x1, y1, x2, y2


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
    # 记录已贴入的区域，用于重叠检查（后贴覆盖先贴会产生"幽灵框"）
    pasted_rects = []

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

        # 计算边界框的中心和尺寸
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_w = x2 - x1
        box_h = y2 - y1

        # 放大4倍
        new_box_w = box_w * 3
        new_box_h = box_h * 3

        # 计算放大后的边界框坐标
        new_x1 = max(0, center_x - new_box_w // 2)
        new_y1 = max(0, center_y - new_box_h // 2)
        new_x2 = min(img2_w, center_x + new_box_w // 2)
        new_y2 = min(img2_h, center_y + new_box_h // 2)

        # 从图1中提取边界框区域
        label1_img = img1[y1:y2, x1:x2]

        # 放大边界框区域
        label1_img = cv2.resize(label1_img, (new_box_w, new_box_h), interpolation=cv2.INTER_CUBIC)

        # 与已贴入区域重叠过多时跳过，避免后贴覆盖先贴的目标像素产生"幽灵框"
        patch_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        overlapped = False
        for px1, py1, px2, py2 in pasted_rects:
            ix = max(0, min(new_x2, px2) - max(new_x1, px1))
            iy = max(0, min(new_y2, py2) - max(new_y1, py1))
            if ix * iy > 0.3 * patch_area:
                overlapped = True
                break
        if overlapped:
            print(f"跳过重叠目标: 与已贴入区域重叠过大 (box at {new_x1},{new_y1})")
            continue

        # 应用随机仿射变换（同时拿到变换矩阵，用于修正标签位置）
        label1_img, M = random_affine_transform(label1_img)

        # 变换后目标实际可见的外接框（图块内坐标）
        ax1, ay1, ax2, ay2 = affine_transformed_rect(M, new_box_w, new_box_h)
        if ax2 - ax1 < 2 or ay2 - ay1 < 2:
            # 目标几乎完全平移/旋出图块，贴上去只剩背景噪声，跳过
            print(f"跳过无效目标: 变换后目标几乎不可见 (box at {new_x1},{new_y1})")
            continue

        # 只裁出目标实际可见的部分，按实际落点粘贴
        dst_x1, dst_y1 = new_x1 + ax1, new_y1 + ay1
        dst_x2, dst_y2 = new_x1 + ax2, new_y1 + ay2

        # 落点同样要裁剪到图像范围内（3倍区域在图像边缘会被截断）
        clip_x1, clip_y1 = max(dst_x1, 0), max(dst_y1, 0)
        clip_x2, clip_y2 = min(dst_x2, img2_w), min(dst_y2, img2_h)
        if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
            continue

        patch = label1_img[ay1:ay1 + (clip_y2 - clip_y1), ax1:ax1 + (clip_x2 - clip_x1)]
        img2[clip_y1:clip_y2, clip_x1:clip_x2] = patch

        # 标签跟随实际可见的目标位置，而不是整个放大区域
        img2_boxes.append([box[0], clip_x1, clip_y1, clip_x2, clip_y2])
        pasted_rects.append([clip_x1, clip_y1, clip_x2, clip_y2])

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
    if not A_image_files:
        print(f"错误: A图像目录中没有图像: {A_images_dir}")
        return
    if not B_image_files:
        print(f"错误: B图像目录中没有图像: {B_images_dir}")
        return

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
        cv2.imwrite(os.path.join(output_images_dir, "1_"+img_file), result_image)

        # 保存标签文件（只包含新添加的框）
        with open(os.path.join(output_labels_dir, "1_"+base_name + ".txt"), 'w') as f:
            for box in result_boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        print(f"处理完成: {img_file} (使用A图像: {A_img_file})")


def main():
    parser = argparse.ArgumentParser(description="YOLO数据增强工具")

    A_images_dir = r"D:\python\gt\label_dataset\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_0"
    A_labels_dir = r"D:\python\gt\label_dataset\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_0_labels"

    B_images_dir = r"D:\python\gt\label_dataset\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_1"
    B_labels_dir = r"D:\python\gt\label_dataset\dataset_2\real_train_Re_labeledjust_0\copy_paste\images_1_labels"

    output_dir = r"D:\python\gt\label_dataset\dataset_2\real_train_Re_labeledjust_0\copy_paste\a2b"

    parser.add_argument("--A_images_dir", type=str, default=A_images_dir, help="A图像目录")
    parser.add_argument("--A_labels_dir", type=str, default=A_labels_dir, help="A标签目录")
    parser.add_argument("--B_images_dir", type=str, default=B_images_dir, help="B图像目录")
    parser.add_argument("--B_labels_dir", type=str, default=B_labels_dir, help="B标签目录（将被忽略）")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="输出目录")
    parser.add_argument("--class_ids", type=int, nargs='+', default=[0], help="要复制的类别ID列表，多个用空格分隔，如果不指定则复制所有类别")

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