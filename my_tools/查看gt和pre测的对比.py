import os
import cv2
import argparse
import numpy as np
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


def parse_yolo_label(label_path):
    """
    解析YOLO标签文件
    :param label_path: 标签文件路径
    :return: 包含边界框信息的列表 [class_id, x_center, y_center, width, height]
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


def draw_yolo_boxes(image, boxes, color, label_prefix="", class_names=None):
    """
    在图像上绘制YOLO边界框
    :param image: 输入图像
    :param boxes: 边界框列表
    :param color: 边界框颜色 (B, G, R)
    :param label_prefix: 标签前缀
    :param class_names: 类别名称列表
    :return: 绘制后的图像
    """
    img_height, img_width = image.shape[:2]

    for box in boxes:
        cls_id, cx, cy, w, h = box
        # 转换为像素坐标
        x1 = int((cx - w / 2) * img_width)
        y1 = int((cy - h / 2) * img_height)
        x2 = int((cx + w / 2) * img_width)
        y2 = int((cy + h / 2) * img_height)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        if class_names and cls_id < len(class_names):
            label = f"{label_prefix}{class_names[cls_id]}"
        else:
            label = f"{label_prefix}{cls_id}"

        # 绘制标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)

        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def move_files(base_dir, img_file, move_dir):
    """
    移动图像和标签文件到指定目录
    :param base_dir: 基础目录
    :param img_file: 图像文件名
    :param move_dir: 移动目标目录
    """
    base_name = os.path.splitext(img_file)[0]

    # 源文件路径
    img_path = os.path.join(base_dir, "images", img_file)
    prelabel_path = os.path.join(base_dir, "prelabels", base_name + ".txt")
    gtlabel_path = os.path.join(base_dir, "gtlabels", base_name + ".txt")

    # 目标目录
    move_img_dir = os.path.join(move_dir, "images")
    move_prelabel_dir = os.path.join(move_dir, "prelabels")
    move_gtlabel_dir = os.path.join(move_dir, "gtlabels")

    # 创建目标目录
    for dir_path in [move_img_dir, move_prelabel_dir, move_gtlabel_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 移动文件
    files_moved = []
    try:
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(move_img_dir, img_file))
            files_moved.append(f"图像: {img_file}")

        if os.path.exists(prelabel_path):
            shutil.move(prelabel_path, os.path.join(move_prelabel_dir, base_name + ".txt"))
            files_moved.append(f"预测标签: {base_name}.txt")

        if os.path.exists(gtlabel_path):
            shutil.move(gtlabel_path, os.path.join(move_gtlabel_dir, base_name + ".txt"))
            files_moved.append(f"真实标签: {base_name}.txt")

        print(f"已移动文件: {', '.join(files_moved)}")
        return True
    except Exception as e:
        print(f"移动文件时出错: {e}")
        return False


def visualize_labels(base_dir, output_dir=None, scale_factor=0.5, class_names=None):
    """
    可视化YOLO标签
    :param base_dir: 包含images, prelabels, gtlabels的目录
    :param output_dir: 输出目录，如果为None则不保存
    :param scale_factor: 图像缩放因子
    :param class_names: 类别名称列表
    """
    # 检查目录结构
    images_dir = os.path.join(base_dir, "images")
    prelabels_dir = os.path.join(base_dir, "prelabels")
    gtlabels_dir = os.path.join(base_dir, "gtlabels")

    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return

    # 创建移动目录
    move_dir = os.path.join(os.path.dirname(base_dir), "move_result_imgs_label")
    os.makedirs(move_dir, exist_ok=True)
    print(f"按 'm' 键移动文件到: {move_dir}")

    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])

    print(f"找到 {len(image_files)} 张图像")

    # 处理每张图像
    for i, img_file in enumerate(image_files):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        prelabel_path = os.path.join(prelabels_dir, base_name + ".txt")
        gtlabel_path = os.path.join(gtlabels_dir, base_name + ".txt")

        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在 {img_path}")
            continue

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        # 读取预测标签
        pred_boxes = parse_yolo_label(prelabel_path)

        # 读取真实标签
        gt_boxes = parse_yolo_label(gtlabel_path)

        # 绘制预测框（红色）
        if pred_boxes:
            image = draw_yolo_boxes(image, pred_boxes, (0, 0, 255), "Pred: ", class_names)

        # 绘制真实框（绿色）
        if gt_boxes:
            image = draw_yolo_boxes(image, gt_boxes, (0, 255, 0), "GT: ", class_names)

        # 添加状态信息
        status_text = f"Image {i + 1}/{len(image_files)}: {img_file}"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'm' to move, 'q' to quit, any key to continue", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 调整图像大小
        if scale_factor != 1.0:
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height))

        # 显示图像
        cv2.imshow("YOLO Labels Visualization", image)

        # 保存图像
        if output_dir:
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, image)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('m'):  # 按m移动文件
            success = move_files(base_dir, img_file, move_dir)
            if success:
                # 从列表中移除已移动的文件
                image_files.remove(img_file)
                # 如果列表为空，退出循环
                if not image_files:
                    break
                # 重置索引，因为列表已更改
                i -= 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="可视化YOLO标签")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="包含images, prelabels, gtlabels的目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，如果不指定则不保存")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="图像缩放因子 (默认: 0.5)")
    parser.add_argument("--class_names", type=str, nargs='+', default=None,
                        help="类别名称列表，例如 'cat' 'dog' 'person'")

    args = parser.parse_args()

    print(f"开始可视化标签...")
    print(f"基础目录: {args.base_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"缩放因子: {args.scale}")
    print(f"类别名称: {args.class_names}")

    visualize_labels(
        args.base_dir,
        args.output_dir,
        args.scale,
        args.class_names
    )

    print("可视化完成!")


if __name__ == "__main__":
    """
        false positives：假阳性， 负例，认成了正例。误检

        false negatives：假阴性，正例，认成了负例。漏检

    """
    # 示例使用
    base_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\train_ng_sys_0911\false_negatives"
    output_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\train_ng_sys_0911\false_negatives_visualization"

    # 如果通过命令行运行，则使用参数解析
    # 如果直接运行脚本，则使用上面的默认值
    if True == 1:
        # 使用默认值
        visualize_labels(
            base_dir,
            output_dir,
            0.5,  # 缩放因子
            None  # 类别名称
        )
    else:
        main()