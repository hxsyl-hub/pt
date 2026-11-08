import os
import argparse
import shutil
from pathlib import Path
import numpy as np


def parse_yolo_label(file_path, has_score=False):
    """
    解析YOLO标签文件
    :param file_path: 标签文件路径
    :param has_score: 是否包含置信度分数
    :return: 包含边界框信息的列表
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) < 5:
                    continue

                if has_score:
                    if len(data) < 6:
                        continue
                    cls_id, cx, cy, w, h, score = map(float, data)
                    boxes.append([cls_id, cx, cy, w, h, score])
                else:
                    cls_id, cx, cy, w, h = map(float, data)
                    boxes.append([cls_id, cx, cy, w, h, 1.0])  # 默认分数为1.0
    except FileNotFoundError:
        pass  # 文件不存在时返回空列表

    return boxes


def calculate_iou(box1, box2):
    """
    计算两个边界框的IOU
    :param box1: [cx, cy, w, h]
    :param box2: [cx, cy, w, h]
    :return: IOU值
    """

    # 转换为中心坐标到角坐标
    def yolo_to_corners(box):
        cx, cy, w, h = box[1:5]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    box1_corners = yolo_to_corners(box1)
    box2_corners = yolo_to_corners(box2)

    # 计算交集区域
    x_left = max(box1_corners[0], box2_corners[0])
    y_top = max(box1_corners[1], box2_corners[1])
    x_right = min(box1_corners[2], box2_corners[2])
    y_bottom = min(box1_corners[3], box2_corners[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集区域
    box1_area = box1[3] * box1[4]
    box2_area = box2[3] * box2[4]
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def find_image_file(base_name, image_dirs, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """
    查找与标签文件对应的图像文件
    :param base_name: 文件基本名（不含扩展名）
    :param image_dirs: 图像目录列表
    :param extensions: 可能的图像扩展名列表
    :return: 找到的图像文件路径，如果未找到则返回None
    """
    for image_dir in image_dirs:
        for ext in extensions:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                return image_path
    return None


def compare_labels(pred_dir, gt_dir, output_dir, image_dirs, iou_threshold=0.5, score_threshold=0.0):
    """
    比较预测标签和真实标签
    :param pred_dir: 预测标签目录
    :param gt_dir: 真实标签目录
    :param output_dir: 输出目录
    :param image_dirs: 图像目录列表
    :param iou_threshold: IOU阈值
    :param score_threshold: 分数阈值
    """
    # 创建输出目录结构
    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    low_score_dir = os.path.join(output_dir, "low_score_predictions")

    # 为每个类别创建images和labels子目录
    for dir_path in [fp_dir, fn_dir, low_score_dir]:
        os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "prelabels"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "gtlabels"), exist_ok=True)

    # 获取所有标签文件
    pred_files = set(os.listdir(pred_dir))
    gt_files = set(os.listdir(gt_dir))
    all_files = pred_files.union(gt_files)

    for file_name in all_files:
        base_name = os.path.splitext(file_name)[0]

        # 查找对应的图像文件
        image_path = find_image_file(base_name, image_dirs)
        if not image_path:
            continue

        pred_boxes = parse_yolo_label(os.path.join(pred_dir, file_name), has_score=True)
        gt_boxes = parse_yolo_label(os.path.join(gt_dir, file_name), has_score=False)

        # 处理低分数预测
        low_score_boxes = [box for box in pred_boxes if box[5] < score_threshold]
        if low_score_boxes:
            # 保存预测标签文件（去除置信度）
            with open(os.path.join(low_score_dir, "prelabels", file_name), 'w') as f:
                for box in low_score_boxes:
                    f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 保存真实标签文件
            if gt_boxes:
                with open(os.path.join(low_score_dir, "gtlabels", file_name), 'w') as f:
                    for box in gt_boxes:
                        f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 复制图像文件
            image_ext = os.path.splitext(image_path)[1]
            shutil.copy2(image_path, os.path.join(low_score_dir, "images", base_name + image_ext))

        # 过滤掉低分数预测
        pred_boxes = [box for box in pred_boxes if box[5] >= score_threshold]

        # 标记匹配的预测和真实框
        matched_pred = [False] * len(pred_boxes)
        matched_gt = [False] * len(gt_boxes)

        # 计算IOU并匹配框
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold and not matched_gt[j]:
                    matched_pred[i] = True
                    matched_gt[j] = True
                    break

        # 保存假阳性（预测有但真实没有）
        fp_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if not matched_pred[i]]
        if fp_boxes:
            # 保存预测标签文件（去除置信度）
            with open(os.path.join(fp_dir, "prelabels", file_name), 'w') as f:
                for box in fp_boxes:
                    f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 保存真实标签文件
            if gt_boxes:
                with open(os.path.join(fp_dir, "gtlabels", file_name), 'w') as f:
                    for box in gt_boxes:
                        f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 复制图像文件
            image_ext = os.path.splitext(image_path)[1]
            shutil.copy2(image_path, os.path.join(fp_dir, "images", base_name + image_ext))

        # 保存假阴性（真实有但预测没有）
        fn_boxes = [gt_boxes[j] for j in range(len(gt_boxes)) if not matched_gt[j]]
        if fn_boxes:
            # 保存预测标签文件（去除置信度）
            if pred_boxes:
                with open(os.path.join(fn_dir, "prelabels", file_name), 'w') as f:
                    for box in pred_boxes:
                        f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 保存真实标签文件
            with open(os.path.join(fn_dir, "gtlabels", file_name), 'w') as f:
                for box in fn_boxes:
                    f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

            # 复制图像文件
            image_ext = os.path.splitext(image_path)[1]
            shutil.copy2(image_path, os.path.join(fn_dir, "images", base_name + image_ext))


def main():
    """
    false positives：假阳性， 负例，认成了正例。误检

    false negatives：假阴性，正例，认成了负例。漏检

    """
    # 预测标签目录
    pred_dir = r"D:\26--ppt\3dprint_guotou\yolov5-7.0\runs\detect\exp7\labels"
    # 真实标签目录
    gt_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\labels\train"
    # 预测对应图片目录
    image_dirs = [r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\images\train"]
    # 漏检和误检输出目录
    output_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeled\train_ng_sys_0911"

    parser = argparse.ArgumentParser(description="比较YOLO预测标签和真实标签")
    parser.add_argument("--pred_dir", type=str, default=pred_dir, help="预测标签目录")
    parser.add_argument("--gt_dir", type=str, default=gt_dir, help="真实标签目录")
    parser.add_argument("--image_dirs", type=str, nargs='+', default=image_dirs,
                        help="图像目录列表，可以指定多个目录")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="输出目录")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IOU阈值 (默认: 0.5)")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="分数阈值，低于此值的预测会被单独保存 (默认: 0.0)")

    args = parser.parse_args()

    print(f"开始处理...")
    print(f"预测目录: {args.pred_dir}")
    print(f"真实标签目录: {args.gt_dir}")
    print(f"图像目录: {args.image_dirs}")
    print(f"输出目录: {args.output_dir}")
    print(f"IOU阈值: {args.iou_threshold}")
    print(f"分数阈值: {args.score_threshold}")

    compare_labels(
        args.pred_dir,
        args.gt_dir,
        args.output_dir,
        args.image_dirs,
        args.iou_threshold,
        args.score_threshold
    )

    print("处理完成!")


if __name__ == "__main__":
    main()