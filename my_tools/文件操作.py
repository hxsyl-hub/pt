import os
import shutil
import cv2


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


def clean_mismatched_files(image_dir, label_dir):
    """
    清理不匹配的图片和标签文件：
    1. 如果图片存在但标签不存在，删除图片
    2. 如果标签存在但图片不存在，删除标签
    """
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # 统计变量
    deleted_images = 0
    deleted_labels = 0

    # 第一步：检查所有图片文件是否有对应的标签文件
    print("检查图片文件是否有对应的标签文件...")
    for filename in os.listdir(image_dir):
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            continue

        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, base_name + '.txt')

        # 如果标签文件不存在，删除图片
        if not os.path.exists(label_path):
            image_path = os.path.join(image_dir, filename)
            try:
                os.remove(image_path)
                deleted_images += 1
                print(f"已删除无对应标签的图片: {image_path}")
            except Exception as e:
                print(f"删除图片文件失败 {image_path}: {e}")

    # 第二步：检查所有标签文件是否有对应的图片文件
    print("检查标签文件是否有对应的图片文件...")
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue

        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, filename)

        # 检查是否存在对应的图片文件
        has_corresponding_image = False
        for ext in image_extensions:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                has_corresponding_image = True
                break

        # 如果图片文件不存在，删除标签
        if not has_corresponding_image:
            try:
                os.remove(label_path)
                deleted_labels += 1
                print(f"已删除无对应图片的标签: {label_path}")
            except Exception as e:
                print(f"删除标签文件失败 {label_path}: {e}")

    return deleted_images, deleted_labels


def move_images_by_class(image_dir, label_dir, target_image_dir, target_label_dir, classes_to_move):
    """
    将包含指定类别的图片和对应的标注文件移动到目标目录

    参数:
    image_dir: 原始图片文件目录
    label_dir: 原始标注文件目录
    target_image_dir: 目标图片文件目录
    target_label_dir: 目标标注文件目录
    classes_to_move: 要移动的类别列表（整数列表）
    """
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # 统计变量
    moved_images = 0
    moved_labels = 0
    checked_files = 0

    # 创建目标目录（如果不存在）
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # 遍历标签目录中的所有文件
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue

        checked_files += 1
        label_path = os.path.join(label_dir, filename)

        # 读取标签文件内容
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"无法读取标签文件 {label_path}: {e}")
            continue

        # 检查是否包含要移动的类别
        should_move = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 1:
                try:
                    class_id = int(parts[0])
                    if class_id in classes_to_move:
                        should_move = True
                        break
                except ValueError:
                    continue

        # 如果包含要移动的类别，移动标签文件和对应的图片文件
        if should_move:
            # 移动标签文件
            try:
                target_label_path = os.path.join(target_label_dir, filename)
                shutil.move(label_path, target_label_path)
                moved_labels += 1
                print(f"已移动标签文件: {label_path} -> {target_label_path}")
            except Exception as e:
                print(f"移动标签文件失败 {label_path}: {e}")

            # 查找并移动对应的图片文件
            base_name = os.path.splitext(filename)[0]
            for ext in image_extensions:
                image_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(image_path):
                    try:
                        target_image_path = os.path.join(target_image_dir, base_name + ext)
                        shutil.move(image_path, target_image_path)
                        moved_images += 1
                        print(f"已移动图片文件: {image_path} -> {target_image_path}")
                        break
                    except Exception as e:
                        print(f"移动图片文件失败 {image_path}: {e}")

    # 打印统计信息
    print(f"\n处理完成!")
    print(f"检查文件数: {checked_files}")
    print(f"移动标签文件数: {moved_labels}")
    print(f"移动图片文件数: {moved_images}")


def move_images_by_box_size(label_dir, jpg_dir, output_dir, min_area=0, max_area=float('inf'), target_size=640):
    """
    移动目标框面积在指定范围内的图片和标签文件到指定目录

    参数:
    label_dir: 标签文件目录路径
    jpg_dir: 图像文件目录路径
    output_dir: 输出目录路径
    min_area: 最小面积阈值（像素平方）
    max_area: 最大面积阈值（像素平方）
    target_size: 目标图像尺寸

    返回:
    dict: 包含移动文件统计信息的字典
    """
    # 创建输出目录
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 统计信息
    stats = {
        'total_images_processed': 0,
        'images_moved': 0,
        'labels_moved': 0,
        'images_with_target_boxes': 0
    }

    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for i, label_filename in enumerate(os.listdir(label_dir)):
        if not label_filename.endswith('.txt'):
            continue

        stats['total_images_processed'] += 1
        if stats['total_images_processed'] % 100 == 0:
            print(f"处理进度: {stats['total_images_processed']}")

        label_file = os.path.join(label_dir, label_filename)
        base_name = os.path.splitext(label_filename)[0]

        # 查找对应的图片文件
        image_file = None
        for ext in image_extensions:
            potential_file = os.path.join(jpg_dir, base_name + ext)
            if os.path.exists(potential_file):
                image_file = potential_file
                break

        if image_file is None:
            continue

        try:
            # 读取图像
            img0 = cv2.imread(image_file)
            if img0 is None:
                continue

            img_height, img_width = img0.shape[:2]

            # 调整图像大小
            img_resized, scale, dw, dh = letterbox(img0, new_shape=target_size)

            # 处理标签文件，获取调整后的边界框
            adjusted_boxes = process_labels(label_file, scale, dw, dh, img_width, img_height)

            # 检查是否有目标框在指定面积范围内
            has_target_box = False
            for box in adjusted_boxes:
                _, x_center, y_center, width, height = box

                # 转换为像素尺寸
                width_px = width * target_size
                height_px = height * target_size
                area = width_px * height_px

                # 检查是否在指定面积范围内
                if min_area <= area <= max_area:
                    has_target_box = True
                    break

            # 如果有目标框在指定范围内，移动文件
            if has_target_box:
                stats['images_with_target_boxes'] += 1

                # 移动图片文件
                try:
                    image_dest = os.path.join(output_image_dir, os.path.basename(image_file))
                    shutil.move(image_file, image_dest)
                    stats['images_moved'] += 1
                    print(f"移动图片: {image_file} -> {image_dest}")
                except Exception as e:
                    print(f"移动图片失败 {image_file}: {e}")

                # 移动标签文件
                try:
                    label_dest = os.path.join(output_label_dir, label_filename)
                    shutil.move(label_file, label_dest)
                    stats['labels_moved'] += 1
                    print(f"移动标签: {label_file} -> {label_dest}")
                except Exception as e:
                    print(f"移动标签失败 {label_file}: {e}")

        except Exception as e:
            print(f"处理文件 {label_filename} 时出错: {e}")
            continue

    # 打印统计信息
    print(f"\n处理完成!")
    print(f"总共处理图片: {stats['total_images_processed']}")
    print(f"包含目标框的图片: {stats['images_with_target_boxes']}")
    print(f"移动的图片文件: {stats['images_moved']}")
    print(f"移动的标签文件: {stats['labels_moved']}")

    return stats


# 使用示例
if __name__ == "__main__":
    # 设置目录路径
    image_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\images\train"
    label_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\labels\train"

    target_image_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\images\train\moved_val_images"
    target_label_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train_Re_labeledjust_0\images\train\moved_val_labels"
    #
    # # 设置要移动的类别（例如，删除类别ID为1和2的图片和标签）
    classes_to_move = [0]
    #
    # # 调用函数
    move_images_by_class(image_dir, label_dir, target_image_dir, target_label_dir, classes_to_move)

    # clean_mismatched_files(image_dir, label_dir)

    # label_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train\labels\train"
    # jpg_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train\images\train"
    #
    # target_image_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train\images\train_32x32_128x128"
    # target_label_dir = r"F:\data_source\3d_print_guotou\dataset_2\real_train\labels\train_32x32_128x128"
    #
    # # 设置最小尺寸阈值和最大尺寸阈值（例如，小于5x5或大于128x128的对象）
    #
    #
    # # 调用函数
    # move_images_by_box_size(
    #     label_dir=label_dir,
    #     jpg_dir=jpg_dir,
    #     output_dir=target_image_dir,
    #     min_area=16384,
    #     max_area=float('inf')  # 32x32
    # )
