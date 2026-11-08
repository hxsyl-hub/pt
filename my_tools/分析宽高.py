import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import glob

plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
def analyze_yolo_boxes(labels_dir, output_dir=None, num_clusters=9):
    """
    分析YOLO标签文件中的边界框宽高比例
    :param labels_dir: YOLO标签文件目录
    :param output_dir: 输出目录，如果为None则不保存结果
    :param num_clusters: K-means聚类的簇数，默认为9（YOLOv5默认anchor数）
    """
    # 获取所有标签文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    if not label_files:
        print(f"警告: 在目录 {labels_dir} 中没有找到标签文件")
        return

    print(f"找到 {len(label_files)} 个标签文件")

    # 读取所有边界框
    all_boxes = []
    widths = []
    heights = []
    ratios = []

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:
                        # 忽略类别ID，只取宽高
                        _, _, _, w, h = map(float, data)
                        widths.append(w)
                        heights.append(h)
                        ratios.append(w / h if h > 0 else 0)
                        all_boxes.append([w, h])
        except Exception as e:
            print(f"读取文件 {label_file} 时出错: {e}")

    if not all_boxes:
        print("警告: 没有找到有效的边界框")
        return

    # 转换为numpy数组
    all_boxes = np.array(all_boxes)
    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(ratios)

    print(f"找到 {len(all_boxes)} 个边界框")
    print(f"宽度范围: {widths.min():.4f} - {widths.max():.4f}")
    print(f"高度范围: {heights.min():.4f} - {heights.max():.4f}")
    print(f"宽高比范围: {ratios.min():.4f} - {ratios.max():.4f}")
    print(f"平均宽高比: {ratios.mean():.4f}")

    # 使用K-means聚类分析anchor尺寸
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_boxes)
    anchors = kmeans.cluster_centers_

    # 按面积排序anchor
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = anchors[sorted_indices]

    print(f"\n推荐的 {num_clusters} 个anchor尺寸 (宽, 高):")
    for i, anchor in enumerate(sorted_anchors):
        print(f"Anchor {i + 1}: ({anchor[0]:.6f}, {anchor[1]:.6f}), 宽高比: {anchor[0] / anchor[1]:.2f}")

    # 绘制宽高比分布直方图
    plt.figure(figsize=(15, 10))

    # 宽高比分布
    plt.subplot(2, 2, 1)
    plt.hist(ratios, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('宽高比 (宽度/高度)')
    plt.ylabel('频率')
    plt.title('边界框宽高比分布')
    plt.grid(True, alpha=0.3)

    # 宽度分布
    plt.subplot(2, 2, 2)
    plt.hist(widths, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('归一化宽度')
    plt.ylabel('频率')
    plt.title('边界框宽度分布')
    plt.grid(True, alpha=0.3)

    # 高度分布
    plt.subplot(2, 2, 3)
    plt.hist(heights, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('归一化高度')
    plt.ylabel('频率')
    plt.title('边界框高度分布')
    plt.grid(True, alpha=0.3)

    # 宽高散点图
    plt.subplot(2, 2, 4)
    plt.scatter(widths, heights, alpha=0.5, s=10, color='purple')
    plt.xlabel('归一化宽度')
    plt.ylabel('归一化高度')
    plt.title('边界框宽高散点图')

    # 标记推荐的anchor
    for anchor in sorted_anchors:
        plt.scatter(anchor[0], anchor[1], s=100, color='red', marker='x')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存图表
        plt.savefig(os.path.join(output_dir, 'anchor_analysis.png'), dpi=300, bbox_inches='tight')

        # 保存anchor信息
        with open(os.path.join(output_dir, 'anchors.txt'), 'w') as f:
            f.write(f"数据集分析结果:\n")
            f.write(f"标签文件数量: {len(label_files)}\n")
            f.write(f"边界框数量: {len(all_boxes)}\n")
            f.write(f"宽度范围: {widths.min():.6f} - {widths.max():.6f}\n")
            f.write(f"高度范围: {heights.min():.6f} - {heights.max():.6f}\n")
            f.write(f"宽高比范围: {ratios.min():.6f} - {ratios.max():.6f}\n")
            f.write(f"平均宽高比: {ratios.mean():.6f}\n\n")

            f.write(f"推荐的 {num_clusters} 个anchor尺寸:\n")
            for i, anchor in enumerate(sorted_anchors):
                f.write(f"Anchor {i + 1}: {anchor[0]:.6f}, {anchor[1]:.6f}, 宽高比: {anchor[0] / anchor[1]:.2f}\n")

        print(f"\n分析结果已保存到 {output_dir}")

    # 显示图表
    plt.show()

    return sorted_anchors


def main():
    parser = argparse.ArgumentParser(description="分析YOLO标签文件中的边界框宽高比例")
    parser.add_argument("--labels_dir", type=str, required=True, help="YOLO标签文件目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，如果不指定则不保存结果")
    parser.add_argument("--num_clusters", type=int, default=9, help="K-means聚类的簇数，默认为9")

    args = parser.parse_args()

    print(f"开始分析YOLO标签文件...")
    print(f"标签目录: {args.labels_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"聚类数量: {args.num_clusters}")

    analyze_yolo_boxes(args.labels_dir, args.output_dir, args.num_clusters)

    print("分析完成!")


if __name__ == "__main__":
    # 示例使用
    labels_dir = r"F:\data_source\3d_print_guotou\dataset_2\val\labels"
    output_dir = r"F:\data_source\3d_print_guotou\dataset_2"

    # 如果通过命令行运行，则使用参数解析
    # 如果直接运行脚本，则使用上面的默认值
    if True == 1:
        # 使用默认值
        analyze_yolo_boxes(labels_dir, output_dir, 9)
    else:
        main()