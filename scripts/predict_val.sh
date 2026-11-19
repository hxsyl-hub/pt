#!/usr/bin/env bash
# 用训练好的权重对验证集/训练集做预测，输出带置信度的 YOLO 标签和可视化结果
#
# 用法（项目根执行）:
#   bash scripts/predict_val.sh
#   SOURCE=label_dataset/dataset_2/train/images NAME=train_pred bash scripts/predict_val.sh
#   WEIGHTS=runs/xxx/weights/best.pt CONF=0.5 bash scripts/predict_val.sh
set -euo pipefail
cd "$(dirname "$0")/.."

YOLO_DIR="${YOLO_DIR:-yolov5-7.0}"
WEIGHTS="${WEIGHTS:-runs/base_v5s_50e_v2/weights/best.pt}"
SOURCE="${SOURCE:-label_dataset/dataset_2/val/images}"
CONF="${CONF:-0.25}"
NAME="${NAME:-val_pred}"
IMGSZ="${IMGSZ:-640}"

[ -f "$WEIGHTS" ] || { echo "权重不存在: $WEIGHTS"; exit 1; }
[ -d "$SOURCE" ] || { echo "图片目录不存在: $SOURCE"; exit 1; }

echo "权重: $WEIGHTS"
echo "图片: $SOURCE"
echo "置信度阈值: $CONF"
echo "输出: runs/detect/$NAME"

python "$YOLO_DIR/detect.py" \
  --weights "$WEIGHTS" \
  --source "$SOURCE" \
  --imgsz "$IMGSZ" \
  --conf-thres "$CONF" \
  --save-txt --save-conf \
  --project runs/detect --name "$NAME" --exist-ok

echo "预测标签: runs/detect/$NAME/labels"
echo "下一步: python scripts/analyze_predictions.py"
