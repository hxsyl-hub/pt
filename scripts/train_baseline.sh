#!/usr/bin/env bash
# 3D 打印喷嘴缺陷检测训练启动脚本
#
# 常用用法（在项目根执行，脚本内部会切到项目根）:
#   bash scripts/train_baseline.sh                          # 双类基线, 50 轮
#   EPOCHS=100 BATCH=16 bash scripts/train_baseline.sh      # 调轮数和 batch
#   HYP=configs/hyp_3dprint_v3.yaml WEIGHTS=runs/base_v5s_50e_v2/weights/best.pt \
#       NAME=base_v5s_50e_v3 bash scripts/train_baseline.sh # 接力训练
#   MULTI_SCALE=0 IMAGE_WEIGHTS=0 bash scripts/train_baseline.sh   # 关闭对应增强
set -euo pipefail
cd "$(dirname "$0")/.."

YOLO_DIR="${YOLO_DIR:-yolov5-7.0}"
DATA="${DATA:-configs/3dprint_nozzle.yaml}"
HYP="${HYP:-configs/hyp_3dprint.yaml}"
WEIGHTS="${WEIGHTS:-yolov5s.pt}"   # 不存在时 train.py 会自动下载
CFG="${CFG:-}"                     # 留空用 yolov5s 结构；有自定义 backbone 时填 yaml 路径
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"                # 需求文档默认 8；显存足够可调 16/32
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-4}"
MULTI_SCALE="${MULTI_SCALE:-1}"
IMAGE_WEIGHTS="${IMAGE_WEIGHTS:-1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.2}"
NAME="${NAME:-run_$(date +%m%d_%H%M)}"

for f in "$DATA" "$HYP"; do
  [ -f "$f" ] || { echo "缺少配置文件: $f"; exit 1; }
done

# 转成绝对路径，避免 cd 到 yolov5 目录后相对路径失效
abs() { if [ -e "$1" ]; then printf '%s/%s' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; else printf '%s' "$1"; fi; }
DATA="$(abs "$DATA")"
HYP="$(abs "$HYP")"
WEIGHTS="$(abs "$WEIGHTS")"
if [ -n "$CFG" ]; then
  CFG="$(abs "$CFG")"
fi

if [ ! -d "$YOLO_DIR" ]; then
  echo "yolov5 仓库不存在，开始克隆 v7.0 ..."
  git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5 "$YOLO_DIR"
fi

ARGS=(
  --weights "$WEIGHTS"
  --data "$DATA"
  --hyp "$HYP"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --imgsz "$IMGSZ"
  --optimizer Adam
  --workers "$WORKERS"
  --label-smoothing "$LABEL_SMOOTHING"
  --project ../runs
  --name "$NAME"
)
if [ "$MULTI_SCALE" = "1" ]; then
  ARGS+=(--multi-scale)
else
  ARGS+=(--no-multi-scale)
fi
if [ "$IMAGE_WEIGHTS" = "1" ]; then
  ARGS+=(--image-weights)
else
  ARGS+=(--no-image-weights)
fi
if [ -n "$CFG" ]; then
  ARGS+=(--cfg "$CFG")
fi

echo "训练配置: data=$DATA hyp=$HYP weights=$WEIGHTS epochs=$EPOCHS batch=$BATCH"
echo "          multi_scale=$MULTI_SCALE image_weights=$IMAGE_WEIGHTS label_smoothing=$LABEL_SMOOTHING"
echo "输出目录: runs/$NAME"
cd "$YOLO_DIR"
python train.py "${ARGS[@]}"
