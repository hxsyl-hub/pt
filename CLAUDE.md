# 项目级说明：3D 打印喷嘴缺陷检测（YOLOv5）

## 项目定位
3D 打印喷嘴 ng（裹头缺陷）/ ok（合格）二分类缺陷检测工程。数据集为 YOLO 格式，由中文工具链治理。

当前数据状态（2026-11-17）：`label_dataset/dataset_2` 共 2564 张（train 1743 / val 534 / test 287）。已完成 SHA256 精确去重、pHash 相似去重与人工审核；训练集保留全部 ng，OK 降采样到与 ng 同量；框尺寸筛选（<32×32、≥128×128）只作用于 OK，避免削减缺陷类大目标。

## 目录职责
- `configs/`：训练与数据配置。`3dprint_nozzle.yaml`（双类）、`3dprint_nozzle_single.yaml`（单类 ng）、`hyp_3dprint.yaml`（超参数）、`yolov5s_3dprint.yaml`（自定义 anchor 的模型配置）。
- `scripts/`：本项目自研流程脚本。`filter_dataset.py`（类别/尺寸筛选与回滚）、`train_baseline.sh`（训练入口）、`package_upload.py`（生成上传压缩包）。
- `analysis_tool/`：数据集分布与质量分析（类别、框大小、宽高比、KMeans anchor、亮度与过曝核对），输出在 `analysis_tool/output/`。
- `dedup_tool/`：标签格式校验转换、SHA256 + pHash 去重与人工审核流程，日志与暂存在 `dedup_tool/_work/`。
- `my_tools/`：参考工具集（标注审计、数据增强、漏检误检统计），未直接参与本数据集治理。
- `label_dataset/dataset_2/`：当前数据集。
- `artifacts/`：流程产物。`筛选_*/` 为筛选备份与清单，`dist/` 为上传压缩包。
- `runs_from_cloud/`：历史训练产物（opt.yaml、hyp.yaml、曲线、权重），只读参考。
- `yolov5-7.0/`：Ultralytics 官方 YOLOv5 v7.0，原样保留。入口 `train.py` / `val.py` / `detect.py` / `export.py`。
- `doc/`：需求文档，不纳入 git。

## 文档流程对照
| 文档步骤 | 对应操作 | 脚本与产物 |
|---|---|---|
| 第 9–11 页 数据分布与质量 | 类别、框尺寸、宽高比、anchor、亮度过曝、标注核对 | `analysis_tool/analyze_dataset.py` → `analysis_tool/output/` |
| 第 11 页 去重复 | SHA256 精确去重 + pHash 相似去重 + 人工审核 | `dedup_tool/去重_人工挑选.py` → `dedup_tool/_work/` |
| 第 12 页 数据筛选 | OK 降采样、移出大小目标 | `scripts/filter_dataset.py` → `artifacts/筛选_*/` |
| 第 12 页 框分析 | KMeans 聚类 anchor | 分析脚本 → `anchors.txt`、`configs/yolov5s_3dprint.yaml` |
| 第 13–15 页 配置与训练 | 数据集 yaml、超参数、50 轮基线 | `configs/`、`scripts/train_baseline.sh` |
| 第 16–17 页 两阶段训练 | 单类训练 → 双类接力 | `scripts/train_baseline.sh` 的 `DATA=` / `WEIGHTS=` 变量 |
| 第 18–20 页 预测与漏检误检 | best.pt 预测验证集、低分/漏检/误检清单 | `scripts/predict_val.sh`、`scripts/analyze_predictions.py` |
| 第 23–25 页 标签修正 | 用预测框替换偏移的 GT、原标签备份 | `scripts/analyze_predictions.py --apply` |
| 第 29–31 页 ng 数据增强 | 缺陷类缩放增强（小目标/全目标）+ 颜色扰动 | `scripts/augment_ng.py` → `artifacts/增强_ng_*/` |

## 常用命令
在项目根执行，Python 环境 `D:\miniconda_envs\gt`。各脚本均带 `argparse`，可用 `--help` 查看参数。

| 目的 | 命令 |
|---|---|
| 数据分布与质量分析 | `python analysis_tool/analyze_dataset.py [--splits train val test]` |
| 标签格式校验（默认只查不改） | `python dedup_tool/v5_格式校验与转换.py [--fix] [--yaml configs/3dprint_nozzle.yaml]` |
| 去重与人工审核 | `python dedup_tool/去重_人工挑选.py --scan / --export / --review / --apply <csv> / --auto / --restore <csv>` |
| 数据筛选预览 | `python scripts/filter_dataset.py` |
| 数据筛选执行 | `python scripts/filter_dataset.py --execute` |
| 筛选回滚 | `python scripts/filter_dataset.py --restore artifacts/筛选_xxx/moved_manifest.csv` |
| 训练 | `bash scripts/train_baseline.sh` |
| 验证集预测 | `bash scripts/predict_val.sh`（可用 `SOURCE=` / `WEIGHTS=` / `CONF=` 覆盖） |
| 预测结果分析 | `python scripts/analyze_predictions.py [--iou 0.5] [--low-score 0.5] [--apply]` |
| ng 类缩放增强预览 | `python scripts/augment_ng.py` |
| ng 类缩放增强执行 | `python scripts/augment_ng.py --execute` |
| 撤销增强 | `python scripts/augment_ng.py --remove artifacts/增强_ng_xxx/added_manifest.csv` |
| 生成上传包 | `python scripts/package_upload.py [--only code|dataset]` |
| 标注一致性审计（参考） | `python my_tools/check_dataset_audit.py [--root <dir>] [--apply]` |
| 漏检误检统计（参考） | `python my_tools/找漏检和误检.py --pred_dir <dir> --gt_dir <dir> ...` |

## 训练实验记录
数据固定为当前 `label_dataset/dataset_2`（train 1743 / val 534），超参文件在 `configs/`。

| 实验 | 配置要点 | 结果（val 534） |
|---|---|---|
| `base_v5s_50e`（旧） | batch 32、无增强、lr0 1e-2，旧版 3783 数据 | P 0.980 / R 0.990 / mAP50 0.991 / mAP50-95 0.882，**验证集不同，不可直接比较** |
| `base_v5s_50e_v2` | `hyp_3dprint.yaml`：Adam lr0 1e-4、batch 8、mosaic/mixup/旋转/剪切/上下翻转/multi-scale 全开 | P 0.991 / R 0.989 / mAP50 0.990 / **mAP50-95 0.814**（ep47 最佳，末段回落） |
| `base_v11bb_v5anchor_50e` | `yolov5-7.0/models/yolov11backbone_v5s.yaml`：YOLO11 backbone（C3k2 / C2PSA）+ YOLOv5 anchor head，其余同 v2 | P 0.983 / R 0.980 / mAP50 0.985 / mAP50-95 0.652（最佳 0.683）。第 0 轮 mAP 仅 0.0009，预训练权重未迁移，等于从零训练；lr 1e-4 下 50 轮未收敛 |
| `v11bb_stage2` | 以 `base_v11bb_v5anchor_50e4/weights/best.pt` 接力，`hyp_3dprint_v3.yaml`（lr0 1e-3），50 轮 | P 0.992 / R 0.989 / mAP50 0.992 / **mAP50-95 0.758**（最佳 0.777，ep30 后震荡持平）。比 stage1 +10.6，仍低于 v2 的 0.814 |
| `v11bb_siou` | `hyp_3dprint_siou.yaml`：CIoU 换 SIoU（`s_iou: true`），以 `v11bb_stage2/weights/best.pt` 接力，其余同 v3 | P 0.994 / R 0.991 / mAP50 0.992 / mAP50-95 0.719（**最佳 0.793@ep19**）。best 比 stage2 +1.4，但 ep19 后回落；分类别看 ng 0.718、ok 0.869，缺陷类定位是短板 |
| `base_v5s_50e_v3` | `hyp_3dprint_v3.yaml`：以 v2 `best.pt` 接力，仅 lr0 提到 1e-3 | 待跑 |
| `base_v5s_50e_v4` | `hyp_3dprint_v4.yaml`：v3 基础再降增强（mixup 0、shear 0、degrees 0、scale 0.3），配 `MULTI_SCALE=0 IMAGE_WEIGHTS=0` | 备用，视 v3 结果决定 |

v2 相对 v1 的 mAP50-95 下降约 7 个点，训练侧可归因于 lr0 过小（收敛慢）与增强偏强；数据侧则是验证集换成了不同样本。

v3 命令：
```bash
HYP=configs/hyp_3dprint_v3.yaml \
WEIGHTS=runs/base_v5s_50e_v2/weights/best.pt \
NAME=base_v5s_50e_v3 bash scripts/train_baseline.sh
```

v4 命令：
```bash
HYP=configs/hyp_3dprint_v4.yaml MULTI_SCALE=0 IMAGE_WEIGHTS=0 \
WEIGHTS=runs/base_v5s_50e_v3/weights/best.pt NAME=base_v5s_50e_v4 bash scripts/train_baseline.sh
```

YOLO11 backbone + YOLOv5 anchor head 实验：
- 模型配置：`yolov5-7.0/models/yolov11backbone_v5s.yaml`
- 模块移植：`yolov5-7.0/models/yolo11_blocks.py`（C2f / C3k / C3k2 / PSA / C2PSA，复用 v5 的 Conv 与 C3）
- 注册位置：`yolov5-7.0/models/yolo.py` 的 `parse_model` 模块集合
```bash
CFG=yolov5-7.0/models/yolov11backbone_v5s.yaml \
NAME=base_v11bb_v5anchor_50e bash scripts/train_baseline.sh
```

## 陷阱与约定
- **路径解析基准**：`configs/3dprint_nozzle.yaml` 的 `path: ../label_dataset/dataset_2` 相对 `yolov5-7.0/` 仓库根解析（`utils/general.py` 的 `check_dataset` 用 `ROOT / path`），与 yaml 文件位置、当前工作目录都无关。
- **两份 yaml 同步**：`configs/3dprint_nozzle.yaml` 与 `yolov5-7.0/data/3dprint_nozzle.yaml` 内容相同，改一处要同步另一处。
- **筛选脚本可重复执行**：每次都按当前数据重新降采样，在已筛选的数据上再 `--execute` 会继续削减 OK；执行前先用预览模式确认。
- **筛选备份可回滚**：`artifacts/筛选_*/moved_manifest.csv` 记录每一张的移动原因与原始路径，`--restore` 可按清单恢复。
- **依赖缺口**：`my_tools/重复过滤.py` 依赖 `imagehash`、`skimage`，本机未安装，已被 `dedup_tool/去重_人工挑选.py` 取代。`analysis_tool` 与 `my_tools/分析宽高.py` 需要 `scikit-learn`。
- **中文路径与字体**：读图统一用 `cv2.imdecode(np.fromfile(...))`，绘图设置 `plt.rcParams['font.family'] = 'SimHei'`。非 Windows 或无 SimHei 环境需调整。
- **审计报告未应用**：`label_dataset/dataset_2/_audit_report.csv` 有 943 条类别疑问（主要为 `nozzle_no_extrusion` 应为 NG），尚未执行 `--apply`。
- **Python 3.12 / 新依赖兼容补丁（2026-11-17）**：目标运行环境为 Python 3.12 + numpy 2.x + Pillow 10，官方 v7.0 有 4 处不兼容，本仓库 `yolov5-7.0/` 已全部修改：
  1. `train.py:300` multi-scale：`random.randrange(imgsz * 0.5, imgsz * 1.5 + gs)` → 两处加 `int()`。否则第 1 个 batch 报 `TypeError: 'float' object cannot be interpreted as an integer`。
  2. `utils/metrics.py:118`：`np.trapz` → `np.trapezoid`（numpy 2.x 移除 `trapz`）。否则验证算 mAP 时报 `AttributeError: module 'numpy' has no attribute 'trapz'`。
  3. `utils/plots.py:91`：`self.font.getsize(label)` → `self.font.getbbox(label)`（Pillow 10 移除 `getsize`）。
  4. `utils/plots.py:170`：同上，PIL-only 的 `text(anchor='bottom')` 分支。
- **train.py 默认值已按需求文档第 14–15 页修改**：`--multi-scale`、`--image-weights` 默认 True（用 `--no-multi-scale` / `--no-image-weights` 关闭），`--optimizer` 默认 Adam，`--workers` 默认 4，`--label-smoothing` 默认 0.2。`scripts/train_baseline.sh` 显式传这些参数，两处保持一致。
- **SIoU 开关**：`utils/metrics.py` 的 `bbox_iou()` 新增 `SIoU` 参数，`utils/loss.py` 的 `ComputeLoss` 读取 hyp 的 `s_iou` 字段（默认 False，仍用 CIoU）。在超参文件里写 `s_iou: true` 即切换到 SIoU，无需改代码。

## 边界
- `yolov5-7.0/` 的兼容补丁、`models/yolo11_blocks.py`、`models/yolov11backbone_v5s.yaml`、`models/yolo.py` 的模块注册、`train.py` 默认值、`utils/metrics.py` 与 `utils/loss.py` 的 SIoU 支持属于已授权的修改；其他官方文件不要动。
- `runs_from_cloud/` 是训练产物，只读参考，不要当源码修改。
