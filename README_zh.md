<h1 align="center">A2A-UAV</h1>
<h3 align="center">面向空对空无人机检测与识别的多模态数据集</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2611.21311"><img src="https://img.shields.io/badge/arXiv-2611.21311-b31b1b?style=flat-square" alt="arXiv 论文"></a>
  <a href="LICENSE-DATA.md"><img src="https://img.shields.io/badge/Data-CC_BY_4.0-0d9488?style=flat-square" alt="数据许可 CC BY 4.0"></a>
</p>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong></p>

A2A-UAV 通过无人机搭载的感知平台，在操场、公园和花园的昼夜场景中采集了 15 架、13 种商用型号目标无人机的时间关联 RGB、全景、热红外、射频时频图和毫米波雷达数据。

<p align="center"><img src="assets/a2a_uav_fig1.jpg" width="100%" alt="A2A-UAV 昼夜场景中的五种模态数据"></p>

## 下载

200 样本预览包 `A2A-UAV_preview.tar.gz` 可通过以下任一入口下载。

- [Google Drive](https://drive.google.com/file/d/1oCtjoF5vfH3MsPDNQEOQ84MvRywHSUso/view?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1_P6F3z18gCyuDagdedcM2Q?pwd=A2AD)（提取码 `A2AD`）

完整数据集将在论文录用后公开，共包含 271 个可独立解压的 `.tar.gz` 归档，总大小约 2.50 TB。将全部归档解压到同一目录，即可得到 `UAV_full_dataset_release/`。图像归档采用 `A2A-UAV_images_<split>_<part>.tar.gz` 命名，`A2A-UAV_metadata.tar.gz` 包含标注和索引。

预览包读取使用仓库 `dataset_tools/preview_manifests/` 中配套的 200 样本索引。将 `DATASET_ROOT` 设为解压后的预览目录或完整发布目录。

```sh
export DATASET_ROOT="/path/to/A2A-UAV_preview"
```

下载完整发布包后，校验归档与解压文件。

```sh
export ARCHIVE_DIR="/path/to/downloaded/archives"
export DATASET_ROOT="/path/to/UAV_full_dataset_release"
python dataset_tools/verify_archives.py "$ARCHIVE_DIR" checksums/archives.sha256
python dataset_tools/verify_release.py "$DATASET_ROOT" --quick
```

完整发布包的归档校验清单见 [checksums/archives.sha256](checksums/archives.sha256)，数据卡见 [DATA_CARD.md](DATA_CARD.md)。

## 数据结构

完整发布包采用以下布局。

```text
UAV_full_dataset_release/
├── annotations/
│   └── instances_{train,val,test}.json
├── images/
│   └── {train,val,test}/<sample_id>/
│       ├── metadata.json
│       ├── rgb.png
│       ├── panorama_lr.png
│       ├── thermal.png
│       ├── spectrum_tf.png
│       └── radar/frame_NNN.csv
├── manifests/
│   └── full_{train,val,test}.jsonl
├── metadata/
│   ├── classification_index.json
│   ├── classification_index_trainable.json
│   └── recording_assignments.csv
└── reports/
    ├── build_summary.md
    └── integrity_report.json
```

| | 训练集 | 验证集 | 测试集 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 采集记录 | 68 | 15 | 26 | 109 |
| 样本 | 61,767 | 16,468 | 27,059 | 105,294 |
| 标注框 | 122,762 | 32,934 | 53,843 | 209,539 |
| 含雷达数据的样本 | 61,737 | 16,468 | 27,058 | 105,263 |

样本编号由划分前缀 `tr`、`va` 或 `te` 与六位数字组成，例如 `tr_000505`。数据按规范化采集记录划分，同一条记录的全部样本属于同一划分。

## 各模态获取方式

样本目录 `images/<split>/<sample_id>/` 包含下表所列的数据。安装 Pillow 和 NumPy 后，在仓库根目录运行读取命令。示例 `tr_000505` 包含在预览包中。

```sh
python -m pip install Pillow numpy
```

| 模态 | 文件 | 格式 | 读取参数 |
| --- | --- | --- | --- |
| RGB | `rgb.png` | 1280 × 720，24 位 PNG | `rgb` |
| 全景 | `panorama_lr.png` | 5760 × 2880，24 位等距柱状投影 PNG | `panorama` |
| 热红外 | `thermal.png` | 384 × 288 PNG | `thermal` |
| 射频时频图 | `spectrum_tf.png` | 6000 × 3600，RGBA PNG | `rf` |
| 雷达 | `radar/frame_NNN.csv` | 每个雷达帧一个 CSV | `radar` |
| 元数据 | `metadata.json` | JSON | — |

命令行读取器输出图像路径、尺寸和像素模式，或样本的雷达帧。

```sh
python -m dataset_tools.read_modality "$DATASET_ROOT" tr_000505 rgb
python -m dataset_tools.read_modality "$DATASET_ROOT" tr_000505 panorama
python -m dataset_tools.read_modality "$DATASET_ROOT" tr_000505 thermal
python -m dataset_tools.read_modality "$DATASET_ROOT" tr_000505 rf
python -m dataset_tools.read_modality "$DATASET_ROOT" tr_000505 radar
```

在 Python 中可以加载一次样本，再读取其各模态。

```python
import numpy as np
from dataset_tools.read_sample import load_sample
from dataset_tools.read_modality import read_image, read_radar

DATASET_ROOT = "/path/to/A2A-UAV_preview"
sample_dir, metadata = load_sample(DATASET_ROOT, "tr_000505")
rgb = np.asarray(read_image(sample_dir, metadata, "rgb"))
panorama = np.asarray(read_image(sample_dir, metadata, "panorama"))
thermal = np.asarray(read_image(sample_dir, metadata, "thermal"))
spectrum = np.asarray(read_image(sample_dir, metadata, "rf"))
radar_frames = read_radar(sample_dir, metadata)
```

### RGB 图像

`rgb.png` 由 Skydroid C12 云台的 RGB 相机采集，视场为 100° × 52°。无人机标注框保存在 `metadata["annotations"]["rgb"]` 中。

### 全景图像

`panorama_lr.png` 是从 Insta360 X5 视频提取的等距柱状投影帧，覆盖 360° 视场。标注框保存在 `metadata["annotations"]["panorama"]` 中。全景检测基准采用 1280 × 1280 像素切片、1024 像素步长，每张全景图生成 18 个切片。

### 热红外图像

`thermal.png` 由 Skydroid C12 的热红外相机采集，光谱范围为 8–14 µm，视场为 24.8° × 18.7°。按 PNG 存储的像素模式读取，以保留位深和通道配置。像素值为显示值，未标定为温度。热红外图像没有边界框标注。

### 射频时频图

`spectrum_tf.png` 由 HAROGIC NXM-60 在中心频率 2.42 GHz、采样率 125 MSa/s 下采集的 0.1 s IQ 片段，经短时傅里叶变换生成。STFT 采用 1024 点 Hamming 窗、512 点重叠和 1024 点 FFT。横轴为 0 至 0.1 s，纵轴为 2.3575 至 2.4825 GHz，jet 色图表示图内的相对对数功率。

每个样本关联同一采集记录中时间最近的射频图像。同一记录内的多个样本可能共享同一张射频图。无人机标签为 `metadata["classification_label"]`。

### 毫米波雷达点云

雷达帧由 Texas Instruments IWR1843AOPEVM 采集，帧周期为 50 ms。同一记录中，样本参考时刻 ±120 ms 范围内的帧被唯一分配给时间最近的可关联样本。含雷达数据的样本在 `radar/` 下保存 1 至 6 帧。

CSV 每行表示一个点，字段顺序为 `x, y, z, intensity, velocity`。XYZ 使用雷达传感器坐标，单位为米；已核对的源记录将其标为 `ti_mmwave_0`。速度为有符号的径向速度，单位为 m/s。读取器保留这些数值。

`read_radar` 返回帧列表及有符号时间偏移 `dt_ns`，其定义为雷达帧时间减去样本参考时间。无雷达数据的样本返回 `[]`。只有表头的 CSV 返回一个 `points: []` 的零点帧。当前发布版提供雷达与样本的时间关联，逐点雷达–相机空间配准尚未提供。

可以用以下代码从完整发布清单筛选含雷达帧的样本。

```python
import json
from pathlib import Path

manifest = Path(DATASET_ROOT) / "manifests/full_train.jsonl"
with manifest.open(encoding="utf-8") as stream:
    radar_rows = [
        row for line in stream if line.strip()
        for row in [json.loads(line)] if row["radar_frames"]
    ]
```

### 样本元数据

`metadata.json` 记录样本编号、划分、采集记录编号、场景、光照条件、无人机标签、模态文件路径、RGB 与全景标注框和雷达帧列表。`scene_001`、`scene_002`、`scene_003` 分别表示操场、公园、花园；`condition_001`、`condition_002` 分别表示夜间、白天。发布版省略采集时间戳，样本编号不表示时间顺序。

`load_sample` 从逐样本元数据或样本索引中返回统一的元数据字典，可以用以下命令查看。

```sh
python dataset_tools/read_sample.py "$DATASET_ROOT" tr_000505
```

## 标注与标签

RGB 与全景标注采用两种格式。`metadata.json` 和 JSONL 清单使用像素单位的 `[x_min, y_min, x_max, y_max]`，`annotations/` 中的 COCO 文件使用 `[x_min, y_min, width, height]`。COCO 图像记录中的 `modality` 字段区分 `rgb` 与 `panorama`。每张图像至多包含一个标注框。

| 无人机型号 | 标签 |
| --- | --- |
| DJI Avata | `a1` |
| DJI Air 2S | `air2s` |
| DJI Air 3 | `air3` |
| DJI Air 3S | `air3s` |
| DJI Avata 2 | `avata2` |
| DJI Flip | `flip` |
| DJI Mavic 2 Pro | `mavic2pro` |
| DJI Mavic 4 Pro | `mavic4pro` |
| DJI Mini 3 Pro | `mini3pro` |
| DJI Mini 4 Pro，第 1 架 | `mini4pro_csq` |
| DJI Mini 4 Pro，第 2 架 | `mini4pro_xm` |
| DJI Mini 4 Pro，第 3 架 | `mini4pro_z` |
| DJI Mini 5 Pro | `mini5pro` |
| DJI Neo | `neo1` |
| DJI Neo 2 | `neo2` |

三个 Mini 4 Pro 标签用于区分同型号的不同无人机个体。

## 基准运行

### 检测

主要检测环境为 Python 3.10、PyTorch 2.1.0（CUDA 11.8）、TorchVision 0.16.0、MMDetection 3.3.0、MMCV 2.1.0 和 MMEngine 0.10.7。

先转换完整发布划分的 RGB 与全景标注，再生成全景切片和训练子集。

```sh
export A2AUAV_BENCHMARK_ROOT="$PWD/benchmark_inputs"
python detection/prepare_annotations.py --dataset-root "$DATASET_ROOT" \
  --rgb-out "$A2AUAV_BENCHMARK_ROOT/rgb/data/coco/annotations" \
  --panorama-out "$A2AUAV_BENCHMARK_ROOT/panorama/data/panorama_coco/annotations"
python detection/panorama/prepare_tiles.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama"
python detection/panorama/prepare_training_subsets.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama"
```

将 [torchvision ResNet-50 ImageNet 预训练权重](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) 放入两个预训练目录。

```sh
mkdir -p "$A2AUAV_BENCHMARK_ROOT/rgb/pretrained" "$A2AUAV_BENCHMARK_ROOT/panorama/pretrained"
curl -L --fail https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
  -o "$A2AUAV_BENCHMARK_ROOT/rgb/pretrained/resnet50-11ad3fa6.pth"
cp "$A2AUAV_BENCHMARK_ROOT/rgb/pretrained/resnet50-11ad3fa6.pth" \
  "$A2AUAV_BENCHMARK_ROOT/panorama/pretrained/resnet50-11ad3fa6.pth"
```

将 `MMDET_ROOT` 设为 MMDetection 3.3.0 源码目录。下例分别用 3 张 GPU 训练 RGB Faster R-CNN，用 2 张 GPU 训练全景 Faster R-CNN。

```sh
export MMDET_ROOT="/path/to/mmdetection"
export PYTHONPATH="$PWD/detection/panorama${PYTHONPATH:+:$PYTHONPATH}"
python -m torch.distributed.run --nproc_per_node=3 "$MMDET_ROOT/tools/train.py" \
  detection/rgb/configs/faster_rcnn.py --launcher pytorch \
  --work-dir "$A2AUAV_BENCHMARK_ROOT/rgb/runs/faster_rcnn"
python -m torch.distributed.run --nproc_per_node=2 "$MMDET_ROOT/tools/train.py" \
  detection/panorama/configs/faster_rcnn.py --launcher pytorch \
  --work-dir "$A2AUAV_BENCHMARK_ROOT/panorama/runs/faster_rcnn"
```

主环境中的其余配置名为 `dynamic_rcnn`、`cascade_rcnn`、`sparse_rcnn`、`dino` 和 `ddq`。现有配置定义了训练长度、负切片抽样、验证监测与停止准则。

FCOS-KL 使用 [KLDet](https://github.com/TinyOD/mmdet-kldet) 的 `79caa41f16c84be8b9a6567002f51c2cad976f8d` 提交，环境为 Python 3.8.20、PyTorch 1.10.2+cu113、MMCV 1.4.0 和 MMDetection 2.13.0。其 RGB 与全景配置采用[独立环境及适配说明](detection/legacy_fcos_kl/README.md)。

使用验证集选定的检查点评估发布版测试集的 27,059 个样本。RGB 和全景评估各覆盖每个测试样本的一张图像。

```sh
python detection/rgb/evaluate.py --benchmark-root "$A2AUAV_BENCHMARK_ROOT" \
  --config detection/rgb/configs/faster_rcnn.py --checkpoint "$RGB_CHECKPOINT" \
  --work-dir "$RGB_EVAL_DIR" --split test
python detection/panorama/evaluate.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama" \
  --run-name faster_rcnn --checkpoint "$PANO_CHECKPOINT" --split test_full
```

全景评估使用 0.55 的 NMS IoU 阈值。对保存的切片预测，可以用以下命令合并评估。

```sh
python detection/panorama/merge_predictions.py \
  --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama" \
  --run-name faster_rcnn --split test_full \
  --results-jsonl "$PANO_PREDICTIONS_JSONL"
```

### 射频识别

以下命令在发布版划分上训练并评估射频识别模型。

从发布版分类索引和校验清单生成 RF 索引。内容相同的时频图共享同一个抽样组。

```sh
export RF_INDEX="$PWD/rf_index.json"
export RF_CACHE="$PWD/rf_cache"
export RF_RUN_DIR="$PWD/runs/rf/efficientnet_v2_s"
python rf_identification/prepare_index.py --dataset-root "$DATASET_ROOT" --output "$RF_INDEX"
python rf_identification/prepare_cache.py --dataset-root "$DATASET_ROOT" \
  --index "$RF_INDEX" --output-dir "$RF_CACHE"
python -m torch.distributed.run --nproc_per_node=2 rf_identification/train_and_evaluate.py \
  --index "$RF_INDEX" --cache-root "$RF_CACHE" --output-dir "$RF_RUN_DIR" \
  --method efficientnet_v2_s --seed 42
```

缓存为采用双线性插值缩放的 1333 × 800 RGB PNG。Recording-PK 抽样每张 GPU 使用 6 个样本，2 张 GPU 的有效批大小为 12。检查点按验证集 macro-F1 选择。可用模型为 `resnet18`、`resnet50`、`efficientnet_v2_s`、`convnext_tiny`、`swin_t` 和 `mobilenet_v3_large`。

使用 `--eval-only` 时，将所选检查点放在运行目录的 `best.pt`。

## 引用与许可

```bibtex
@misc{liang2026a2auavmultimodalairtoairdataset,
  title={A2A-UAV: A Multimodal Air-to-Air Dataset for UAV Detection and Identification},
  author={Hongtao Liang and Yiyao Wan and Kezhi Zhu and Jiayi Zhang and Zequn Jiang and Ziyu Zhou and Jiahuan Ji and Fuhui Zhou and Qihui Wu},
  year={2026},
  eprint={2611.21311},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.21311},
}
```

机器可读引用元数据见 [CITATION.cff](CITATION.cff)。代码采用 [Apache-2.0](LICENSE)，数据文件采用 [CC BY 4.0](LICENSE-DATA.md)。
