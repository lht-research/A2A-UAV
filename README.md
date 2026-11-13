# A2A-UAV

[中文](README_zh.md) · [Dataset card](docs/DATASET_CARD.md)

A multimodal air-to-air dataset for UAV detection and identification. The release contains RGB, panoramic, thermal infrared and RF time–frequency images, radar frames where available, and sample metadata.

![A2A-UAV examples](assets/a2a_uav_fig1.jpg)

## Data availability

Download the preview package (`A2A-UAV_preview.tar.gz`) from either mirror.

- [Google Drive](https://drive.google.com/file/d/1oCtjoF5vfH3MsPDNQEOQ84MvRywHSUso/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1c7TXhvHKnL0h4Wx4UmDwog?pwd=A2AD) (extraction code `A2AD`)

The full dataset will be made publicly available after the paper is accepted.

The full release comprises 271 independently extractable archives, approximately 2.50 TB in total. Once available, extract the archives into the same empty directory to obtain `UAV_full_dataset_release/`.

| | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Samples | 61,767 | 16,468 | 27,059 | 105,294 |
| RGB and panoramic image records | 123,534 | 32,936 | 54,118 | 210,588 |
| Bounding boxes | 122,762 | 32,934 | 53,843 | 209,539 |

The data cover 15 UAV labels. Capture timestamps are omitted. Recording identifiers group related samples, while sample IDs do not encode chronology. Radar frame offsets are relative to the associated sample reference time.

## Dataset

Set `DATASET_ROOT` to the extracted `UAV_full_dataset_release` directory.

```sh
python dataset_tools/read_sample.py "$DATASET_ROOT" tr_000001
python dataset_tools/verify_release.py "$DATASET_ROOT" --quick
python dataset_tools/verify_archives.py "$ARCHIVE_DIR" "$ARCHIVE_CHECKSUM_FILE"
```

The sample reader returns paths to four PNGs, optional radar CSVs, and the sample's annotations.

## Detection

The recorded environment used Python 3.10, PyTorch 2.1.0 with CUDA 11.8, TorchVision 0.16.0, MMDetection 3.3.0, MMCV 2.1.0, and MMEngine 0.10.7.

```sh
export A2AUAV_BENCHMARK_ROOT="$PWD/benchmark_inputs"
python detection/prepare_annotations.py --dataset-root "$DATASET_ROOT" \
  --rgb-out "$A2AUAV_BENCHMARK_ROOT/rgb/data/coco/annotations" \
  --panorama-out "$A2AUAV_BENCHMARK_ROOT/panorama/data/panorama_coco/annotations"
python detection/panorama/prepare_tiles.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama"
python detection/panorama/prepare_training_subsets.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama"
```

Use MMDetection's `tools/train.py` with the selected RGB or panorama configuration. Add `detection/panorama` to `PYTHONPATH` for panorama training. Place the initialization weights under the corresponding `pretrained/` path in the benchmark directory.

```sh
python detection/rgb/evaluate.py --benchmark-root "$A2AUAV_BENCHMARK_ROOT" \
  --config detection/rgb/configs/faster_rcnn.py --checkpoint "$RGB_CHECKPOINT" \
  --work-dir "$RGB_EVAL_DIR" --split test
python detection/panorama/evaluate.py --experiment-root "$A2AUAV_BENCHMARK_ROOT/panorama" \
  --run-name faster_rcnn --checkpoint "$PANO_CHECKPOINT" --split test_full
```

Panorama settings are 1280-pixel tiles, 1024-pixel stride, and 0.55 NMS IoU. Use `merge_predictions.py` to score saved tile predictions. For FCOS-KL, use the matching KLDet checkout, legacy MMDetection environment, and configuration in `detection/legacy_fcos_kl/`.

## RF identification

Build the RF index from the downloaded classification index and checksum inventory. Identical spectrograms share a sampling group.

```sh
python rf_identification/prepare_index.py --dataset-root "$DATASET_ROOT" --output "$RF_INDEX"
python rf_identification/prepare_cache.py --dataset-root "$DATASET_ROOT" \
  --index "$RF_INDEX" --output-dir "$RF_CACHE"
python -m torch.distributed.run --nproc_per_node=2 rf_identification/train_and_evaluate.py \
  --cohort downloaded --index "$RF_INDEX" --cache-root "$RF_CACHE" --output-dir "$RF_RUN_DIR" \
  --method efficientnet_v2_s --seed 42
```

The cache uses 1333 × 800 RGB PNGs with bilinear resizing. For evaluation with `--eval-only`, place the selected checkpoint at `best.pt` in the run directory. The default `--cohort paper` uses the archived experimental index. See `recorded_settings.json` for the recorded model settings.

The paper's benchmark metrics refer to archived experimental inputs and checkpoints. Using the downloadable dataset's split files evaluates the methods on that release.

## Citation and license

See [CITATION.cff](CITATION.cff) for citation metadata and [LICENSE.md](LICENSE.md) for the dataset license.
