# A2A-UAV data card

105,294 samples; 15 labels from 13 UAV models; 109 recording groups.
210,588 RGB/panorama images; 209,539 boxes.

| Split | Samples | Recording groups | Boxes |
| --- | ---: | ---: | ---: |
| Train | 61,767 | 68 | 122,762 |
| Validation | 16,468 | 15 | 32,934 |
| Test | 27,059 | 26 | 53,843 |

| Sample file | Format |
| --- | --- |
| `rgb.png` | 1280 × 720 |
| `panorama_lr.png` | 5760 × 2880 |
| `thermal.png` | 384 × 288 |
| `spectrum_tf.png` | 6000 × 3600 |
| `metadata.json` | Labels, boxes, grouping and modality paths |
| `radar/frame_NNN.csv` | x, y, z, intensity, velocity; optional |

Boxes use native pixels: metadata/JSONL uses `[x_min,y_min,x_max,y_max]`; COCO uses `[x_min,y_min,width,height]`. Each RGB/panorama image has at most one box. Samples with both views unannotated are excluded; 1,049 single-sided unannotated images remain with their paired annotated view.

Radar: 505,673 frames in 105,263 samples. The 31 samples without radar are split 30/0/1. Header-only CSVs are valid zero-point frames (2,467). Assignment is exclusive within ±120 ms. `csv` points to the nearest frame; `point_count` refers to that frame, `total_point_count` to all assigned frames. `dt_ns` is signed frame time minus reference time. Frame paths are sample-relative in metadata and root-relative in manifests.

`scene_001/002/003` means playground/park/garden; `condition_001/002` means night/day. Timestamps are null; IDs have gaps and do not encode time order.

Thermal and RF PNGs are processed display products, without calibrated temperature/power values. Radar returns have no per-point UAV identity. Reused RF windows within a recording are not independent RF acquisitions.

Filtering rules, statistics and audit evidence: `reports/integrity_report.json`.
