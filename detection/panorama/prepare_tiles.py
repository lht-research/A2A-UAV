from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--min-box-keep-ratio", type=float, default=0.3)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def clip_bbox(bbox: list[float], tile_x: int, tile_y: int, tile_size: int, min_box_size: float, min_keep_ratio: float):
    x, y, w, h = [float(value) for value in bbox]
    x2 = x + w
    y2 = y + h
    ix1 = max(x, float(tile_x))
    iy1 = max(y, float(tile_y))
    ix2 = min(x2, float(tile_x + tile_size))
    iy2 = min(y2, float(tile_y + tile_size))
    clipped_w = ix2 - ix1
    clipped_h = iy2 - iy1
    if clipped_w < min_box_size or clipped_h < min_box_size:
        return None
    visible = clipped_w * clipped_h
    if visible / max(w * h, 1.0) < min_keep_ratio:
        return None
    return [ix1 - tile_x, iy1 - tile_y, clipped_w, clipped_h], visible


def generate_split(
    experiment_root: Path,
    split: str,
    tile_size: int,
    overlap: float,
    min_box_size: float,
    min_keep_ratio: float,
) -> dict:
    source = load_json(
        experiment_root / "data" / "panorama_coco" / "annotations" / f"instances_{split}_panorama_single.json"
    )
    out_dir = experiment_root / "data" / "coco_tiles" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = int(round(tile_size * (1.0 - overlap)))
    annotations_by_image = defaultdict(list)
    for ann in source["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1
    widths = []
    heights = []
    condition_counts = Counter()
    positive_tiles = 0
    negative_tiles = 0

    for source_image in source["images"]:
        source_image_id = int(source_image["id"])
        source_anns = annotations_by_image[source_image_id]
        x_starts = tile_starts(int(source_image["width"]), tile_size, stride)
        y_starts = tile_starts(int(source_image["height"]), tile_size, stride)
        for tile_y in y_starts:
            for tile_x in x_starts:
                tile_image = {
                    "id": next_image_id,
                    "file_name": f"virtual_tiles/{split}/{source_image['sample_id']}_x{tile_x}_y{tile_y}_s{tile_size}.png",
                    "width": tile_size,
                    "height": tile_size,
                    "source_file_name": source_image["file_name"],
                    "source_image_id": source_image_id,
                    "sample_id": source_image["sample_id"],
                    "recording_key": source_image.get("recording_key"),
                    "condition": source_image.get("condition"),
                    "scene": source_image.get("scene"),
                    "session": source_image.get("session"),
                    "split": split,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_size": tile_size,
                    "source_width": int(source_image["width"]),
                    "source_height": int(source_image["height"]),
                    "virtual_tile": True,
                }
                images.append(tile_image)
                condition_counts[str(source_image.get("condition") or "unknown")] += 1
                tile_positive = False
                for source_ann in source_anns:
                    clipped = clip_bbox(
                        source_ann["bbox"],
                        tile_x,
                        tile_y,
                        tile_size,
                        min_box_size,
                        min_keep_ratio,
                    )
                    if clipped is None:
                        continue
                    clipped_bbox, clipped_area = clipped
                    widths.append(clipped_bbox[2])
                    heights.append(clipped_bbox[3])
                    annotations.append(
                        {
                            "id": next_ann_id,
                            "image_id": next_image_id,
                            "category_id": 1,
                            "bbox": clipped_bbox,
                            "area": clipped_area,
                            "iscrowd": 0,
                            "source_annotation_id": source_ann["id"],
                        }
                    )
                    next_ann_id += 1
                    tile_positive = True
                if tile_positive:
                    positive_tiles += 1
                else:
                    negative_tiles += 1
                next_image_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "uav", "supercategory": "uav"}],
    }
    out_path = out_dir / f"instances_{split}_tiles_full_virtual.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    audit = {
        "split": split,
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "images": len(images),
        "annotations": len(annotations),
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "median_width_px": sorted(widths)[len(widths) // 2] if widths else None,
        "median_height_px": sorted(heights)[len(heights) // 2] if heights else None,
        "condition_counts": dict(sorted(condition_counts.items())),
        "ann_file": str(out_path.resolve()),
    }
    (experiment_root / "summary" / f"tile_audit_{split}.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return audit


def main() -> None:
    args = parse_args()
    (args.experiment_root / "summary").mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = generate_split(
            args.experiment_root,
            split,
            args.tile_size,
            args.overlap,
            args.min_box_size,
            args.min_box_keep_ratio,
        )
    out_path = args.experiment_root / "summary" / "tile_audit_all.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
