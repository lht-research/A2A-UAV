from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--pkl-file", type=Path, required=True)
    parser.add_argument("--jsonl-out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_array_for_one_class(item: object) -> np.ndarray:
    if isinstance(item, (list, tuple)):
        if not item:
            return np.zeros((0, 5), dtype=np.float32)
        arrays = [
            np.asarray(arr, dtype=np.float32).reshape((-1, 5))
            for arr in item
            if np.asarray(arr).size
        ]
        if not arrays:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate(arrays, axis=0)
    arr = np.asarray(item, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return arr.reshape((-1, 5))


def main() -> None:
    args = parse_args()
    ann = load_json(args.ann_file)
    images = ann.get("images", [])
    with args.pkl_file.open("rb") as handle:
        outputs = pickle.load(handle)
    if len(outputs) != len(images):
        raise RuntimeError(f"pkl output count {len(outputs)} != annotation image count {len(images)}")

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    dets = 0
    with args.jsonl_out.open("w", encoding="utf-8") as handle:
        for image, output in zip(images, outputs):
            arr = as_array_for_one_class(output)
            tile_x = float(image.get("tile_x", 0))
            tile_y = float(image.get("tile_y", 0))
            if arr.shape[0]:
                bboxes = arr[:, :4].copy()
                bboxes[:, 0::2] += tile_x
                bboxes[:, 1::2] += tile_y
                scores = arr[:, 4].astype(np.float32).tolist()
                labels = [0] * int(arr.shape[0])
                bboxes_list = bboxes.astype(np.float32).tolist()
            else:
                scores = []
                labels = []
                bboxes_list = []
            record = {
                "sample_key": f"tile_id:{int(image['id'])}",
                "tile_image_id": int(image["id"]),
                "panorama_image_id": int(image.get("source_image_id", image["id"])),
                "sample_id": image.get("sample_id"),
                "source_width": int(image.get("source_width", image.get("width", 0))),
                "source_height": int(image.get("source_height", image.get("height", 0))),
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes_list,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            records += 1
            dets += len(scores)

    print(json.dumps({"jsonl_out": str(args.jsonl_out), "records": records, "detections": dets}, indent=2))


if __name__ == "__main__":
    main()
