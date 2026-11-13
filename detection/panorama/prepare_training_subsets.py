from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--negative-total-fraction", type=float, default=0.20)
    parser.add_argument("--monitor-positive-target", type=int, default=1500)
    parser.add_argument("--monitor-negative-target", type=int, default=500)
    parser.add_argument("--monitor-min-annotations", type=int, default=1500)
    parser.add_argument("--overfit-positive-target", type=int, default=256)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def subset_by_image_ids(source: dict, selected_image_ids: set[int]) -> dict:
    return {
        "images": [image for image in source["images"] if int(image["id"]) in selected_image_ids],
        "annotations": [ann for ann in source["annotations"] if int(ann["image_id"]) in selected_image_ids],
        "categories": source["categories"],
    }


def annotations_by_image(source: dict) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for ann in source["annotations"]:
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def stratified_negative_sample(images: list[dict], keep_count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for image in images:
        key = str(image.get("condition") or "unknown")
        groups[key].append(image)
    rng = random.Random(seed)
    selected = []
    leftovers = []
    total = len(images)
    for condition, bucket in sorted(groups.items()):
        rng.shuffle(bucket)
        quota = int(round(keep_count * len(bucket) / max(total, 1)))
        selected.extend(bucket[:quota])
        leftovers.extend(bucket[quota:])
    selected_ids = {int(image["id"]) for image in selected}
    shortfall = keep_count - len(selected)
    if shortfall > 0:
        rng.shuffle(leftovers)
        for image in leftovers:
            image_id = int(image["id"])
            if image_id in selected_ids:
                continue
            selected.append(image)
            selected_ids.add(image_id)
            shortfall -= 1
            if shortfall <= 0:
                break
    selected.sort(key=lambda item: (str(item.get("condition") or ""), str(item.get("recording_key") or ""), int(item["id"])))
    return selected[:keep_count]


def build_train_neg20(train_source: dict, seed: int, negative_total_fraction: float) -> tuple[dict, dict]:
    anns_by_image = annotations_by_image(train_source)
    positive_images = [image for image in train_source["images"] if anns_by_image.get(int(image["id"]))]
    negative_images = [image for image in train_source["images"] if not anns_by_image.get(int(image["id"]))]
    negative_keep = min(
        len(negative_images),
        int(round(len(positive_images) * negative_total_fraction / max(1.0 - negative_total_fraction, 1e-9))),
    )
    selected_negative = stratified_negative_sample(negative_images, negative_keep, seed)
    selected_ids = {int(image["id"]) for image in positive_images + selected_negative}
    subset = subset_by_image_ids(train_source, selected_ids)
    audit = {
        "positive_tiles": len(positive_images),
        "negative_tiles_available": len(negative_images),
        "negative_tiles_selected": len(selected_negative),
        "negative_fraction_of_total": len(selected_negative) / max(len(subset["images"]), 1),
        "seed": seed,
        "condition_counts": _count_by_key(subset["images"], "condition"),
        "recording_key_count": len({str(image.get("recording_key") or "") for image in subset["images"]}),
        "annotations": len(subset["annotations"]),
    }
    return subset, audit


def _count_by_key(images: list[dict], key: str) -> dict[str, int]:
    counts = defaultdict(int)
    for image in images:
        counts[str(image.get(key) or "unknown")] += 1
    return dict(sorted(counts.items()))


def build_monitor_subset(
    val_source: dict,
    seed: int,
    positive_target: int,
    negative_target: int,
    min_annotations: int,
) -> tuple[dict, dict, list[int]]:
    anns_by_image = annotations_by_image(val_source)
    positive_images = [image for image in val_source["images"] if anns_by_image.get(int(image["id"]))]
    negative_images = [image for image in val_source["images"] if not anns_by_image.get(int(image["id"]))]
    rng = random.Random(seed)

    positive_buckets = defaultdict(list)
    for image in positive_images:
        positive_buckets[str(image.get("condition") or "unknown")].append(image)
    negative_buckets = defaultdict(list)
    for image in negative_images:
        negative_buckets[str(image.get("condition") or "unknown")].append(image)

    selected_positive = []
    for condition in sorted(positive_buckets):
        bucket = positive_buckets[condition]
        rng.shuffle(bucket)
        quota = max(1, int(round(positive_target * len(bucket) / max(len(positive_images), 1))))
        selected_positive.extend(bucket[:quota])
    if len(selected_positive) < positive_target:
        remaining = [image for image in positive_images if int(image["id"]) not in {int(row["id"]) for row in selected_positive}]
        rng.shuffle(remaining)
        selected_positive.extend(remaining[: positive_target - len(selected_positive)])
    selected_positive = selected_positive[:positive_target]

    annotations = sum(len(anns_by_image[int(image["id"])]) for image in selected_positive)
    if annotations < min_annotations:
        remaining = [image for image in positive_images if int(image["id"]) not in {int(row["id"]) for row in selected_positive}]
        remaining.sort(key=lambda image: len(anns_by_image[int(image["id"])]), reverse=True)
        for image in remaining:
            selected_positive.append(image)
            annotations += len(anns_by_image[int(image["id"])])
            if annotations >= min_annotations:
                break

    selected_negative = []
    for condition in sorted(negative_buckets):
        bucket = negative_buckets[condition]
        rng.shuffle(bucket)
        quota = max(1, int(round(negative_target * len(bucket) / max(len(negative_images), 1))))
        selected_negative.extend(bucket[:quota])
    if len(selected_negative) < negative_target:
        remaining = [image for image in negative_images if int(image["id"]) not in {int(row["id"]) for row in selected_negative}]
        rng.shuffle(remaining)
        selected_negative.extend(remaining[: negative_target - len(selected_negative)])
    selected_negative = selected_negative[:negative_target]

    selected_ids = {int(image["id"]) for image in selected_positive + selected_negative}
    subset = subset_by_image_ids(val_source, selected_ids)
    probe_candidates = sorted(selected_positive, key=lambda item: (str(item.get("condition") or ""), int(item["id"])))
    probe_image_ids = [int(image["id"]) for image in probe_candidates[:8]]
    audit = {
        "positive_tiles": len(selected_positive),
        "negative_tiles": len(selected_negative),
        "annotations": len(subset["annotations"]),
        "seed": seed,
        "condition_counts": _count_by_key(subset["images"], "condition"),
        "probe_image_ids": probe_image_ids,
    }
    return subset, audit, probe_image_ids


def build_overfit_subset(train_source: dict, seed: int, positive_target: int) -> tuple[dict, dict]:
    anns_by_image = annotations_by_image(train_source)
    positive_images = [image for image in train_source["images"] if anns_by_image.get(int(image["id"]))]
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for image in positive_images:
        grouped[str(image.get("condition") or "unknown")].append(image)
    selected = []
    for condition in sorted(grouped):
        bucket = grouped[condition]
        rng.shuffle(bucket)
        quota = max(1, int(round(positive_target * len(bucket) / max(len(positive_images), 1))))
        selected.extend(bucket[:quota])
    if len(selected) < positive_target:
        remaining = [image for image in positive_images if int(image["id"]) not in {int(row["id"]) for row in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: positive_target - len(selected)])
    selected = selected[:positive_target]
    selected_ids = {int(image["id"]) for image in selected}
    subset = subset_by_image_ids(train_source, selected_ids)
    audit = {
        "positive_tiles": len(selected),
        "annotations": len(subset["annotations"]),
        "seed": seed,
        "condition_counts": _count_by_key(subset["images"], "condition"),
    }
    return subset, audit


def build_gate1_subset(train_source: dict, run_name: str) -> tuple[dict, dict]:
    anns_by_image = annotations_by_image(train_source)
    positive_images = [image for image in train_source["images"] if anns_by_image.get(int(image["id"]))]
    batch_size = 4 if run_name in {"faster_rcnn", "dynamic_rcnn", "cascade_rcnn", "sparse_rcnn"} else 1
    selected = positive_images[:batch_size]
    selected_ids = {int(image["id"]) for image in selected}
    subset = subset_by_image_ids(train_source, selected_ids)
    audit = {
        "run_name": run_name,
        "images": len(subset["images"]),
        "annotations": len(subset["annotations"]),
        "batch_size": batch_size,
    }
    return subset, audit




def main() -> None:
    args = parse_args()
    ann_dir = args.experiment_root / "data" / "coco_tiles" / "annotations"
    train_source = load_json(ann_dir / "instances_train_tiles_full_virtual.json")
    val_source = load_json(ann_dir / "instances_val_tiles_full_virtual.json")

    train_neg20, train_audit = build_train_neg20(train_source, args.seed, args.negative_total_fraction)
    train_neg20_path = ann_dir / "instances_train_tiles_neg20_virtual.json"
    save_json(train_neg20_path, train_neg20)

    monitor_subset, monitor_audit, probe_image_ids = build_monitor_subset(
        val_source,
        args.seed,
        args.monitor_positive_target,
        args.monitor_negative_target,
        args.monitor_min_annotations,
    )
    monitor_path = ann_dir / "instances_val_tiles_monitor_virtual.json"
    save_json(monitor_path, monitor_subset)

    overfit_subset, overfit_audit = build_overfit_subset(train_source, args.seed, args.overfit_positive_target)
    overfit_path = ann_dir / "instances_train_tiles_overfit256_virtual.json"
    save_json(overfit_path, overfit_subset)

    gate1_payload = {}
    for run_name in (
        "faster_rcnn",
        "dynamic_rcnn",
        "cascade_rcnn",
        "sparse_rcnn",
        "dino",
        "ddq",
    ):
        subset, audit = build_gate1_subset(train_source, run_name)
        out_path = ann_dir / f"instances_train_tiles_gate1_{run_name}_virtual.json"
        save_json(out_path, subset)
        audit["ann_file"] = str(out_path.resolve())
        gate1_payload[run_name] = audit


    payload = {
        "train_neg20": {
            **train_audit,
            "ann_file": str(train_neg20_path.resolve()),
        },
        "val_monitor": {
            **monitor_audit,
            "ann_file": str(monitor_path.resolve()),
        },
        "overfit256": {
            **overfit_audit,
            "ann_file": str(overfit_path.resolve()),
        },
        "gate1": gate1_payload,
    }
    save_json(args.experiment_root / "summary" / "protocol_subset_audit.json", payload)
    save_json(
        args.experiment_root / "summary" / "probe_tiles.json",
        {
            "probe_image_ids": probe_image_ids,
            "monitor_ann_file": str(monitor_path.resolve()),
        },
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
