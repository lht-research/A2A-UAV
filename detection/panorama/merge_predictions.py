from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from mmengine.config import Config
from mmengine.fileio import dump
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val_benchmark", "val_full", "test_full"], required=True)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--outfile-prefix", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--iou-thr", type=float, default=None)
    parser.add_argument("--max-dets-per-image", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_best_checkpoint(run_dir: Path) -> Path | None:
    status_path = run_dir / "best_checkpoint_status.json"
    if status_path.exists():
        return Path(load_json(status_path)["checkpoint"])
    candidates = sorted(run_dir.glob("best_*.pth"))
    if candidates:
        return candidates[-1]
    iter_candidates = sorted(run_dir.glob("iter_*.pth"))
    if iter_candidates:
        return iter_candidates[-1]
    return None


def config_path_or_default(cfg: Config, key: str, default: Path) -> Path:
    value = getattr(cfg, key, None)
    if value is None:
        return default
    return Path(value)


def split_ann_files(experiment_root: Path, cfg: Config, split: str) -> tuple[Path, Path]:
    if split == "val_benchmark":
        tile_ann = experiment_root / "data" / "coco_tiles" / "annotations" / "instances_val_tiles_benchmark.json"
        gt_ann = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_val_benchmark_panorama_single.json"
    elif split == "val_full":
        tile_ann = config_path_or_default(
            cfg,
            "val_full_ann_file",
            experiment_root / "data" / "coco_tiles" / "annotations" / "instances_val_tiles_full_virtual.json",
        )
        gt_ann = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_val_panorama_single.json"
    else:
        tile_ann = config_path_or_default(
            cfg,
            "test_full_ann_file",
            experiment_root / "data" / "coco_tiles" / "annotations" / "instances_test_tiles_full_virtual.json",
        )
        gt_ann = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_test_panorama_single.json"
    return tile_ann, gt_ann


def load_config_for_summary(path: Path) -> Config:
    try:
        return Config.fromfile(str(path))
    except ImportError as exc:
        text = path.read_text(encoding="utf-8")
        if "custom_imports" not in text:
            raise
        patched = text + "\ncustom_imports = dict(imports=[], allow_failed_imports=True)\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
            handle.write(patched)
            tmp_path = Path(handle.name)
        try:
            return Config.fromfile(str(tmp_path))
        except Exception:
            raise exc
        finally:
            tmp_path.unlink(missing_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def empty_group() -> dict[str, object]:
    return {"scores": [], "labels": [], "bboxes": []}


def flush_group(image_id: int, group: dict[str, object], cat_ids: list[int], iou_thr: float, max_dets: int) -> tuple[list[dict], int]:
    scores_list = group["scores"]
    if not scores_list:
        return [], 0
    scores = torch.tensor(scores_list, dtype=torch.float32)
    labels = torch.tensor(group["labels"], dtype=torch.int64)
    boxes = torch.tensor(group["bboxes"], dtype=torch.float32)
    keep = batched_nms(boxes, scores, labels, iou_thr)
    if max_dets > 0:
        keep = keep[:max_dets]
    keep_scores = scores[keep]
    keep_labels = labels[keep]
    keep_boxes = boxes[keep]
    merged = []
    for box, score, label in zip(keep_boxes.tolist(), keep_scores.tolist(), keep_labels.tolist()):
        x1, y1, x2, y2 = box
        merged.append(
            {
                "image_id": int(image_id),
                "category_id": cat_ids[int(label)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return merged, int(keep_scores.numel())


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    code_dir = Path(__file__).resolve().parent
    os.environ["A2AUAV_BENCHMARK_ROOT"] = str(experiment_root.parent)
    os.chdir(experiment_root)
    sys.path.insert(0, str(code_dir))
    cfg = load_config_for_summary(code_dir / "configs" / f"{args.run_name}.py")
    run_dir = experiment_root / "runs" / args.run_name
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = args.results_jsonl or (eval_dir / f"{args.split}_predictions.jsonl")
    outfile_prefix = args.outfile_prefix or (eval_dir / args.split)
    checkpoint = args.checkpoint or resolve_best_checkpoint(run_dir)
    tile_ann, gt_ann = split_ann_files(experiment_root, cfg, args.split)
    tile_count = len(load_json(tile_ann).get("images", []))
    coco_gt = COCO(str(gt_ann))
    cat_ids = coco_gt.getCatIds()
    img_ids = coco_gt.getImgIds()
    iou_thr = float(args.iou_thr if args.iou_thr is not None else cfg.test_evaluator.get("iou_thr", 0.55))
    max_dets = int(args.max_dets_per_image)

    prefix = Path(outfile_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_path = prefix.with_suffix(".merged_bbox.json")
    summary_path = prefix.with_suffix(".merge_summary.json")
    metrics_path = eval_dir / f"{args.split}_metrics.json"
    progress_path = eval_dir / f"{args.split}_streaming_summary_progress.json"

    current_image_id = None
    current_group = empty_group()
    merged_predictions: list[dict] = []
    tile_prediction_count = 0
    merged_prediction_count = 0
    records = 0
    panorama_count = 0
    bad_order = False
    t0 = time.time()

    for record in iter_jsonl(results_jsonl):
        records += 1
        image_id = int(record["panorama_image_id"])
        if current_image_id is None:
            current_image_id = image_id
        if image_id != current_image_id:
            merged, kept = flush_group(current_image_id, current_group, cat_ids, iou_thr, max_dets)
            merged_predictions.extend(merged)
            merged_prediction_count += kept
            panorama_count += 1
            current_image_id = image_id
            current_group = empty_group()
        if image_id != current_image_id:
            bad_order = True
        scores = record.get("scores", [])
        current_group["scores"].extend(scores)
        current_group["labels"].extend(record.get("labels", []))
        current_group["bboxes"].extend(record.get("bboxes", []))
        tile_prediction_count += len(scores)
        if args.log_every > 0 and records % args.log_every == 0:
            progress_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "records": records,
                        "tile_count": tile_count,
                        "panorama_count_flushed": panorama_count,
                        "tile_prediction_count": tile_prediction_count,
                        "merged_prediction_count": merged_prediction_count,
                        "elapsed_sec": round(time.time() - t0, 2),
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

    if current_image_id is not None:
        merged, kept = flush_group(current_image_id, current_group, cat_ids, iou_thr, max_dets)
        merged_predictions.extend(merged)
        merged_prediction_count += kept
        panorama_count += 1

    dump(merged_predictions, str(merged_path))
    summary_payload = {
        "ann_file": str(gt_ann),
        "results_jsonl_path": str(results_jsonl),
        "iou_thr": iou_thr,
        "max_dets_per_image": max_dets,
        "jsonl_record_count": records,
        "expected_tile_count": tile_count,
        "tile_prediction_count": tile_prediction_count,
        "merged_prediction_count": merged_prediction_count,
        "panorama_count": panorama_count,
        "result_file": str(merged_path),
        "streaming": True,
        "input_panorama_order_changed": bad_order,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    coco_dt = coco_gt.loadRes(merged_predictions) if merged_predictions else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats.tolist()
    raw_metrics = {
        "bbox_mAP": float(stats[0]),
        "bbox_mAP_50": float(stats[1]),
        "bbox_mAP_75": float(stats[2]),
        "bbox_mAP_s": float(stats[3]),
        "bbox_mAP_m": float(stats[4]),
        "bbox_mAP_l": float(stats[5]),
        "tile_prediction_count": float(tile_prediction_count),
        "merged_prediction_count": float(merged_prediction_count),
        "panorama_count": float(panorama_count),
    }
    metrics = {f"panorama/{key}": value for key, value in raw_metrics.items()}
    payload = {
        "split": args.split,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "metrics": metrics,
        "tile_ann_file": str(tile_ann),
        "gt_ann_file": str(gt_ann),
        "results_jsonl_path": str(results_jsonl),
        "streaming_summary": True,
        "max_dets_per_image": max_dets,
        "merge_summary_path": str(summary_path),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    progress_path.write_text(
        json.dumps({"status": "completed", "records": records, "metrics_path": str(metrics_path)}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(metrics_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
