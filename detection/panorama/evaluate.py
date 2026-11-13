from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val_benchmark", "val_full", "test_full"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_best_checkpoint(run_dir: Path) -> Path:
    status_path = run_dir / "best_checkpoint_status.json"
    if status_path.exists():
        payload = load_json(status_path)
        return Path(payload["checkpoint"])
    candidates = sorted(run_dir.glob("best_*.pth"))
    if candidates:
        return candidates[-1]
    iter_candidates = sorted(run_dir.glob("iter_*.pth"))
    if iter_candidates:
        return iter_candidates[-1]
    raise FileNotFoundError(f"no checkpoint found under {run_dir}")


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_sample_keys(results_jsonl_path: Path, metric_cls) -> set[str]:
    if not results_jsonl_path.exists():
        return set()
    keys: dict[str, None] = {}
    with results_jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = metric_cls.sample_key(record.get("sample_id"), record.get("tile_image_id"))
            if key in keys:
                del keys[key]
            keys[key] = None
    return set(keys)


def build_remaining_subset(source: dict, completed_sample_keys: set[str], metric_cls) -> tuple[dict, int]:
    if not completed_sample_keys:
        return source, 0
    remaining_images = []
    remaining_ids = set()
    skipped = 0
    for image in source.get("images", []):
        sample_key = metric_cls.sample_key(image.get("sample_id"), image.get("id"))
        if sample_key in completed_sample_keys:
            skipped += 1
            continue
        remaining_images.append(image)
        remaining_ids.add(int(image["id"]))
    remaining_annotations = [ann for ann in source.get("annotations", []) if int(ann["image_id"]) in remaining_ids]
    subset = dict(source)
    subset["images"] = remaining_images
    subset["annotations"] = remaining_annotations
    return subset, skipped


def _prepend_pythonpath(path: Path) -> str:
    prior = os.environ.get("PYTHONPATH")
    return str(path) + (os.pathsep + prior if prior else "")


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    code_dir = Path(__file__).resolve().parent
    os.environ["A2AUAV_BENCHMARK_ROOT"] = str(experiment_root.parent)
    sys.path.insert(0, str(code_dir))
    os.chdir(experiment_root)
    os.environ["PYTHONPATH"] = _prepend_pythonpath(code_dir)
    init_default_scope("mmdet")
    import lib.uav_coco_tile_dataset
    import lib.tile_loading
    import lib.wall_clock_hook
    import lib.panorama_merge_metric as panorama_merge_metric
    import lib.protocol_runtime

    cfg = Config.fromfile(str(code_dir / "configs" / f"{args.run_name}.py"))
    run_dir = experiment_root / "runs" / args.run_name
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    progress_path = eval_dir / f"{args.split}_progress.json"
    results_jsonl_path = eval_dir / f"{args.split}_predictions.jsonl"
    if args.checkpoint is None:
        args.checkpoint = resolve_best_checkpoint(run_dir)

    if args.split == "val_benchmark":
        ann_file = experiment_root / "data" / "coco_tiles" / "annotations" / "instances_val_tiles_benchmark.json"
        gt_ann_file = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_val_benchmark_panorama_single.json"
    elif args.split == "val_full":
        ann_file = Path(cfg.val_full_ann_file)
        gt_ann_file = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_val_panorama_single.json"
    else:
        ann_file = Path(cfg.test_full_ann_file)
        gt_ann_file = experiment_root / "data" / "panorama_coco" / "annotations" / "instances_test_panorama_single.json"

    tile_source = load_coco(ann_file)
    total_tile_count = len(tile_source.get("images", []))
    completed_sample_keys = load_completed_sample_keys(
        results_jsonl_path, panorama_merge_metric.PanoramaMergeCocoMetric
    )

    if args.resume and completed_sample_keys:
        remaining_subset, skipped = build_remaining_subset(
            tile_source, completed_sample_keys, panorama_merge_metric.PanoramaMergeCocoMetric
        )
        if not remaining_subset.get("images"):
            metrics = panorama_merge_metric.PanoramaMergeCocoMetric.summarize_from_jsonl(
                ann_file=str(gt_ann_file),
                results_jsonl_path=results_jsonl_path,
                outfile_prefix=str((eval_dir / args.split).resolve()),
                iou_thr=float(cfg.test_evaluator.get("iou_thr", 0.55)),
                max_dets_per_image=int(cfg.test_evaluator.get("max_dets_per_image", 1000)),
            )
            out_path = eval_dir / f"{args.split}_metrics.json"
            out_path.write_text(
                json.dumps(
                    {
                        "split": args.split,
                        "checkpoint": str(args.checkpoint),
                        "metrics": metrics,
                        "tile_ann_file": str(ann_file),
                        "gt_ann_file": str(gt_ann_file),
                        "resumed": True,
                        "completed_sample_count": len(completed_sample_keys),
                        "remaining_sample_count": 0,
                        "results_jsonl_path": str(results_jsonl_path),
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(out_path.read_text(encoding="utf-8"))
            return
        resume_ann_file = eval_dir / f"{args.split}_resume_remaining.json"
        resume_ann_file.write_text(json.dumps(remaining_subset, ensure_ascii=True, indent=2), encoding="utf-8")
        ann_file = resume_ann_file
        tile_source = remaining_subset
    else:
        skipped = 0

    cfg.work_dir = str(run_dir / f"eval_{args.split}")
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.persistent_workers = args.num_workers > 0
    cfg.test_dataloader.dataset.ann_file = str(ann_file)
    cfg.test_evaluator.ann_file = str(gt_ann_file)
    cfg.test_evaluator.outfile_prefix = str((eval_dir / args.split).resolve())
    cfg.test_evaluator.results_jsonl_path = str(results_jsonl_path)
    cfg.test_evaluator.progress_path = str(progress_path)
    cfg.test_evaluator.expected_total_samples = total_tile_count
    cfg.test_evaluator.completed_before_run = len(completed_sample_keys)
    cfg.test_evaluator.keep_in_memory = False
    cfg.load_from = str(args.checkpoint)
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    out_path = eval_dir / f"{args.split}_metrics.json"
    out_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "metrics": metrics,
                "tile_ann_file": str(ann_file),
                "gt_ann_file": str(gt_ann_file),
                "resumed": bool(args.resume),
                "completed_sample_count": len(completed_sample_keys),
                "remaining_sample_count": len(tile_source.get("images", [])),
                "skipped_sample_count": skipped,
                "results_jsonl_path": str(results_jsonl_path),
                "progress_path": str(progress_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
