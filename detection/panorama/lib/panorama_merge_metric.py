from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms

from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmdet.registry import METRICS


@METRICS.register_module()
class PanoramaMergeCocoMetric(BaseMetric):

    default_prefix = "panorama"

    def __init__(
        self,
        ann_file: str,
        outfile_prefix: str | None = None,
        iou_thr: float = 0.55,
        max_dets_per_image: int = 1000,
        results_jsonl_path: str | None = None,
        progress_path: str | None = None,
        expected_total_samples: int | None = None,
        completed_before_run: int = 0,
        keep_in_memory: bool = True,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = str(ann_file)
        self.outfile_prefix = outfile_prefix
        self.iou_thr = float(iou_thr)
        self.max_dets_per_image = int(max_dets_per_image)
        self.results_jsonl_path = Path(results_jsonl_path) if results_jsonl_path else None
        self.progress_path = Path(progress_path) if progress_path else None
        self.expected_total_samples = int(expected_total_samples) if expected_total_samples is not None else None
        self.completed_before_run = int(completed_before_run)
        self.keep_in_memory = bool(keep_in_memory)
        self.coco_api = COCO(self.ann_file)
        self.cat_ids = self.coco_api.getCatIds()
        self.img_ids = self.coco_api.getImgIds()
        self._processed_keys_current_run: set[str] = set()

        if self.results_jsonl_path:
            self.results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.progress_path:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_progress("initialized")

    @staticmethod
    def sample_key(sample_id: object, tile_image_id: object) -> str:


        return f"tile_id:{int(tile_image_id)}"

    @classmethod
    def load_results_jsonl(cls, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(payload)
        return records

    @classmethod
    def dedupe_records(cls, records: Iterable[dict]) -> list[dict]:
        unique: dict[str, dict] = {}
        for record in records:
            key = cls.sample_key(record.get("sample_id"), record.get("tile_image_id"))
            if key in unique:
                del unique[key]
            unique[key] = record
        return list(unique.values())

    def _write_progress(self, status: str, **extra: object) -> None:
        if not self.progress_path:
            return
        payload = {
            "status": status,
            "ann_file": self.ann_file,
            "results_jsonl_path": str(self.results_jsonl_path) if self.results_jsonl_path else None,
            "processed_current_run_samples": len(self._processed_keys_current_run),
            "completed_before_run_samples": self.completed_before_run,
            "processed_total_samples": self.completed_before_run + len(self._processed_keys_current_run),
            "expected_total_samples": self.expected_total_samples,
            "updated_at": time.time(),
            **extra,
        }
        self.progress_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _append_result_jsonl(self, record: dict) -> None:
        if not self.results_jsonl_path:
            return
        with self.results_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _group_predictions(self, records: Iterable[dict]) -> tuple[dict[int, dict[str, object]], int]:
        grouped: dict[int, dict[str, object]] = defaultdict(
            lambda: {
                "scores": [],
                "labels": [],
                "bboxes": [],
                "sample_id": None,
                "source_width": 0,
                "source_height": 0,
            }
        )
        merge_before = 0
        for item in records:
            target = grouped[int(item["panorama_image_id"])]
            target["scores"].extend(item["scores"])
            target["labels"].extend(item["labels"])
            target["bboxes"].extend(item["bboxes"])
            target["sample_id"] = item.get("sample_id")
            target["source_width"] = item.get("source_width", 0)
            target["source_height"] = item.get("source_height", 0)
            merge_before += len(item["scores"])
        return grouped, merge_before

    def _evaluate_records(self, records: Iterable[dict]) -> Dict[str, float]:
        return self.summarize_from_records(records)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            if hasattr(sample, "pred_instances") and hasattr(sample, "metainfo"):
                pred_instances = sample.pred_instances
                metainfo = sample.metainfo
            elif isinstance(sample, dict):
                pred_instances = sample["pred_instances"]
                metainfo = sample.get("metainfo", sample)
            else:
                pred_instances = sample["pred_instances"]
                metainfo = getattr(sample, "metainfo", {})
            bboxes = pred_instances["bboxes"].detach().cpu()
            scores = pred_instances["scores"].detach().cpu()
            labels = pred_instances["labels"].detach().cpu()
            tile_x = float(metainfo.get("tile_x", 0))
            tile_y = float(metainfo.get("tile_y", 0))
            tile_image_id = int(metainfo["img_id"])
            sample_id = metainfo.get("sample_id")
            sample_key = self.sample_key(sample_id, tile_image_id)
            panorama_image_id = int(metainfo.get("source_image_id", metainfo["img_id"]))
            source_width = int(metainfo.get("source_width", metainfo.get("width", 0)))
            source_height = int(metainfo.get("source_height", metainfo.get("height", 0)))

            if len(bboxes) > 0:
                bboxes = bboxes.clone()
                bboxes[:, 0::2] += tile_x
                bboxes[:, 1::2] += tile_y

            self.results.append(
                {
                    "sample_key": sample_key,
                    "tile_image_id": tile_image_id,
                    "panorama_image_id": panorama_image_id,
                    "sample_id": sample_id,
                    "source_width": source_width,
                    "source_height": source_height,
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),
                    "bboxes": bboxes.tolist(),
                }
            )
            record = self.results[-1]
            self._processed_keys_current_run.add(sample_key)
            self._append_result_jsonl(record)
            self._write_progress("running", last_sample_key=sample_key)
            if not self.keep_in_memory:
                self.results.pop()

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        if self.results_jsonl_path and self.results_jsonl_path.exists():
            records = self.dedupe_records(self.load_results_jsonl(self.results_jsonl_path))
        else:
            records = self.dedupe_records(results)
        metrics = self._evaluate_records(records)
        self._write_progress(
            "completed",
            processed_total_samples=len(records),
            metrics=metrics,
        )
        return metrics

    def _materialize_outputs(self, grouped: dict[int, dict[str, object]], merge_before: int) -> tuple[list[dict], int]:
        merged_predictions = []
        merge_after = 0
        for image_id, item in grouped.items():
            if not item["scores"]:
                continue
            scores = torch.tensor(item["scores"], dtype=torch.float32)
            labels = torch.tensor(item["labels"], dtype=torch.int64)
            boxes = torch.tensor(item["bboxes"], dtype=torch.float32)
            keep = batched_nms(boxes, scores, labels, self.iou_thr)
            if self.max_dets_per_image > 0:
                keep = keep[: self.max_dets_per_image]
            keep_scores = scores[keep]
            keep_labels = labels[keep]
            keep_boxes = boxes[keep]
            merge_after += int(keep_scores.numel())
            for box, score, label in zip(keep_boxes.tolist(), keep_scores.tolist(), keep_labels.tolist()):
                x1, y1, x2, y2 = box
                merged_predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": self.cat_ids[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
        return merged_predictions, merge_after

    def summarize_from_records(self, records: Iterable[dict]) -> Dict[str, float]:
        grouped, merge_before = self._group_predictions(records)
        merged_predictions, merge_after = self._materialize_outputs(grouped, merge_before)

        if self.outfile_prefix:
            prefix = Path(self.outfile_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            merged_path = prefix.with_suffix(".merged_bbox.json")
            dump(merged_predictions, str(merged_path))
            summary_path = prefix.with_suffix(".merge_summary.json")
            summary_payload = {
                "ann_file": self.ann_file,
                "iou_thr": self.iou_thr,
                "tile_prediction_count": merge_before,
                "merged_prediction_count": merge_after,
                "panorama_count": len(grouped),
                "result_file": str(merged_path),
            }
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        if merged_predictions:
            coco_dt = self.coco_api.loadRes(merged_predictions)
        else:
            coco_dt = self.coco_api.loadRes([])
        coco_eval = COCOeval(self.coco_api, coco_dt, "bbox")
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist()
        return {
            "bbox_mAP": float(stats[0]),
            "bbox_mAP_50": float(stats[1]),
            "bbox_mAP_75": float(stats[2]),
            "bbox_mAP_s": float(stats[3]),
            "bbox_mAP_m": float(stats[4]),
            "bbox_mAP_l": float(stats[5]),
            "tile_prediction_count": float(merge_before),
            "merged_prediction_count": float(merge_after),
            "panorama_count": float(len(grouped)),
        }

    @classmethod
    def summarize_from_jsonl(
        cls,
        ann_file: str,
        results_jsonl_path: str | Path,
        outfile_prefix: str | None = None,
        iou_thr: float = 0.55,
        max_dets_per_image: int = 1000,
    ) -> Dict[str, float]:
        metric = cls(
            ann_file=ann_file,
            outfile_prefix=outfile_prefix,
            iou_thr=iou_thr,
            max_dets_per_image=max_dets_per_image,
            results_jsonl_path=str(results_jsonl_path),
            keep_in_memory=False,
        )
        records = cls.dedupe_records(cls.load_results_jsonl(Path(results_jsonl_path)))
        return metric.summarize_from_records(records)
