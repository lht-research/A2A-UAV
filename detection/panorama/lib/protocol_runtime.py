from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import HOOKS, METRICS, OPTIM_WRAPPERS


def _iter_grad_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [param for param in params if param.requires_grad and param.grad is not None]


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    params = _iter_grad_params(params)
    if not params:
        return 0.0
    norms = [param.grad.detach().norm(2) for param in params]
    if not norms:
        return 0.0
    stacked = torch.stack([norm.to(norms[0].device) for norm in norms])
    return float(torch.norm(stacked, 2).item())


class _ProtocolGradNormMixin:
    def _clip_grad(self) -> None:
        params = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group["params"])
        params = _iter_grad_params(params)
        if not params:
            self.message_hub.update_scalar("train/grad_norm_pre", 0.0)
            self.message_hub.update_scalar("train/grad_norm", 0.0)
            self.message_hub.update_scalar("train/grad_norm_post", 0.0)
            return
        grad_norm_pre = _grad_norm(params)
        self.clip_func(params, **self.clip_grad_kwargs)
        grad_norm_post = _grad_norm(params)
        self.message_hub.update_scalar("train/grad_norm_pre", grad_norm_pre)
        self.message_hub.update_scalar("train/grad_norm", grad_norm_post)
        self.message_hub.update_scalar("train/grad_norm_post", grad_norm_post)


@OPTIM_WRAPPERS.register_module()
class ProtocolOptimWrapper(_ProtocolGradNormMixin, OptimWrapper):
    pass


@OPTIM_WRAPPERS.register_module()
class ProtocolAmpOptimWrapper(_ProtocolGradNormMixin, AmpOptimWrapper):
    pass


@HOOKS.register_module()
class FiniteLossGuardHook(Hook):
    priority = "LOW"

    def __init__(self, max_check_iters: int = 500, strict: bool = True) -> None:
        self.max_check_iters = int(max_check_iters)
        self.strict = bool(strict)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        current_iter = int(runner.iter) + 1
        if current_iter > self.max_check_iters or not isinstance(outputs, dict):
            return
        for key, value in outputs.items():
            if hasattr(value, "item"):
                scalar = float(value.item())
            elif isinstance(value, (int, float)):
                scalar = float(value)
            else:
                continue
            if math.isfinite(scalar):
                continue
            message = f"Non-finite train output at iter={current_iter}: {key}={scalar}"
            runner.logger.error(message)
            if self.strict:
                raise RuntimeError(message)
            runner.train_loop.stop_training = True
            return


@HOOKS.register_module()
class ProtocolBestCheckpointHook(Hook):
    priority = "LOWEST"

    def __init__(
        self,
        monitor: str,
        out_dir: str,
        min_score_to_save: float = 1e-12,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.out_dir = Path(out_dir)
        self.min_score_to_save = float(min_score_to_save)
        self.min_delta = float(min_delta)
        self.best_score: float | None = None

    def after_val_epoch(self, runner, metrics: dict) -> None:
        if not is_main_process():
            return
        if self.monitor not in metrics:
            return
        score = float(metrics[self.monitor])
        if not math.isfinite(score) or score <= self.min_score_to_save:
            return
        if self.best_score is not None and score <= self.best_score + self.min_delta:
            return
        current_iter = int(runner.iter)
        work_dir = Path(runner.work_dir)
        checkpoint_path = work_dir / f"iter_{current_iter}.pth"
        if not checkpoint_path.exists():
            candidates = sorted(work_dir.glob("iter_*.pth"))
            if not candidates:
                runner.logger.warning(
                    "ProtocolBestCheckpointHook could not find an iter checkpoint at iter=%s in %s",
                    current_iter,
                    work_dir,
                )
                return
            checkpoint_path = candidates[-1]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.out_dir / f"best_{self.monitor.replace('/', '_')}_iter_{current_iter}.pth"
        shutil.copy2(checkpoint_path, target_path)
        summary_path = self.out_dir / "best_checkpoint_status.json"
        summary_path.write_text(
            json.dumps(
                {
                    "monitor": self.monitor,
                    "score": score,
                    "iter": current_iter,
                    "checkpoint": str(target_path.resolve()),
                    "source_checkpoint": str(checkpoint_path.resolve()),
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.best_score = score
        runner.logger.info(
            "ProtocolBestCheckpointHook updated best checkpoint: %s score=%.6f",
            target_path,
            score,
        )


@HOOKS.register_module()
class ProtocolEarlyStopHook(Hook):
    priority = "LOWEST"

    def __init__(
        self,
        monitor: str,
        burn_in_iters: int = 6000,
        start_threshold: float = 0.02,
        patience: int = 3,
        min_delta: float = 0.0,
        hard_fail_iter: int = 12000,
        collapse_monitor: str = "monitor/probe_max_score",
        collapse_after_iter: int = 4000,
        collapse_threshold: float = 1e-3,
    ) -> None:
        self.monitor = monitor
        self.burn_in_iters = int(burn_in_iters)
        self.start_threshold = float(start_threshold)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.hard_fail_iter = int(hard_fail_iter)
        self.collapse_monitor = collapse_monitor
        self.collapse_after_iter = int(collapse_after_iter)
        self.collapse_threshold = float(collapse_threshold)
        self.best_score: float | None = None
        self.wait_count = 0
        self.active = False

    def _write_stop_reason(self, runner, payload: dict) -> None:
        if not is_main_process():
            return
        out_path = Path(runner.work_dir) / "protocol_stop_status.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def after_val_epoch(self, runner, metrics: dict) -> None:
        if self.monitor not in metrics:
            runner.logger.warning("ProtocolEarlyStopHook monitor missing: %s", self.monitor)
            return
        current_iter = int(runner.iter)
        current_score = float(metrics[self.monitor])
        probe_score = float(metrics.get(self.collapse_monitor, 0.0))

        if current_iter >= self.hard_fail_iter and current_score <= 0.0:
            reason = {
                "reason": "hard_fail_all_zero",
                "iter": current_iter,
                "monitor": self.monitor,
                "score": current_score,
            }
            self._write_stop_reason(runner, reason)
            runner.logger.error(
                "ProtocolEarlyStopHook stopping at iter=%s because %s stayed at %.6f",
                current_iter,
                self.monitor,
                current_score,
            )
            runner.train_loop.stop_training = True
            return

        if current_iter > self.collapse_after_iter and probe_score < self.collapse_threshold:
            reason = {
                "reason": "collapse_probe",
                "iter": current_iter,
                "monitor": self.collapse_monitor,
                "score": probe_score,
            }
            self._write_stop_reason(runner, reason)
            runner.logger.error(
                "ProtocolEarlyStopHook stopping at iter=%s because %s=%.6e < %.6e",
                current_iter,
                self.collapse_monitor,
                probe_score,
                self.collapse_threshold,
            )
            runner.train_loop.stop_training = True
            return

        if not math.isfinite(current_score) or current_score <= 0.0:
            return

        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            if current_iter >= self.burn_in_iters and current_score > self.start_threshold:
                self.active = True
                self.wait_count = 0
            return

        if current_iter < self.burn_in_iters:
            return

        if not self.active:
            if current_score > self.start_threshold:
                self.active = True
                self.wait_count = 0
                if self.best_score is None or current_score > self.best_score:
                    self.best_score = current_score
            return

        self.wait_count += 1
        if self.wait_count < self.patience:
            return

        reason = {
            "reason": "patience_exhausted",
            "iter": current_iter,
            "monitor": self.monitor,
            "score": current_score,
            "best_score": self.best_score,
            "wait_count": self.wait_count,
        }
        self._write_stop_reason(runner, reason)
        runner.logger.info(
            "ProtocolEarlyStopHook stopping at iter=%s after %s flat monitor records; best=%.6f current=%.6f",
            current_iter,
            self.wait_count,
            self.best_score or 0.0,
            current_score,
        )
        runner.train_loop.stop_training = True


@METRICS.register_module()
class MonitorCocoMetric(CocoMetric):
    default_prefix = "monitor"

    def __init__(self, probe_image_ids: Sequence[int] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probe_image_ids = {int(image_id) for image_id in (probe_image_ids or [])}

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_instances"]
            result["img_id"] = data_sample["img_id"]
            result["bboxes"] = pred["bboxes"].cpu().numpy()
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()
            gt = dict()
            gt["width"] = data_sample["ori_shape"][1]
            gt["height"] = data_sample["ori_shape"][0]
            gt["img_id"] = data_sample["img_id"]
            if self._coco_api is None:
                assert "instances" in data_sample
                gt["anns"] = data_sample["instances"]
            img_id = int(data_sample["img_id"])
            if img_id in self.probe_image_ids:
                scores = pred["scores"].detach().cpu()
                result["probe_max_score"] = float(scores.max().item()) if scores.numel() else 0.0
                result["probe_boxes_ge_0_3"] = int((scores >= 0.3).sum().item()) if scores.numel() else 0
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> dict[str, float]:
        metrics = super().compute_metrics(results)
        _, preds = zip(*results)
        probe_rows = [pred for pred in preds if "probe_max_score" in pred]
        if probe_rows:
            metrics["probe_max_score"] = max(float(row["probe_max_score"]) for row in probe_rows)
            metrics["probe_boxes_ge_0_3"] = float(sum(int(row["probe_boxes_ge_0_3"]) for row in probe_rows))
            metrics["probe_image_count"] = float(len(probe_rows))
        else:
            metrics["probe_max_score"] = 0.0
            metrics["probe_boxes_ge_0_3"] = 0.0
            metrics["probe_image_count"] = 0.0
        return metrics
