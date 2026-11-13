from __future__ import annotations

import time

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class WallClockLimitHook(Hook):

    priority = "LOW"

    def __init__(self, max_seconds: float, check_interval: int = 50) -> None:
        self.max_seconds = float(max_seconds)
        self.check_interval = int(check_interval)
        self._start_time: float | None = None

    def before_train(self, runner) -> None:
        self._start_time = time.time()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if self._start_time is None:
            return
        if (runner.iter + 1) % self.check_interval != 0:
            return
        elapsed = time.time() - self._start_time
        if elapsed >= self.max_seconds:
            runner.logger.warning(
                "WallClockLimitHook stopping training at iter=%s after %.1f seconds",
                runner.iter + 1,
                elapsed,
            )
            runner.train_loop.stop_training = True
