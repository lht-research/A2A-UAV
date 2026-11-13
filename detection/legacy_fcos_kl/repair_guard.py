import json
import math
from pathlib import Path
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class RepairValidationGuard(Hook):
    def __init__(self, monitor='bbox_mAP_50', patience=4, min_delta=0.005, max_empty_epochs=3):
        self.monitor=monitor; self.patience=patience; self.min_delta=min_delta
        self.max_empty_epochs=max_empty_epochs; self.empty=0; self.bad=0; self.best=None
    def after_train_epoch(self, runner):
        output=runner.log_buffer.output or {}
        value=output.get(self.monitor)
        valid=value is not None and math.isfinite(float(value)) and float(value)>0
        state={'epoch':runner.epoch+1,'monitor':self.monitor,'score':value,'valid':valid}
        if not valid:
            self.empty+=1
            if self.empty>=self.max_empty_epochs:
                state.update(status='failed_empty_or_zero_validation',empty_epochs=self.empty)
                if runner.rank==0: (Path(runner.work_dir)/'validation_guard.json').write_text(json.dumps(state,indent=2))
                raise RuntimeError('Repair validation guard: three consecutive empty/zero validations; inspect before retraining.')
        else:
            self.empty=0
            score=float(value)
            if self.best is None or score>self.best+self.min_delta:
                self.best=score; self.bad=0
                if runner.rank==0: runner.save_checkpoint(runner.work_dir,filename_tmpl='best_validation.pth',create_symlink=False)
            else: self.bad+=1
            if self.bad>=self.patience:
                runner._max_epochs=runner.epoch+1
                state['stop_reason']='validation_plateau'
        state.update(status='validating',best=self.best,flat_epochs=self.bad,empty_epochs=self.empty)
        if runner.rank==0: (Path(runner.work_dir)/'validation_guard.json').write_text(json.dumps(state,indent=2))
