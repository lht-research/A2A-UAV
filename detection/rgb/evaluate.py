import argparse
import os
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    os.environ["A2AUAV_BENCHMARK_ROOT"] = str(root)
    cfg = Config.fromfile(str(args.config.resolve()))
    annotations = root / "rgb" / "data" / "coco" / "annotations" / f"instances_{args.split}_rgb_single.json"
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    cfg.test_dataloader.dataset.ann_file = str(annotations)
    cfg.test_evaluator.ann_file = str(annotations)
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.work_dir = str(args.work_dir.resolve())
    Runner.from_cfg(cfg).test()


if __name__ == "__main__":
    main()
