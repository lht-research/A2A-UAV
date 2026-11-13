import argparse
import json
from pathlib import Path
import re

from PIL import Image

SIZE = (1333, 800)
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.index.read_text(encoding="utf-8"))["rows"]
    plan = []
    ids = set()
    for split in SPLITS:
        for row in rows[split]:
            sample_id = row["sample_id"]
            if row["split"] != split or not re.fullmatch(r"(?:tr|va|te)_[0-9]{6}", sample_id) or sample_id in ids:
                raise ValueError(f"Invalid or repeated sample ID: {sample_id}")
            ids.add(sample_id)
            image = args.dataset_root / "images" / split / sample_id / "spectrum_tf.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            plan.append((image, args.output_dir / split / f"{sample_id}.png"))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (args.output_dir / split).mkdir()
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    for source, target in plan:
        with Image.open(source) as im:
            im.convert("RGB").resize(SIZE, resample=bilinear).save(target)
    policy = {"width": SIZE[0], "height": SIZE[1], "mode": "RGB", "interpolation": "PIL BILINEAR"}
    (args.output_dir / "cache_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(f"Prepared {len(plan)} spectrograms")


if __name__ == "__main__":
    main()
