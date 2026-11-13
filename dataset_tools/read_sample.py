import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Extracted UAV_full_dataset_release directory")
    parser.add_argument("sample_id", help="Sample identifier, for example tr_000001")
    args = parser.parse_args()
    match = re.fullmatch(r"(tr|va|te)_[0-9]{6}", args.sample_id)
    if match is None:
        parser.error("sample_id must contain a split prefix and six digits")

    split = {"tr": "train", "va": "val", "te": "test"}[match[1]]
    sample_dir = args.root / "images" / split / args.sample_id
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["sample_id"] != args.sample_id or metadata["split"] != split:
        raise ValueError(f"Unexpected sample identity in {sample_dir / 'metadata.json'}")

    modalities = metadata["modalities"]
    radar = metadata.get("radar_pointcloud") or {}
    print(json.dumps({
        "sample_id": metadata["sample_id"],
        "split": metadata["split"],
        "classification_label": metadata["classification_label"],
        "images": {name: str(sample_dir / modalities[name])
                   for name in ("rgb", "panorama_lr", "thermal", "spectrum_tf")},
        "radar_frames": [str(sample_dir / frame["file"]) for frame in radar.get("frames", [])],
        "annotations": metadata["annotations"],
    }, indent=2))


if __name__ == "__main__":
    main()
