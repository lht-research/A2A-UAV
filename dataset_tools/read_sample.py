import argparse
import json
import re
from pathlib import Path


def load_manifest_sample(root, split, sample_id, sample_dir):
    """Read the index used by previews whose archive has no per-sample metadata."""
    path = root / "manifests" / f"full_{split}.jsonl"
    sample_rel = sample_dir.relative_to(root)
    filenames = {
        "rgb": "rgb.png",
        "panorama_lr": "panorama_lr.png",
        "thermal": "thermal.png",
        "spectrum_tf": "spectrum_tf.png",
    }
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sample_id"] != sample_id:
                continue
            if row["split"] != split or Path(row["sample_dir"]) != sample_rel:
                raise ValueError(f"Unexpected sample identity in {path}")
            return {
                "sample_id": sample_id,
                "split": split,
                "classification_label": row["classification_label"],
                "modalities": {
                    name: str(Path(row["files"][filename]).relative_to(sample_rel))
                    for name, filename in filenames.items()
                },
                "annotations": row["annotations"],
                "radar_pointcloud": {"frames": [
                    {**frame, "file": str(Path(frame["file"]).relative_to(sample_rel))}
                    for frame in row.get("radar_frames", [])
                ]},
            }
    raise ValueError(f"Sample {sample_id} is not listed in {path}")


def load_sample(root, sample_id):
    match = re.fullmatch(r"(tr|va|te)_[0-9]{6}", sample_id)
    if match is None:
        raise ValueError("sample_id must contain a split prefix and six digits")

    split = {"tr": "train", "va": "val", "te": "test"}[match[1]]
    root = Path(root)
    sample_dir = root / "images" / split / sample_id
    metadata_path = sample_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        metadata = load_manifest_sample(root, split, sample_id, sample_dir)
    if metadata["sample_id"] != sample_id or metadata["split"] != split:
        raise ValueError(f"Unexpected sample identity in {metadata_path}")
    return sample_dir, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Extracted A2A-UAV dataset or preview directory")
    parser.add_argument("sample_id", help="Sample identifier, for example tr_000505")
    args = parser.parse_args()
    try:
        sample_dir, metadata = load_sample(args.root, args.sample_id)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
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
