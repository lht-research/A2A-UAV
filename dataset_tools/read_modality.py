"""Read any of the five modalities in an extracted A2A-UAV sample."""

import argparse
import csv
import json
from pathlib import Path

from .read_sample import load_sample

IMAGE_KEYS = {
    "rgb": "rgb",
    "panorama": "panorama_lr",
    "thermal": "thermal",
    "rf": "spectrum_tf",
}
RADAR_FIELDS = ("x", "y", "z", "intensity", "velocity")


def read_image(sample_dir, metadata, modality):
    """Decode a released PNG as a Pillow image in its stored mode."""
    from PIL import Image

    path = sample_dir / metadata["modalities"][IMAGE_KEYS[modality]]
    with Image.open(path, formats=["PNG"]) as image:
        image.load()
        return image.copy()


def read_radar(sample_dir, metadata):
    """Return a list of radar frames with their time offsets and point rows."""
    frames = (metadata.get("radar_pointcloud") or {}).get("frames", [])
    results = []
    for frame in frames:
        path = sample_dir / frame["file"]
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.reader(stream)
            if tuple(next(rows, [])) != RADAR_FIELDS:
                raise ValueError(f"Unexpected radar CSV header in {path}")
            points = []
            for row in rows:
                if not row:
                    continue
                if len(row) != len(RADAR_FIELDS):
                    raise ValueError(f"Expected five radar values on line {rows.line_num} in {path}")
                try:
                    points.append([float(value) for value in row])
                except ValueError as exc:
                    raise ValueError(f"Invalid radar value on line {rows.line_num} in {path}") from exc
        if len(points) != frame["point_count"]:
            raise ValueError(f"Radar point count differs from metadata in {path}")
        results.append({
            "frame_id": frame["frame_id"],
            "file": frame["file"],
            "dt_ns": frame["dt_ns"],
            "point_count": len(points),
            "points": points,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted A2A-UAV dataset or preview directory")
    parser.add_argument("sample_id", help="Sample ID such as tr_000505")
    parser.add_argument("modality", choices=[*IMAGE_KEYS, "radar"])
    args = parser.parse_args()
    try:
        sample_dir, metadata = load_sample(args.root, args.sample_id)
        result = {
            "sample_id": args.sample_id,
            "split": metadata["split"],
            "modality": args.modality,
        }
        if args.modality == "radar":
            result.update(fields=RADAR_FIELDS, frames=read_radar(sample_dir, metadata))
        else:
            with read_image(sample_dir, metadata, args.modality) as image:
                result.update(file=metadata["modalities"][IMAGE_KEYS[args.modality]],
                              width=image.width, height=image.height, mode=image.mode)
        print(json.dumps(result, indent=2, allow_nan=False))
    except ModuleNotFoundError as exc:
        if exc.name != "PIL":
            raise
        parser.error("Image readers require Pillow. Install it with python -m pip install Pillow.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
