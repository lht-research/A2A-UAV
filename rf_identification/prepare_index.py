import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SPLITS = {"train": "tr", "val": "va", "test": "te"}


def spectrum_checksums(path):
    checksums = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.rstrip().endswith("/spectrum_tf.png"):
                continue
            digest, name = line.rstrip().split("  ", 1)
            if name in checksums or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Invalid checksum entry for {name}")
            checksums[name] = digest
    return checksums


def build_index(root):
    index = json.loads((root / "metadata/classification_index_trainable.json").read_text(encoding="utf-8"))
    checksums = spectrum_checksums(root / "SHA256SUMS")
    classes = index["class_to_idx"]
    rows = index["rows"]
    if set(rows) != set(SPLITS) or len(classes) != 15:
        raise ValueError("Expected three splits and 15 classes")
    counts = {split: len(rows[split]) for split in SPLITS}
    if counts != index["counts"]:
        raise ValueError("Classification index counts do not match its rows")

    sample_ids = set()
    window_records = {}
    recording_splits = {}
    training_recordings = defaultdict(set)
    for split, prefix in SPLITS.items():
        for row in rows[split]:
            sample_id = row["sample_id"]
            path = f"images/{split}/{sample_id}/spectrum_tf.png"
            if (not re.fullmatch(prefix + r"_[0-9]{6}", sample_id)
                    or sample_id in sample_ids or row["split"] != split
                    or row["path"] != path or row["label"] not in classes
                    or not row["spectrum_readable"]):
                raise ValueError(f"Invalid classification entry for {sample_id}")
            sample_ids.add(sample_id)
            key = checksums[path]
            recording = row["canonical_recording_key"]
            if recording_splits.setdefault(recording, split) != split:
                raise ValueError(f"Recording crosses splits: {recording}")
            if window_records.setdefault(key, (split, recording)) != (split, recording):
                raise ValueError(f"Identical spectrograms cross recording groups: {sample_id}")
            if split == "train":
                training_recordings[row["label"]].add(recording)
            row["iq_key"] = key
    if any(len(training_recordings[label]) < 2 for label in classes):
        raise ValueError("Every class needs at least two distinct training recordings")
    index["audit"] = {"cohort": "downloaded", "cap_per_recording": 0,
                      "iq_grouping": "spectrogram_sha256"}
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build_index(args.dataset_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, separators=(",", ":"))
        stream.write("\n")
    windows = {row["iq_key"] for rows in index["rows"].values() for row in rows}
    print(f"Wrote {sum(index['counts'].values()):,} samples and {len(windows):,} spectrogram groups")


if __name__ == "__main__":
    main()
