from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise SystemExit(f"{path}:{line_number}: expected '<sha256> <filename>'")
            checksum, filename = parts[0], parts[-1]
            entries.append((checksum, filename))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_dir", type=Path, help="Directory containing downloaded archives")
    parser.add_argument("sha256s", type=Path, help="Checksum file, e.g. metadata/archive_sha256s.txt")
    args = parser.parse_args()

    entries = read_checksums(args.sha256s)
    failures = 0

    for expected, filename in entries:
        path = args.archive_dir / Path(filename).name
        if not path.exists():
            print(f"MISSING {filename}")
            failures += 1
            continue
        observed = file_sha256(path)
        if observed == expected:
            print(f"OK {path.name}")
        else:
            print(f"FAIL {path.name} expected={expected} observed={observed}")
            failures += 1

    if failures:
        print(f"verification_failed: {failures} file(s)")
        return 1
    print(f"verification_ok: {len(entries)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
