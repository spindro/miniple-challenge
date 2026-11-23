#!/usr/bin/env python3
"""Download and verify the optional prepared 3B corpus (standard library only)."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "stage1_l2.bin", "decay_l2.bin", "decay_l3_qa.bin", "decay_l3_style.bin",
    "val_l2.bin", "tokenizer.json", "metadata.json",
}
PART_NAMES = tuple(f"miniple-data-3b.tar.part{i:02d}" for i in range(1, 4))
MAX_PART_BYTES = 2_010_000_000
BLOCK = 1024 * 1024


def validate_record(record, label):
    if not isinstance(record, dict):
        raise ValueError(f"Invalid metadata for {label}")
    size = record.get("bytes")
    digest = record.get("sha256")
    if type(size) is not int or size <= 0:
        raise ValueError(f"Invalid byte count for {label}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid SHA256 for {label}")


def load_metadata(root):
    config = json.loads((root / "data_distribution.json").read_text())
    manifest = json.loads((root / "baselines/3b/data_manifest.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("Invalid data_distribution.json")
    repo = config.get("repository", "")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("repository must be a GitHub owner/repository")
    if any(component in (".", "..") for component in repo.split("/")):
        raise ValueError("Invalid repository")
    release = config.get("release")
    if not isinstance(release, str) or not release or any(ord(c) < 32 for c in release):
        raise ValueError("Invalid release tag")
    parts = config.get("parts")
    if not isinstance(parts, list) or len(parts) != 3:
        raise ValueError("Expected exactly three data pieces")
    for part, expected_name in zip(parts, PART_NAMES):
        validate_record(part, expected_name)
        if part.get("name") != expected_name or part["bytes"] > MAX_PART_BYTES:
            raise ValueError(f"Invalid piece name or size: expected {expected_name}")
    if not isinstance(manifest, dict) or set(manifest) != FILES:
        raise ValueError("Canonical data manifest must contain exactly the seven corpus files")
    for name, record in manifest.items():
        validate_record(record, name)
    return config, manifest


def verify_file(path, record):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or unsafe file: {path}")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def verify_existing(target, manifest):
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"Existing destination is not a safe directory: {target}")
    for name, record in manifest.items():
        verify_file(target / name, record)


def download_part(url, destination, record, opener):
    """Promote only a fully verified piece; never overwrite a cached file."""
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "miniple-data-downloader"})
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            count = 0
            next_progress = 128 * BLOCK
            with opener(request, timeout=120) as response:
                while True:
                    block = response.read(BLOCK)
                    if not block:
                        break
                    count += len(block)
                    if count > record["bytes"]:
                        raise ValueError(f"Download exceeds expected size: {destination.name}")
                    out.write(block)
                    digest.update(block)
                    if count >= next_progress:
                        print(f"  {destination.name}: {100 * count / record['bytes']:.1f}% "
                              f"({count / 1e9:.2f} / {record['bytes'] / 1e9:.2f} GB)", flush=True)
                        next_progress += 128 * BLOCK
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Downloaded size or SHA256 mismatch: {destination.name}")
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify_file(destination, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class PieceStream(io.RawIOBase):
    """A bounded reader across verified pieces, without assembling another archive."""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("PieceStream requires a bounded read")
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            block = self.current.read(size - len(result))
            if block:
                result.extend(block)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_verified(paths, staging, manifest):
    seen = set()
    with PieceStream(paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            name = member.name
            if (name not in manifest or name in seen
                    or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.pax_headers):
                raise ValueError(f"Unexpected, duplicate, or unsafe archive member: {name!r}")
            record = manifest[name]
            if member.size != record["bytes"]:
                raise ValueError(f"Archive file size mismatch: {name}")
            seen.add(name)
            digest = hashlib.sha256()
            count = 0
            with archive.extractfile(member) as source, (staging / name).open("xb") as out:
                while True:
                    block = source.read(min(BLOCK, record["bytes"] - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > record["bytes"]:
                        raise ValueError(f"Archive file exceeds expected size: {name}")
                    out.write(block)
                    digest.update(block)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Extracted size or SHA256 mismatch: {name}")
    if seen != set(manifest):
        raise ValueError(f"Archive is missing files: {', '.join(sorted(set(manifest) - seen))}")


def install_verified(staging, target, manifest):
    """Reserve a new directory, then publish verified files, with metadata last."""
    try:
        target.mkdir()
    except FileExistsError:
        verify_existing(target, manifest)
        return
    installed = []
    try:
        for name in sorted(set(manifest) - {"metadata.json"}) + ["metadata.json"]:
            os.link(staging / name, target / name)
            installed.append(name)
    except BaseException:
        # Roll back only files we created. Never remove an existing corpus or
        # anything another process may have placed in the reserved directory.
        for name in reversed(installed):
            path = target / name
            if path.exists() and not path.is_symlink() and path.samefile(staging / name):
                path.unlink()
        try:
            target.rmdir()
        except OSError:
            pass
        raise


def download_data(root=ROOT, data_dir=None, *, opener=None):
    root = Path(root)
    target = Path(data_dir) if data_dir is not None else root / "data/ultrafineweb_3b"
    config, manifest = load_metadata(root)
    if target.exists() or target.is_symlink():
        verify_existing(target, manifest)
        print(f"Already verified: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = target.parent / f".{target.name}.download"
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError(f"Unsafe download cache: {cache}")
    cache.mkdir(mode=0o700, exist_ok=True)
    missing = []
    for part in config["parts"]:
        path = cache / part["name"]
        if path.exists() or path.is_symlink():
            verify_file(path, part)
        else:
            missing.append(part)
    required = sum(record["bytes"] for record in manifest.values()) + sum(p["bytes"] for p in missing)
    if shutil.disk_usage(target.parent).free < required + 16 * BLOCK:
        raise OSError(f"Insufficient free space: need at least {(required + 16 * BLOCK) / 1e9:.2f} GB")
    opener = urlopen if opener is None else opener
    base = f"https://github.com/{config['repository']}/releases/download/{quote(config['release'], safe='')}"
    for part in missing:
        print(f"Downloading {part['name']} ({part['bytes'] / 1e9:.2f} GB)", flush=True)
        download_part(f"{base}/{part['name']}", cache / part["name"], part, opener)
    print("Verifying and installing the seven corpus files...", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.extract-", dir=target.parent) as directory:
        staging = Path(directory)
        extract_verified([cache / name for name in PART_NAMES], staging, manifest)
        install_verified(staging, target, manifest)
    for name in PART_NAMES:
        (cache / name).unlink(missing_ok=True)
    try:
        cache.rmdir()
    except OSError:
        pass
    print(f"Verified corpus ready: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Destination (default: repository/data/ultrafineweb_3b)")
    args = parser.parse_args()
    try:
        download_data(data_dir=args.data_dir)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Data download failed: {error}\nExisting corpus files were not overwritten.\n")


if __name__ == "__main__":
    main()
