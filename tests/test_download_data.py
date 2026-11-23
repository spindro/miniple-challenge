"""Small offline fixtures for the optional prepared corpus download."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as downloader


def record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / "data/ultrafineweb_3b"
        self.cache = self.target.parent / ".ultrafineweb_3b.download"
        self.files = {name: (name + "\n").encode() * 3 for name in downloader.FILES}
        self.manifest = {name: record(data) for name, data in self.files.items()}
        path = self.root / "baselines/3b/data_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.manifest))
        self.calls = []
        self.prepare_archive()

    def prepare_archive(self, *, changed=None, extra=None, omit=None):
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, original in sorted(self.files.items()):
                if name == omit:
                    continue
                data = changed.get(name, original) if changed else original
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if extra is not None:
                member, data = extra
                archive.addfile(member, io.BytesIO(data))
        blob = out.getvalue()
        # Deliberately split inside tar headers and member payloads.
        boundary = len(blob) // 3 + 7
        self.pieces = dict(zip(downloader.PART_NAMES, (blob[:boundary], blob[boundary:2 * boundary], blob[2 * boundary:])))
        self.config = {
            "repository": "example/course", "release": "data-3b-v1",
            "parts": [{"name": name, **record(data)} for name, data in self.pieces.items()],
        }
        self.write_config()

    def write_config(self):
        (self.root / "data_distribution.json").write_text(json.dumps(self.config))

    def opener(self, request, timeout):
        self.assertEqual(timeout, 120)
        self.assertTrue(request.full_url.startswith("https://github.com/example/course/releases/download/data-3b-v1/"))
        name = request.full_url.rsplit("/", 1)[-1]
        self.calls.append(name)
        if name not in self.pieces:
            raise URLError("piece unavailable")
        return io.BytesIO(self.pieces[name])

    def run_download(self, **kwargs):
        return downloader.download_data(self.root, opener=self.opener, **kwargs)

    def assert_no_staging(self):
        self.assertEqual(list(self.target.parent.glob(".ultrafineweb_3b.extract-*")), [])
        self.assertEqual(list(self.cache.glob("*.part")), [])

    def test_success_and_idempotence(self):
        self.assertEqual(self.run_download(), self.target)
        self.assertEqual(set(path.name for path in self.target.iterdir()), downloader.FILES)
        for name, content in self.files.items():
            self.assertEqual((self.target / name).read_bytes(), content)
        self.assertEqual(self.calls, list(downloader.PART_NAMES))
        self.assertFalse(self.cache.exists())
        self.calls.clear()
        self.run_download()
        self.assertEqual(self.calls, [])
        self.assert_no_staging()

    def test_custom_destination(self):
        custom = self.root / "custom/corpus"
        self.assertEqual(self.run_download(data_dir=custom), custom)
        self.assertTrue((custom / "metadata.json").is_file())
        self.assertFalse(self.target.exists())

    def test_missing_piece_retry_reuses_verified_cache(self):
        name = downloader.PART_NAMES[-1]
        saved = self.pieces.pop(name)
        with self.assertRaises(URLError):
            self.run_download()
        self.assertFalse(self.target.exists())
        for earlier in downloader.PART_NAMES[:2]:
            self.assertEqual((self.cache / earlier).read_bytes(), self.pieces[earlier])
        self.assert_no_staging()
        self.pieces[name] = saved
        self.calls.clear()
        self.run_download()
        self.assertEqual(self.calls, [name])

    def test_corrupt_download_is_not_cached_or_installed(self):
        name = downloader.PART_NAMES[1]
        original = self.pieces[name]
        self.pieces[name] = bytes([original[0] ^ 1]) + original[1:]
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_download()
        self.assertFalse(self.target.exists())
        self.assertTrue((self.cache / downloader.PART_NAMES[0]).is_file())
        self.assertFalse((self.cache / name).exists())
        self.assert_no_staging()

    def test_oversized_download_is_rejected(self):
        self.pieces[downloader.PART_NAMES[0]] += b"extra"
        with self.assertRaisesRegex(ValueError, "exceeds expected size"):
            self.run_download()
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_corrupt_cached_piece_fails_before_network(self):
        self.cache.mkdir(parents=True)
        (self.cache / downloader.PART_NAMES[0]).write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "Size mismatch"):
            self.run_download()
        self.assertEqual(self.calls, [])

    def test_insufficient_space_fails_before_network(self):
        with patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, "Insufficient free space"):
                self.run_download()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.target.exists())

    def test_extracted_file_hash_is_checked(self):
        name = "stage1_l2.bin"
        data = self.files[name]
        self.prepare_archive(changed={name: b"x" * len(data)})
        with self.assertRaisesRegex(ValueError, "Extracted size or SHA256 mismatch"):
            self.run_download()
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_missing_archive_file_is_rejected(self):
        self.prepare_archive(omit="metadata.json")
        with self.assertRaisesRegex(ValueError, "missing files"):
            self.run_download()
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_traversal_duplicate_and_symlink_are_rejected(self):
        cases = [tarfile.TarInfo("../escape"), tarfile.TarInfo("metadata.json"), tarfile.TarInfo("link")]
        cases[-1].type = tarfile.SYMTYPE
        cases[-1].linkname = "../escape"
        for member in cases:
            with self.subTest(name=member.name):
                self.prepare_archive(extra=(member, b""))
                for path in self.cache.glob("*"):
                    path.unlink()
                with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                    self.run_download()
                self.assertFalse(self.target.exists())
                self.assertFalse((self.root / "escape").exists())
                self.assert_no_staging()

    def test_existing_invalid_directory_is_preserved(self):
        self.target.mkdir(parents=True)
        sentinel = self.target / "my-data.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "Missing or unsafe file"):
            self.run_download()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.target.iterdir()), [sentinel])
        self.assertEqual(self.calls, [])

    def test_existing_destination_symlink_is_rejected(self):
        other = self.root / "other"
        other.mkdir()
        self.target.parent.mkdir()
        self.target.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "safe directory"):
            self.run_download()
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(self.calls, [])

    def test_destination_created_during_download_is_preserved(self):
        original = self.opener
        def racing_opener(request, timeout):
            if not self.target.exists():
                self.target.mkdir()
                (self.target / "sentinel").write_text("another process")
            return original(request, timeout)
        with self.assertRaisesRegex(ValueError, "Missing or unsafe file"):
            downloader.download_data(self.root, opener=racing_opener)
        self.assertEqual([p.name for p in self.target.iterdir()], ["sentinel"])
        self.assert_no_staging()

    def test_install_failure_rolls_back_only_new_links(self):
        original = downloader.os.link
        def fail_metadata(source, destination):
            if Path(destination) == self.target / "metadata.json":
                raise OSError("fixture installation failure")
            return original(source, destination)
        with patch.object(downloader.os, "link", side_effect=fail_metadata):
            with self.assertRaisesRegex(OSError, "fixture installation failure"):
                self.run_download()
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_invalid_config_is_rejected_before_network(self):
        original = json.dumps(self.config)
        cases = [
            lambda c: c["parts"].pop(),
            lambda c: c["parts"][0].update(name="../escape"),
            lambda c: c["parts"][0].update(bytes=True),
            lambda c: c["parts"][0].update(bytes=downloader.MAX_PART_BYTES + 1),
            lambda c: c.update(repository="../example"),
            lambda c: c["parts"][0].update(sha256="invalid"),
        ]
        for alter in cases:
            self.config = json.loads(original)
            alter(self.config)
            self.write_config()
            with self.assertRaises(ValueError):
                self.run_download()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
