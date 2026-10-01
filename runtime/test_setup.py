"""Preservation, failure and process-lifetime checks for the actual installer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from runtime import setup


class SnapshotInstallation(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        source = root / "cache" / "snapshot"
        blobs = root / "cache" / "blobs"
        source.mkdir(parents=True)
        blobs.mkdir()
        content = {"config.json": b'{"model_type":"fixture"}', "model.safetensors": b"immutable test weight bytes"}
        records = []
        for name, data in content.items():
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            (source / name).symlink_to(Path("../blobs") / digest)
            records.append({"name": name, "sha256": digest, "bytes": len(data)})
        spec = {"repository": "fixture/test", "revision": "a" * 40, "modelType": "fixture", "files": records, "bytes": sum(r["bytes"] for r in records)}
        return source, spec

    def test_cached_symlinks_become_independent_files_and_mutable_metadata_is_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.fixture(directory)
            destination = Path(directory) / "installed"
            setup.install_snapshot(source, destination, spec)
            self.assertFalse((destination / "model.safetensors").is_symlink())
            self.assertEqual((destination / "model.safetensors").stat().st_ino, (source / "model.safetensors").stat().st_ino)
            self.assertNotEqual((destination / "config.json").stat().st_ino, (source / "config.json").stat().st_ino)
            shutil.rmtree(Path(directory) / "cache")
            self.assertEqual(setup.verify_snapshot(destination, spec), spec["bytes"])

    def test_same_size_corruption_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.fixture(directory)
            file = source / "model.safetensors"
            file.write_bytes(b"x" * file.stat().st_size)
            destination = Path(directory) / "installed"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                setup.install_snapshot(source, destination, spec)
            self.assertFalse(destination.exists())

    def test_interrupted_verification_can_retry_and_never_publishes_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.fixture(directory)
            destination = Path(directory) / "installed"
            def interrupted(*args):
                raise InterruptedError("Cancelled")
            with self.assertRaises(InterruptedError):
                setup.install_snapshot(source, destination, spec, interrupted)
            self.assertFalse(destination.exists())
            setup.install_snapshot(source, destination, spec)
            self.assertEqual(setup.verify_snapshot(destination, spec), spec["bytes"])

    def test_existing_verified_snapshot_survives_an_invalid_replacement_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.fixture(directory)
            destination = Path(directory) / "installed"
            setup.install_snapshot(source, destination, spec)
            setup.install_snapshot(Path(directory) / "missing", destination, spec)
            self.assertEqual(setup.verify_snapshot(destination, spec), spec["bytes"])

    def test_unsafe_catalog_cannot_write_outside_the_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.fixture(directory)
            spec["files"][0]["name"] = "../escape"
            with self.assertRaises(ValueError):
                setup.install_snapshot(source, Path(directory) / "installed", spec)
            self.assertFalse((Path(directory) / "escape").exists())


class SetupLifecycle(unittest.TestCase):
    def run_setup(self, owner, body):
        code = "import time; from runtime import setup; setup.setup_model=" + body + "; setup.main()"
        return subprocess.run([sys.executable, "-c", code], input=json.dumps({"task": "model", "ownerPid": owner}) + "\n", text=True, capture_output=True, timeout=3)

    def test_missing_owner_stops_a_model_installer_that_would_otherwise_continue(self):
        result = self.run_setup(0, "lambda request: time.sleep(30)")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_live_owner_receives_the_result(self):
        result = self.run_setup(os.getpid(), "lambda request: setup.emit('result', modelPath='installed')")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["modelPath"], "installed")


if __name__ == "__main__":
    unittest.main()
