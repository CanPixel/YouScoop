"""Verified desktop runtime/model setup. JSON request and events, no model inference.

Rust creates the pinned venv before invoking this module. Model snapshots are
immutable, published only after verification, and independent of the checkout.
"""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

PROTOCOL = sys.stdout


def emit(kind, **payload):
    PROTOCOL.write(json.dumps({"kind": kind, **payload}) + "\n")
    PROTOCOL.flush()


def watch_parent(owner):
    def watch():
        while True:
            if os.getppid() != owner:
                os._exit(0)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as output:
        json.dump(data, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def runtime_probe():
    import av
    import mlx.core as mx
    import mlx_lm
    import librosa
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    if not torch.backends.mps.is_available():
        raise ValueError("This desktop runtime requires Apple Silicon Metal support")
    # Importing is enough to check dependency compatibility; never load weights.
    versions = {name: importlib.metadata.version(name) for name in
                ("av", "mlx", "mlx-lm", "librosa", "torch", "transformers")}
    expected = {"av": "19.0.0", "mlx": "0.32.3", "mlx-lm": "0.31.3", "librosa": "0.11.0", "torch": "2.14.0", "transformers": "5.17.0"}
    if versions != expected:
        raise ValueError("The configured Python environment does not match the tested desktop dependency versions")
    return {"pythonVersion": sys.version.split()[0], "python": sys.executable,
            "mpsAvailable": True,
            "versions": versions}


def validate_catalog(spec):
    names = set()
    for f in spec["files"]:
        name = f["name"]
        if not name or Path(name).name != name or name in names or name in (".", ".."):
            raise ValueError("Invalid model catalog filename")
        names.add(name)
        if f["bytes"] < 1 or len(f["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in f["sha256"]):
            raise ValueError("Invalid model catalog checksum")
    if not names or spec["bytes"] != sum(f["bytes"] for f in spec["files"]):
        raise ValueError("Invalid model catalog size")


def verify_snapshot(root, spec, progress=lambda *_: None):
    root = Path(root)
    verified = 0
    for f in spec["files"]:
        path = root / f["name"]
        if not path.is_file() or path.stat().st_size != f["bytes"]:
            raise ValueError(f"Missing or incomplete model file: {f['name']}")
        digest = hashlib.sha256()
        received = 0
        last = 0.0
        with path.open("rb") as content:
            while chunk := content.read(8 * 1024 * 1024):
                digest.update(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last > 0.5:
                    progress(verified + received, spec["bytes"], f["name"])
                    last = now
        if digest.hexdigest() != f["sha256"]:
            raise ValueError(f"Model checksum mismatch: {f['name']}")
        verified += received
        progress(verified, spec["bytes"], f["name"])
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != spec["modelType"]:
        raise ValueError("Model architecture does not match the selected snapshot")
    return verified


def install_snapshot(source, destination, spec, progress=lambda *_: None):
    """Leave the active destination untouched until an entire snapshot verifies."""
    validate_catalog(spec)
    source, destination = Path(source), Path(destination)
    if destination.exists():
        verify_snapshot(destination, spec, progress)
        return destination
    staging = destination.with_name(destination.name + ".incoming")
    staging.mkdir(parents=True, exist_ok=True)
    for f in spec["files"]:
        original = source / f["name"]
        target = staging / f["name"]
        if not original.is_file() or original.stat().st_size != f["bytes"]:
            raise ValueError(f"Cached model file is incomplete: {f['name']}")
        # Snapshot symlinks point into HF's cache. Resolve them before retaining
        # an independent inode reference; small mutable metadata gets a copy.
        if target.exists() or target.is_symlink():
            target.unlink()
        if original.suffix == ".safetensors":
            try:
                os.link(original.resolve(), target)
            except OSError:
                # Another filesystem cannot share an inode. Do not overwrite a
                # verified active snapshot; this copy remains in staging.
                shutil.copyfile(original, target)
        else:
            shutil.copyfile(original, target)
    verify_snapshot(staging, spec, progress)
    atomic_json(staging / "installation.json", {"repository": spec["repository"],
                "revision": spec["revision"], "files": spec["files"], "verifiedAt": int(time.time())})
    staging.rename(destination)
    return destination


def setup_model(request):
    catalog = request["catalog"]
    model = request["model"]
    if model not in ("omni", "gemma"):
        raise ValueError("Unknown desktop model")
    spec = catalog["models"][model]
    validate_catalog(spec)
    root = Path(request["root"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "models" / f"{model}-{spec['revision']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = request.get("source")
    if not destination.exists() and not source:
        from huggingface_hub import snapshot_download
        emit("status", detail=f"Downloading {model} snapshot ({spec['bytes'] / 1e9:.2f} GB)")
        source = snapshot_download(repo_id=spec["repository"], revision=spec["revision"],
                                   cache_dir=str(root / "hub"),
                                   allow_patterns=[f["name"] for f in spec["files"]],
                                   max_workers=3)
    emit("status", detail=f"Verifying and preserving {model} files")
    destination = install_snapshot(source or destination, destination, spec,
                    lambda received, total, name: emit("status", detail=f"Verifying {model}: {name}", received=received, total=total))
    emit("result", modelPath=str(destination), bytes=spec["bytes"], revision=spec["revision"])


def main():
    try:
        request = json.loads(sys.stdin.readline())
        watch_parent(int(request.get("ownerPid", os.getppid())))
        with contextlib.redirect_stdout(sys.stderr):
            if request["task"] == "probe":
                emit("status", detail="Checking the installed inference runtime")
                emit("result", probe=runtime_probe())
            elif request["task"] == "model":
                setup_model(request)
            else:
                raise ValueError("Unknown setup task")
    except Exception as error:
        emit("error", message=str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
