"""Exercise the archive's actual model-unload function across processes."""
import argparse
import json
import pathlib
import subprocess
import tempfile
import time


def wait_marker(path, expected, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if path.exists() and path.read_text() == expected:
            return
        if process.poll() is not None:
            raise RuntimeError(f"Probe exited before {expected}: {process.returncode}")
        time.sleep(0.04)
    raise TimeoutError(f"Probe did not reach {expected}")


def rss(pid):
    return int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(pid)]).strip()) * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-probe", required=True)
    parser.add_argument("--heavy-probe", required=True)
    parser.add_argument("--chat", required=True)
    parser.add_argument("--embed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    children = []
    with tempfile.TemporaryDirectory(prefix="sacri-memory-") as temp:
        root = pathlib.Path(temp)
        with output.with_suffix(".log").open("w") as log:
            try:
                start = time.monotonic()
                small = subprocess.Popen([args.archive_probe, str(root / "locks"), args.chat, args.embed, str(root / "archive")], stdin=subprocess.PIPE, stdout=log, stderr=log)
                children.append(small)
                wait_marker(root / "archive", "loaded", small)
                loaded = rss(small.pid)
                loaded_seconds = time.monotonic() - start
                heavy = subprocess.Popen([args.heavy_probe, "heavy", str(root / "locks"), str(root / "heavy")], stdin=subprocess.PIPE, stdout=log, stderr=log)
                children.append(heavy)
                handoff_start = time.monotonic()
                wait_marker(root / "archive", "released", small)
                wait_marker(root / "heavy", "exclusive", heavy)
                handoff_seconds = time.monotonic() - handoff_start
                time.sleep(0.2)
                released = rss(small.pid)
                result = {
                    "model": pathlib.Path(args.chat).name,
                    "loadSeconds": loaded_seconds,
                    "handoffSeconds": handoff_seconds,
                    "residentRssBytes": loaded,
                    "releasedRssBytes": released,
                    "rssReductionBytes": loaded - released,
                    "heavyReservationGrantedAfterModelDrop": True,
                    "scope": "Actual archive unload function and Qwen/BGE weights, separate processes; no GUI or voice lifecycle exercised.",
                }
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2))
            finally:
                for child in reversed(children):
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=10)


if __name__ == "__main__":
    main()
