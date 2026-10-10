"""Native worker lifetime checks without allocating model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class WorkerLifecycle(unittest.TestCase):
    def run_worker(self, owner, body):
        code = "import sys,time; sys.path.insert(0," + repr(str(Path(__file__).parent)) + "); import worker; worker.gemma=" + body + "; worker.main()"
        return subprocess.run([sys.executable, "-c", code], input=json.dumps({"task": "gemma", "ownerPid": owner}) + "\n", text=True, capture_output=True, timeout=3)

    def test_dead_owner_stops_a_worker_that_would_otherwise_continue(self):
        result = self.run_worker(0, "lambda request: time.sleep(30)")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_live_owner_can_receive_the_result(self):
        result = self.run_worker(os.getpid(), "lambda request: worker.emit('result',answer='completed')")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["answer"], "completed")


if __name__ == "__main__":
    unittest.main()
