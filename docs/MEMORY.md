# Model memory coordination

YouScoop and SACRiNOMiCON share a desktop inference coordinator in `~/Library/Application Support/.sacrinomicon-inference`. The two applications can remain open while a large investigation runs. Idle archive and YouScoop models release their weights before Omni or Gemma loads. Browsing, transcript inspection and archive navigation do not require resident models.

Active small-model work holds a shared intent lease. Resident weights hold a separate shared activity lease. A large worker first obtains exclusive intent, allowing active work to finish and preventing new small-model admission. Each application's idle watcher then drops its model and activity lease. The large worker obtains exclusive activity before spawning Python. These leases are OS file locks, so an abandoned file is not a stale ownership record.

The Python child inherits both exclusive locks. An abruptly killed parent cannot release the reservation while its child still owns GPU weights. The worker also watches its parent and exits if orphaned. Cancellation during lock waits is checked every 40 milliseconds. Worker cancellation kills and waits for the child before releasing its reservation. Android uses its own app-local coordinator; desktop ingestion is not a phone requirement.

Chat and inspection show waiting/loading status. YouScoop's setup screen depends on installed model files, so releasing idle weights does not remove the library from view. A failed engine load can be retried from chat.

## Verification

- Three separate-process tests cover active-work admission, cancellation without leaked locks, idle resident release, exclusive admission, crash release and a surviving worker whose parent was killed.
- Two Python lifecycle tests exercise the actual worker protocol with a stub model: a live owner completes; a missing owner exits early. They do not measure model accuracy.
- A native hardware probe loaded the actual archive Qwen3 4B and BGE weights, then called the same idle-unload function as the archive watcher. A separate process obtained the heavy reservation only after model release. Resident process RSS fell from 4,503,846,912 to 39,534,592 bytes; the measured handoff took 0.151 seconds. See `experiments/model-trial/results/archive-memory-handoff.json`.
- Both native applications compile with the coordinator. The Android ARM64 debug package also builds.

The hardware probe does not exercise microphone permissions, GUI navigation, voice interruption or a full two-hour ingestion. Those remain separate verification gates.

## Reproduction

```sh
cargo test --offline --manifest-path crates/inference-coordinator/Cargo.toml
python3 -m unittest runtime/test_worker_lifecycle.py
cargo build --offline --manifest-path apps/sacrinomicon/src-tauri/Cargo.toml --features diagnostics --example memory_handoff
python3 experiments/model-trial/memory_handoff.py \
  --archive-probe apps/sacrinomicon/src-tauri/target/debug/examples/memory_handoff \
  --heavy-probe crates/inference-coordinator/target/debug/lease_probe \
  --chat "$HOME/Library/Application Support/com.canpixel.youscoop/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  --embed "$HOME/Library/Application Support/com.canpixel.youscoop/models/bge-small-en-v1.5-q8_0.gguf" \
  --output experiments/model-trial/results/archive-memory-handoff.json
```

The hardware probe needs local Metal access. Model paths may differ on another machine.
