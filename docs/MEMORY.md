# Model memory coordination

YouScoop's small models and its large Gemma worker are admitted through a desktop inference coordinator: file locks in `~/Library/Application Support/.sacrinomicon-inference` (the folder name is historical). Other local apps that use the same locks, such as WiZMUTH, can stay open while a large investigation runs: idle models in any of them release their weights before Gemma loads. Browsing and transcript inspection do not need resident models.

Active small-model work holds a shared intent lease. Resident weights hold a separate shared activity lease. A large worker first obtains exclusive intent, allowing active work to finish and preventing new small-model admission. Each application's idle watcher then drops its model and activity lease. The large worker obtains exclusive activity before spawning Python. These leases are OS file locks, so an abandoned file is not a stale ownership record.

The Python child inherits both exclusive locks. An abruptly killed parent cannot release the reservation while its child still owns GPU weights. The worker also watches its parent and exits if orphaned. Cancellation during lock waits is checked every 40 milliseconds. Worker cancellation kills and waits for the child before releasing its reservation.

Chat and inspection show waiting/loading status. YouScoop's setup screen depends on installed model files, so releasing idle weights does not remove the library from view. A failed engine load can be retried from chat.

## Verification

- Three separate-process tests cover active-work admission, cancellation without leaked locks, idle resident release, exclusive admission, crash release and a surviving worker whose parent was killed.
- Two Python lifecycle tests exercise the actual worker protocol with a stub model: a live owner completes; a missing owner exits early. They do not measure model accuracy.
- A native hardware probe loaded Qwen3 4B and BGE weights, then called the same idle-unload function the app's watcher uses. A separate process obtained the heavy reservation only after model release. Resident process RSS fell from 4.5 GB to 40 MB; the handoff took 0.151 seconds. (That probe now lives in the WiZMUTH project.)

The probe does not exercise GUI navigation or a full two-hour ingestion.

## Reproduction

```sh
cargo test --offline --manifest-path crates/inference-coordinator/Cargo.toml
python3 -m unittest runtime/test_worker_lifecycle.py
```
