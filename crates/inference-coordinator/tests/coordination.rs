use inference_coordinator::Coordinator;
use std::{
    io::Write,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn process(mode: &str, dir: &Path, marker: &Path) -> Process {
    Process(
        Command::new(env!("CARGO_BIN_EXE_lease_probe"))
            .args([mode, dir.to_str().unwrap(), marker.to_str().unwrap()])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}
fn marker(path: &Path, expected: &str) {
    let start = Instant::now();
    while std::fs::read_to_string(path).ok().as_deref() != Some(expected) {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "Process never reached {expected}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn heavy_waits_for_residents_excludes_new_work_and_releases_after_process_death() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("locks");
    let coordinator = Coordinator::new(&root).unwrap();
    let mut resident = process("resident", &root, &dir.path().join("small"));
    marker(&dir.path().join("small"), "resident");
    let mut heavy = process("heavy", &root, &dir.path().join("large"));
    marker(&dir.path().join("large"), "requested");
    assert!(coordinator.heavy_requested().unwrap());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("large")).unwrap(),
        "requested"
    );
    resident.0.stdin.as_mut().unwrap().write_all(b"\n").unwrap();
    assert!(resident.0.wait().unwrap().success());
    marker(&dir.path().join("large"), "exclusive");
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let other = coordinator.clone();
    let waiting = std::thread::spawn(move || other.admit(&flag).map(|_| ()));
    std::thread::sleep(Duration::from_millis(80));
    assert!(!waiting.is_finished());
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(
        waiting.join().unwrap().unwrap_err().kind(),
        std::io::ErrorKind::Interrupted
    );
    heavy.0.kill().unwrap();
    let _ = heavy.0.wait();
    let admitted = coordinator.admit(&AtomicBool::new(false)).unwrap();
    let _resident = coordinator.resident(&admitted).unwrap();
    assert!(!coordinator.heavy_requested().unwrap());
}
#[test]
fn active_work_finishes_before_heavy_intent_and_cancelled_wait_leaks_no_lock() {
    let dir = tempfile::tempdir().unwrap();
    let c = Coordinator::new(dir.path()).unwrap();
    let active = c.admit(&AtomicBool::new(false)).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let other = c.clone();
    let waiting = std::thread::spawn(move || other.request_heavy(&flag).map(|_| ()));
    std::thread::sleep(Duration::from_millis(80));
    assert!(!waiting.is_finished());
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(
        waiting.join().unwrap().unwrap_err().kind(),
        std::io::ErrorKind::Interrupted
    );
    drop(active);
    let _exclusive = c
        .request_heavy(&AtomicBool::new(false))
        .unwrap()
        .reserve(&AtomicBool::new(false))
        .unwrap();
}
#[test]
fn killing_the_parent_does_not_release_a_live_workers_memory_reservation() {
    let dir = tempfile::tempdir().unwrap();
    let coordinator = Coordinator::new(dir.path().join("locks")).unwrap();
    let path = dir.path().join("child");
    struct Release(std::path::PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, b"release");
        }
    }
    let release = Release(path.with_extension("release"));
    let mut parent = process("parent", &dir.path().join("locks"), &path);
    marker(&path, "inherited");
    parent.0.kill().unwrap();
    let _ = parent.0.wait();
    assert!(coordinator.heavy_requested().unwrap());
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let other = coordinator.clone();
    let waiting = std::thread::spawn(move || other.admit(&flag).map(|_| ()));
    std::thread::sleep(Duration::from_millis(80));
    assert!(!waiting.is_finished());
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(
        waiting.join().unwrap().unwrap_err().kind(),
        std::io::ErrorKind::Interrupted
    );
    drop(release);
    let deadline = Instant::now() + Duration::from_secs(3);
    while coordinator.heavy_requested().unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    let _admission = coordinator.admit(&AtomicBool::new(false)).unwrap();
}
