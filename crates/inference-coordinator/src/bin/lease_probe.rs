//! Process fixture for coordination tests; allocates no models.
use inference_coordinator::Coordinator;
use std::{path::PathBuf, sync::atomic::AtomicBool};
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(std::io::Error::other(
            "Usage: lease_probe resident|heavy DIRECTORY MARKER",
        ));
    }
    let coordinator = Coordinator::new(&args[2])?;
    let cancel = AtomicBool::new(false);
    let marker = PathBuf::from(&args[3]);
    match args[1].as_str() {
        "parent" => {
            let exclusive = coordinator.request_heavy(&cancel)?.reserve(&cancel)?;
            let mut command = std::process::Command::new(std::env::current_exe()?);
            command
                .args(["inherited", &args[2], &args[3]])
                .stdin(std::process::Stdio::null());
            exclusive.inherit_into(&mut command)?;
            let _child = command.spawn()?;
            let _ = std::io::stdin().read_line(&mut String::new())?;
        }
        "inherited" => {
            std::fs::write(&marker, "inherited")?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while !marker.with_extension("release").exists() && std::time::Instant::now() < deadline
            {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        "resident" => {
            let admission = coordinator.admit(&cancel)?;
            let _resident = coordinator.resident(&admission)?;
            drop(admission);
            std::fs::write(marker, "resident")?;
            let _ = std::io::stdin().read_line(&mut String::new())?;
        }
        "heavy" => {
            let intent = coordinator.request_heavy(&cancel)?;
            std::fs::write(&marker, "requested")?;
            let _exclusive = intent.reserve(&cancel)?;
            std::fs::write(marker, "exclusive")?;
            let _ = std::io::stdin().read_line(&mut String::new())?;
        }
        _ => return Err(std::io::Error::other("Unknown lease mode")),
    }
    Ok(())
}
