//! Cross-process memory admission. OS locks are released if an app crashes.
//! A resident model holds activity.shared. Active work also holds intent.shared.
//! A large worker owns intent.exclusive, asks idle apps to unload, then takes
//! activity.exclusive before allocating its weights. No PID file is authority.
use std::{
    fs::{File, OpenOptions, TryLockError},
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

#[derive(Clone)]
pub struct Coordinator {
    dir: PathBuf,
}
pub struct Admission {
    dir: PathBuf,
    _intent: File,
}
pub struct Resident {
    _activity: File,
}
pub struct HeavyIntent {
    dir: PathBuf,
    _intent: File,
}
pub struct Exclusive {
    _activity: File,
    _intent: File,
}

impl Coordinator {
    pub fn new(dir: impl AsRef<Path>) -> io::Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        Ok(Self {
            dir: dir.as_ref().to_path_buf(),
        })
    }
    pub fn admit(&self, cancel: &AtomicBool) -> io::Result<Admission> {
        let file = open(&self.dir.join("intent.lock"))?;
        wait(&file, true, cancel)?;
        Ok(Admission {
            dir: self.dir.clone(),
            _intent: file,
        })
    }
    pub fn resident(&self, admission: &Admission) -> io::Result<Resident> {
        if admission.dir != self.dir {
            return Err(io::Error::other("Admission belongs to another coordinator"));
        }
        let file = open(&self.dir.join("activity.lock"))?;
        file.try_lock_shared().map_err(lock_error)?;
        Ok(Resident { _activity: file })
    }
    pub fn request_heavy(&self, cancel: &AtomicBool) -> io::Result<HeavyIntent> {
        let file = open(&self.dir.join("intent.lock"))?;
        wait(&file, false, cancel)?;
        Ok(HeavyIntent {
            dir: self.dir.clone(),
            _intent: file,
        })
    }
    pub fn heavy_requested(&self) -> io::Result<bool> {
        match open(&self.dir.join("intent.lock"))?.try_lock_shared() {
            Ok(()) => Ok(false),
            Err(TryLockError::WouldBlock) => Ok(true),
            Err(TryLockError::Error(error)) => Err(error),
        }
    }
}
impl HeavyIntent {
    pub fn reserve(self, cancel: &AtomicBool) -> io::Result<Exclusive> {
        let file = open(&self.dir.join("activity.lock"))?;
        wait(&file, false, cancel)?;
        Ok(Exclusive {
            _activity: file,
            _intent: self._intent,
        })
    }
}
impl Exclusive {
    /// Keep the lease owned by the worker too, so an abruptly killed parent
    /// cannot grant another app memory while its child still holds GPU weights.
    #[cfg(unix)]
    pub fn inherit_into(&self, command: &mut std::process::Command) -> io::Result<()> {
        use std::os::unix::{io::AsRawFd, process::CommandExt};
        let activity = self._activity.try_clone()?;
        let intent = self._intent.try_clone()?;
        // The closure owns cloned handles, so they remain valid until spawn
        // even if the original Exclusive is dropped. fcntl is signal-safe;
        // the child callback allocates nothing and acquires no Rust locks.
        unsafe {
            command.pre_exec(move || {
                for fd in [activity.as_raw_fd(), intent.as_raw_fd()] {
                    let flags = libc::fcntl(fd, libc::F_GETFD);
                    if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        Ok(())
    }
    #[cfg(not(unix))]
    pub fn inherit_into(&self, _command: &mut std::process::Command) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Desktop model workers currently require Unix lease inheritance",
        ))
    }
}
fn open(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}
fn lock_error(error: TryLockError) -> io::Error {
    match error {
        TryLockError::WouldBlock => io::Error::from(io::ErrorKind::WouldBlock),
        TryLockError::Error(error) => error,
    }
}
fn wait(file: &File, shared: bool, cancel: &AtomicBool) -> io::Result<()> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "Cancelled"));
        }
        match if shared {
            file.try_lock_shared()
        } else {
            file.try_lock()
        } {
            Ok(()) => return Ok(()),
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(40)),
            Err(TryLockError::Error(error)) => return Err(error),
        }
    }
}
