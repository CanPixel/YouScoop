# Vendored fork of whisper-rs 0.16

Vendored alongside whisper-rs-sys (see `../whisper-rs-sys/README-fork.md` for
the ggml-pinning rationale). Local changes to the Rust crate itself:

## `set_abort_callback_safe` type-confusion fix (`src/whisper_params.rs`)

Upstream 0.16 stores the abort closure as a `Box<Box<dyn FnMut() -> bool>>`
but instantiates the C trampoline with the caller's concrete closure type
`F`, so the trampoline reinterprets the fat `Box<dyn ...>` as `F` —
undefined behavior as soon as whisper.cpp invokes the callback. Fixed by
instantiating `trampoline::<Box<dyn FnMut() -> bool>>`, mirroring what
upstream's own `set_progress_callback_safe` does.

When bumping whisper-rs, check whether upstream has fixed this before
dropping the patch.
