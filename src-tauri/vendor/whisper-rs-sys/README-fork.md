# Vendored fork of whisper-rs-sys 0.15.0

This is the published whisper-rs-sys 0.15.0 crate with one change: the
vendored `whisper.cpp/` tree is replaced by upstream whisper.cpp at commit
`b932ec55298a6ebd2bcd2a0d2e62f9df61e6008d` (the last commit shipping
**ggml 0.13.1**), with backend directories we don't build (CUDA, Vulkan,
SYCL, ...) pruned.

## Why

Both whisper-rs-sys and llama-cpp-sys-2 statically link their own copy of
ggml. The linker keeps whichever archive it sees first, so **both libraries
run on a single ggml copy**. If the two vendored ggml versions differ, the
`GGML_OP` enum values baked into each library's compiled code disagree with
the ggml runtime that actually won the link.

Concretely, with the stock crates (whisper ggml 0.9.x vs llama ggml 0.13.1):
llama 0.13.1 inserts `GGML_OP_GATED_DELTA_NET` mid-enum, so llama's code
reads whisper-built `GGML_OP_UNARY` (GELU) nodes as `GATED_DELTA_NET`.
Creating a context for the BGE embedding model then aborts the process in
`llama_context::sched_reserve()`'s fused-GDN auto-resolution pass:

```
GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0) failed
```

Pinning whisper.cpp to the same ggml vintage (0.13.1, verified identical
`ggml_op` and `ggml_unary_op` enums) makes the duplicate link benign.

## When bumping llama-cpp-sys-2 or whisper-rs

Re-verify the enums still match:

```sh
awk '/enum ggml_op \{/,/\};/' <each ggml.h> | grep -o 'GGML_OP_[A-Z0-9_]*'
```

against both `vendor/whisper-rs-sys/whisper.cpp/ggml/include/ggml.h` and the
llama-cpp-sys-2 vendored `llama.cpp/ggml/include/ggml.h`, and pick a
whisper.cpp commit whose ggml matches (search `git log -S 'set(GGML_VERSION_MINOR ...)'
-- ggml/CMakeLists.txt` in whisper.cpp).
