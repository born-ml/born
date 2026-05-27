# Changelog

All notable changes to the Born ML Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-05-27

### Added

- **GPU shared encoder accumulator** ([ADR-012](docs/dev/ADR-012-gpu-encoder-batching-buffer-cache.md))
  - One CommandEncoder for N compute passes instead of N encoders
  - 128 Finish() calls → 1 per batch. GPU utilization 55% → 70-80%
  - All 15 lazy ops simplified via `addComputePassToEncoder` (-456 lines)
- **GPU input buffer cache** — `getOrCreateInputBuffer`: tensor→GPU buffer identity mapping, weight matrices uploaded once
- **`Tensor.Persist()` / `Unpersist()` API** — marks GPU tensors to survive `ReclaimMemory` between training steps (carry state, rotary embeddings, model buffers)
- **GPU training example** (`examples/gpu-training/`) — MLP on synthetic data, 77.8 steps/sec
- **LazyMode for 11 ops** — Embedding, Cast, Or, And, Not, Eq, Ne, Ge, Le, Greater, Less — eliminates last CPU readbacks in forward pass
- **TieredPool with device limits** (ADR-017) — 12 log-spaced buckets from `device.Limits()`, budget enforcement, onOOM callback
- **FlushGPU + ReclaimMemory proxy** on AutodiffBackend
- **40 new tests** — 17 shared encoder/buffer cache, 12 pool, 11 activation (Sigmoid, Tanh, SiLU, Log forward+backward)
- **GPU training regression test** — 20-step MLP OOM test

### Changed

- **SumDim/MeanDim backward** — migrated from CPU broadcastTo/unsqueezeDim to `backend.Expand`/`backend.Reshape` (ADR-009 partial, -70 lines CPU code)
- **Sigmoid/Tanh backward** — zero CPU allocation via scalar ops composition (ADR-009)
- **CrossEntropy forward** — composed via backend ops, eliminates CPU readback
- **All 42 AutodiffBackend methods** now delegate to `b.inner.XXX()` (TASK-149/150) — zero CPU bypass in forward pass
- **Pool cleanup threshold** — gpuPoolFreeThresh 5→2 for faster deallocation of unused pages
- **wgpu upgraded** v0.28.11 → v0.29.0 (triple-backend: Pure Go / Rust / WASM)

### Fixed

- **GPU buffer leak** — `TieredPool.Release()` silently ignored non-pool buffers (params, uniforms, transient inputs). 43K buffers leaked per step. Now properly calls `buffer.Release()` for non-pool buffers
- **Carry state crash** — `ReclaimMemory()` destroyed model-level persistent tensors between training steps. Fixed via Persist/Unpersist lifecycle
- **GPU batched dispatch** — auto-flush pending command buffers every 128 dispatches. Prevents Windows TDR timeout (VK_ERROR_DEVICE_LOST) on integrated GPUs
- **Adam.Step in NoGrad** — prevents optimizer ops from recording on autodiff tape
- **Backward stop recording** — prevents ClearTape from killing optimizer moments
- **DeferReleaseGPUBuffer** — returns to pool immediately when no encoder active
- **backend.Release** — drains liveGPU + flushes pending, zero GC warnings on shutdown
- **Pool bucket cap** at MaxBufferSize from device limits

## [0.9.0] - 2026-05-17

### Added

- **CPU parallel BatchMatMul** — `sync.WaitGroup` + goroutines across batch dimension
  - Threshold: B ≤ 4 → sequential, B > 4 → parallel (`runtime.NumCPU()` workers)
  - All 4 variants: Float32, Float64, BroadcastFloat32, BroadcastFloat64
  - Fixed race condition: capped slice prevents overlapping zero-init across goroutines
- **CPU cache-tiled blocked MatMul** — 3-5x speedup for large matrices
  - i-block→k-block→j-block loop order (sequential B-matrix access)
  - Block sizes: 64 (float32, 16KB L1), 32 (float64, 8KB L1)
  - Threshold: m×n×k < 262K → naive fallback (overhead > benefit)
  - Micro-kernel extracted for compiler inlining
- **AVX2 SIMD MatMul micro-kernel** via Go 1.26 `goexperiment.simd`
  - `simd/archsimd`: LoadFloat32x8, BroadcastFloat32x8, MulAdd (FMA)
  - 4-row × 16-wide register block, zero allocations
  - Benchmark: 128×128 micro-kernel 3693 → 1058 ns/op (**3.49x**)
  - Build tag: `//go:build amd64 && goexperiment.simd` + scalar fallback
  - 13 correctness subtests
- **GPU batched dispatch** — queue lazy ops, single `queue.Submit` on Data() access
  - `finishAndQueueLazy` queues command buffers instead of immediate Submit
  - `flushCommands` submits all pending in single variadic `queue.Submit(cmdBufs...)`
  - All GPU resources (buffers + bind groups) kept alive via `lazyResources` until after Submit
  - Before: 50+ Submits per transformer forward pass (~25ms overhead). After: 1 Submit per readback
- **Embedding backward tests** — 5 tests covering 1D/2D shapes, duplicate indices, gradient flow, MulScalar chain
- **Go 1.26** — minimum Go version updated from 1.25 to 1.26 for `simd/archsimd` support

### Fixed

- **GPU batched dispatch**: bind groups and input buffers kept alive until after queue.Submit (BUG-LAZY-DEFER-RELEASE for all resource types)
- **GPU buffer copy**: `copyGPUBuffer` flushes pending commands + Poll before copy (prevents reading unsubmitted staging data)

## [0.8.3] - 2026-05-16

### Added

- **WebGPU GPU compute shaders for SelectAdd/ScatterAdd** — eliminates CPU-fallback bottleneck
  - SelectAdd: per-destination-row WGSL shader, no f32 atomics required
  - ScatterAdd: per-destination-element WGSL shader, supports up to 6D tensors
  - Results stay on GPU as lazy tensors — no GPU→CPU readback for intermediate backward results
  - Before: 27K ReadGPUBuffer calls per HRM backward step. After: 1 GPU dispatch each
  - HRM training: step time reduced from minutes to seconds
  - 13 GPU tests with CPU-GPU numeric parity verification
  - CPU fallback retained for non-lazy mode and unsupported dtypes
- `BORN_DEBUG_GPU=1` environment variable for diagnostic stderr logging of ReadGPUBuffer Poll/Map calls

### Fixed

- **WebGPU ReadGPUBuffer**: 10s timeout on `buffer.Map()` to prevent infinite hang if staging buffer is in invalid state

## [0.8.2] - 2026-05-16

### Added

- `SelectAdd` backend operation — scatter-add with 1-D indices (Embedding backward)
  - CPU: all dtypes with flat-index computation, 7 tests
  - WebGPU: CPU-data fallback (f32 atomics not in WGSL core spec)
- `ScatterAdd` backend operation — scatter-add with N-D indices (Gather backward)
  - CPU: all dtypes, validates shapes and index bounds
  - WebGPU: CPU-data fallback
- `MulScalarOp`, `AddScalarOp`, `SubScalarOp`, `DivScalarOp` autodiff operations
  - All four scalar ops now record on the gradient tape
  - Previously scalar ops were proxy-only — gradients did not flow through them

### Fixed

- **Tokenizer**: HuggingFace tokenizer now applies normalizer from tokenizer.json
  - SentencePiece models (LLaMA, Mistral) require Prepend+Replace normalizer to map spaces to `▁` (U+2581)
  - Without normalization, every token was wrong ("The" → ID 1576 instead of "▁The" → ID 450)
  - Supported normalizers: Sequence, Prepend, Replace, Lowercase, Strip
  - 13 new tests for normalizer parsing and word splitting
- **Autodiff**: Scalar ops (MulScalar, AddScalar, SubScalar, DivScalar) not recorded on gradient tape
  - Embedding weights received zero gradients when scaled by `embedScale * tokenEmbedding`
  - All models using `MulScalar` in forward pass had broken gradient flow

### Changed

- **Autodiff backward ops**: Migrated 7 ops from CPU-fallback to forward composition ([ADR-009](docs/dev/ADR-009-backward-ops-composition.md))
  - SiLU, Log, ReLU, CrossEntropy, MeanDim, Embedding, Gather backward now use backend ops only
  - Tensors never leave the GPU during backward pass (eliminates GPU→CPU readback)
  - Helper functions (`sumAll`, `sumAlongDimension`, `negateGradient`) now delegate to backend
  - Follows Burn (Rust) reference architecture: all gradients via forward ops composition
  - Net -835 lines of CPU-only backward code replaced by backend-delegated operations

## [0.8.1] - 2026-05-15

### Added

- `models/llama`: New LLaMA model package with GGUF loading and injectable attention
  - `Model[B]`, `Layer[B]`, `NewModel`, `NewModelCache` — full transformer decoder
  - Grouped-Query Attention (GQA) with RoPE (rotate-half convention)
  - SwiGLU FFN, RMSNorm, incremental KV-cache decoding
  - `WithAttentionFunc` option for runtime attention replacement (Flash Attention, etc.)
  - `Layer.DebugForward` returns attn and FFN contributions for diagnostics
  - `LoadGGUF(path, backend)` — loads Q4_K, Q5_K, Q6_K, Q8_0, F16, F32 weights from GGUF files
  - Implements `generate.LLMModel` interface — drop-in for `generate.TextGenerator`
  - Tested with TinyLlama-1.1B-Q8_0: Paris top-1 answer confirmed
  - Note: Q4_K_M (4-bit) requires quantized matmul for correct inference; Q8_0 (8-bit) works with full dequantization
- `loader`: Public API for model loading (`LoadGGUF`, `LoadSafeTensors`)
  - Namespace-clean: `loader.LoadGGUF(path, backend)` at module root
- `nn.SetSeed(seed)` / `nn.ResetSeed()` for reproducible weight initialization
  - Seeds both nn (Xavier, Embedding) and tensor (Randn, Rand) random sources
  - Thread-safe (sync.Mutex per package)
  - Enables deterministic model creation for experiments and testing
  - Public API: `nn.SetSeed(42)` before `nn.NewLinear(...)` guarantees identical weights
- `Clamp` element-wise tensor operation ([#61](https://github.com/born-ml/born/pull/61) by [@bennibbelink](https://github.com/bennibbelink))
  - Restricts values to `[min, max]` range
  - CPU: `int32`, `int64`, `float32`, `float64`
  - WebGPU: `float32`, `int32` (dedicated WGSL `clamp()` shader)
  - Autodiff backward: gradient masked by `min <= x <= max`
  - Panics on NaN bounds (float types)
  - `minBound > maxBound` → all values set to `maxBound` (matches PyTorch)
- `internal/loader`: `GGMLMapper` — maps GGUF-native (`blk.{i}.*`) tensor names to Born standard names
  - `DetectNaming` identifies HuggingFace vs GGML weight naming conventions automatically
  - `GetMapperForNaming` selects the correct mapper from a weight name sample

### Fixed

- **RoPE**: Fixed rotate-half convention (was interleaved, caused incorrect positional encoding for LLaMA)
  - Interleaved: `[-x1, x0, -x3, x2, ...]` — wrong for LLaMA/HuggingFace models
  - Rotate-half: `[-xn, x0, ..., -x2n, xn+1, ...]` — correct convention now implemented
- **GGUF Q4_K / Q5_K**: Correct scale unpacking algorithm
  - `sc[0..7]` extracted from low 6 bits of scale bytes (was reading wrong bit positions)
  - `m[0..7]` (minimum values) correctly assembled from high 2 bits + low nibble pattern
- **GGUF Float16**: Correct subnormal handling in `Float16ToFloat32`
  - Subnormals (`exp==0, mantissa!=0`) now expand correctly to `(-1)^sign * 2^-14 * mantissa/1024`
  - Previously treated subnormals as zero, silently corrupting F16 model weights
- **GGUF loader**: GGML tensor naming support (`blk.{i}.*` format)
  - Files produced by llama.cpp use GGML names; Born previously only handled HuggingFace names
  - Weight routing now correctly maps `blk.0.attn_q.weight → layers.0.attn.q.weight`
- **LLaMA loader**: Tied embeddings — `lm_head.weight` is now copied from `embedding.weight`
  when absent in the GGUF file (standard for TinyLlama, LLaMA-2, etc.)

### Changed

- **WebGPU SiLU**: SiLU activation shader connected to backend (`ops.go` now exposes `SiLU` method)
  - Previously the WGSL shader existed but was unreachable; now fully wired
- **gogpu/wgpu**: upgraded v0.26.8 → v0.27.5

## [0.8.0] - 2026-04-26

### Changed

- **WebGPU backend migrated from go-webgpu to gogpu/wgpu** ([#40](https://github.com/born-ml/born/issues/40))
  - Replaced `github.com/go-webgpu/webgpu` with `github.com/gogpu/wgpu` v0.26.8 (pure Go, zero CGO)
  - **No more shared library dependency** — no `.dll`/`.so`/`.dylib` downloads needed
  - True single binary deployment: `go build` produces executable with GPU support built in
  - Vulkan primary compute backend — stable across all platforms and GPU vendors
  - WGSL shaders unchanged — full backward compatibility
  - Fixed: PipelineLayout kept alive for Vulkan SetBindGroup (was freed prematurely)
  - Fixed: lazy ops immediate submit (prevents buffer lifetime issues with DestroyQueue)
  - Fixed: lazy chain `copyGPUBuffer` immediate submit (prevents stale data in chained ops)
  - Fixed: `runtime.KeepAlive` guards prevent GC finalizer races on GPU buffers
  - Fixed: `Poll(PollWait)` in Release() ensures GPU idle before resource destruction
  - All 105 GPU tests pass, validated with real model training (HRM, 20 epochs, 0 crashes)

### Added

- `Sign` and `Abs` element-wise tensor operations — full vertical slice ([#59](https://github.com/born-ml/born/pull/59) by [@bennibbelink](https://github.com/bennibbelink))
  - `Backend.Sign` / `Backend.Abs` interface methods
  - CPU implementation with per-type helpers: `uint8`, `int32`, `int64`, `float32`, `float64`
  - Integer `Abs` uses two's-complement wraparound semantics (`abs(MinInt) == MinInt`), matching Burn / NumPy / PyTorch
  - WebGPU implementation (float32 only, with dtype guards)
  - Autodiff support: `SignOp` (zero gradient) and `AbsOp` (grad × sign)
  - Mock backend, public `Tensor.Sign()` / `Tensor.Abs()` API
  - Comprehensive tests including NaN, ±Inf, `MinInt`/`MaxInt` edge cases

## [0.7.16] - 2026-04-10

### 🎉 Community Contributions — @gmohmad & @bennibbelink

Third external contributor [@gmohmad](https://github.com/gmohmad) with 5 PRs! Plus continued work from [@bennibbelink](https://github.com/bennibbelink).

**Added**:
- ONNX `LayerNormalization` operator with new `normalization_ops.go` category ([#47](https://github.com/born-ml/born/pull/47) by @gmohmad)
- `BroadcastShapesMatMul` — NumPy-style broadcasting for batched matrix multiplication ([#49](https://github.com/born-ml/born/pull/49) by @gmohmad)
- `BatchMatMul` now supports 2D×3D, singleton batch dims, multi-dim broadcasting
- ONNX `MatMul` auto-delegates to `BatchMatMul` for >2D inputs
- `tensor.BroadcastShapesMatMul` public API
- ONNX `AttributeProto` tensor attribute (field 5) parsing ([#53](https://github.com/born-ml/born/pull/53) by @gmohmad)

**Fixed**:
- `Squeeze` scalar handling: returns `Shape{}` (scalar) instead of `Shape{1}` (1D) ([#50](https://github.com/born-ml/born/pull/50) by @gmohmad)
- ONNX `AttributeProto` parser: correct protobuf field numbers, non-packed encoding support ([#53](https://github.com/born-ml/born/pull/53) by @gmohmad)
- CPU backend: prevent inplace mutation when operands alias — `Mul(x,x)` no longer corrupts input ([#55](https://github.com/born-ml/born/pull/55), fixes [#45](https://github.com/born-ml/born/issues/45), reported by @gmohmad)
- CI: added `test` gate job for branch protection required check ([#52](https://github.com/born-ml/born/pull/52))

**Refactored**:
- `ConvDims` and `PoolDims` parameter structs to reduce argument counts in conv2d/maxpool2d ([#46](https://github.com/born-ml/born/pull/46) by @bennibbelink)
- Moved `ConvDims`/`PoolDims` to `internal/tensor/` shared package, eliminating autodiff→cpu cross-dependency (fixes [#48](https://github.com/born-ml/born/issues/48))
- Extracted 14 helper functions from conv2d/maxpool2d inner loops (fixes [#17](https://github.com/born-ml/born/issues/17)) — compiler-inlined, Conv2D batch path ~28% faster

**Added** (PR #56 by @gmohmad):
- ONNX comparison operators: Greater, GreaterOrEqual, Less, LessOrEqual
- ONNX logical operators: Not, And, Or, Xor (new `logical_ops.go`)
- ONNX Erf operator
- Broadcasting for boolean ops (Or, And) and all comparison ops in CPU backend

**Fixed** (PR #56 by @gmohmad):
- Updated `onnx/onnx.go` doc comment to match all registered operators (fixes [#43](https://github.com/born-ml/born/issues/43))

**ONNX operators**: 39 → 49

---

## [0.7.15] - 2026-04-07

### 🎉 Community Contribution — Erf Operator

Second external contribution! Thanks to [@bennibbelink](https://github.com/bennibbelink).

**Added**:
- `Erf` (error function) operator — full vertical slice across the entire stack
- Backend interface: `Erf(x *RawTensor) *RawTensor`
- CPU backend: `math.Erf` for float32/float64
- WebGPU backend: Abramowitz & Stegun polynomial approximation shader
- Autodiff: backward pass with correct derivative `2/√π · exp(-x²)`
- Mock backend, Tensor API (`tensor.Erf()`)
- Comprehensive tests: forward + backward, float32/float64, edge cases (Inf, NaN)

**Links**:
- PR: [#37](https://github.com/born-ml/born/pull/37) by @bennibbelink

---

## [0.7.14] - 2026-03-04

### 🎉 Community Contribution — ONNX Equal Operator

First external contribution! Thanks to [@jsully1720](https://github.com/jsully1720).

**Added**:
- ONNX `Equal` operator — binary element-wise comparison returning bool tensor
- New `comparison_ops.go` category for ONNX comparison operators
- `registerComparisonOps()` wired into operator registry

**ONNX operators**: 38 → 39

**Links**:
- PR: [#34](https://github.com/born-ml/born/pull/34) by @jsully1720
- Issue: [#35](https://github.com/born-ml/born/issues/35)

---

## [0.7.13] - 2026-03-02

### 🔧 Dependencies Update

Update WebGPU backend to v0.4.1 with critical ABI compliance fixes.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.4.0 → **v0.4.1**
- `go-webgpu/goffi` v0.4.0 → **v0.4.1** (indirect)

**Upstream Bug Fixes (ABI compliance)**:
- Float32 encoding: correct XMM bit patterns via `math.Float32bits`
- AMD64 Unix stack: arguments beyond 6 GP registers properly pushed to stack
- ARM64 Unix stack: arguments beyond 8 GP registers correctly spilled to stack
- AMD64 struct returns (9-16 bytes): RAX+RDX register pair properly assembled
- AMD64 sret pointer: structs > 16 bytes use caller buffer as first argument (RDI)
- ARM64 HFA spilling: Homogeneous Floating-Point Aggregate overflow follows AAPCS64

**Upstream Enhancements**:
- `runtime.KeepAlive` prevents GC of argument pointers during FFI calls
- `ErrTooManyArguments` overflow detection for calls exceeding 15 arguments

**Impact**: Critical ABI correctness fixes for multi-platform GPU backend reliability.

**Links**:
- Upstream release: [go-webgpu v0.4.1](https://github.com/go-webgpu/webgpu/releases/tag/v0.4.1)

---

## [0.7.12] - 2026-02-27

### 🔧 Dependencies Update

Update WebGPU backend to v0.4.0 with FFI hardening and improved library loading.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.3.2 → **v0.4.0**

**Upstream Improvements**:
- Null handle guards on 27 public FFI methods — prevents SIGSEGV on nil/released objects
- `ptrFromUintptr` helper — eliminates all `go vet` unsafe.Pointer warnings
- `WGPU_NATIVE_PATH` env var for custom wgpu-native library path
- `loadLibrary` returns `(Library, error)` with proper error propagation
- Windows DLL eager loading — errors surface at init, not at first use
- Enhanced `Init()` error messages with library path and remediation suggestions
- 85 new null guard test cases

**Impact**: Significantly improved safety and debuggability of GPU backend initialization.

**Links**:
- Upstream release: [go-webgpu v0.4.0](https://github.com/go-webgpu/webgpu/releases/tag/v0.4.0)

---

## [0.7.11] - 2026-02-27

### 🔧 Dependencies Update

Update WebGPU backend to v0.3.2 with crosscall2 callback integration.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.3.1 → **v0.3.2**
- `go-webgpu/goffi` v0.3.9 → **v0.4.0** (indirect)

**Upstream Improvements**:
- crosscall2 integration — callbacks now work from C-library-created threads (Metal, wgpu-native)
- fakecgo trampoline register fixes synced with purego v0.10.0

**Impact**: Improved callback reliability on macOS Metal and native WebGPU implementations.

**Links**:
- Upstream release: [go-webgpu v0.3.2](https://github.com/go-webgpu/webgpu/releases/tag/v0.3.2)

---

## [0.7.10] - 2026-02-18

### 🔧 Dependencies Update

Update WebGPU backend to v0.3.1 with critical ARM64 callback fix.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.3.0 → **v0.3.1**
- `go-webgpu/goffi` v0.3.8 → **v0.3.9** (indirect)

**Upstream Fixes**:
- ARM64 callback trampoline rewrite — fixes LR corruption for callbacks at index > 0
- Symbol rename to prevent linker collision with purego

**Code Quality**:
- Removed 101 unused `//nolint:gosec` directives (gosec linter updated, no longer flags these)
- Standardized remaining nolint comments to short format

**Impact**: Critical fix for macOS Apple Silicon and Linux ARM64 users.

**Links**:
- Upstream release: [go-webgpu v0.3.1](https://github.com/go-webgpu/webgpu/releases/tag/v0.3.1)

---

## [0.7.9] - 2026-02-09

### 🔧 Dependencies Update

Update WebGPU backend to v0.3.0 with new capability-querying API and typed errors.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.2.1 → **v0.3.0**

**New Upstream Features Available**:
- `Surface.GetCapabilities()` — query supported formats, present modes, alpha modes
- `Device.GetFeatures()` / `Device.HasFeature()` — feature enumeration
- `Device.GetLimits()` — device limits (experimental)
- Typed errors with `errors.Is()` / `errors.As()` support (`ErrValidation`, `ErrOutOfMemory`, `ErrInternal`, `ErrDeviceLost`)
- Resource leak detection via `SetDebugMode(true)` / `ReportLeaks()`

**Links**:
- Upstream release: [go-webgpu v0.3.0](https://github.com/go-webgpu/webgpu/releases/tag/v0.3.0)

---

## [0.7.8] - 2026-01-29

### 🔧 GoGPU Ecosystem Integration (Phase 1)

Migrate WebGPU backend to use unified `gputypes` for future dual-backend support.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.1.4 → **v0.2.1**
- `go-webgpu/goffi` v0.3.7 → **v0.3.8**
- `dlclark/regexp2` v1.10.0 → **v1.11.5**
- `google/uuid` v1.3.0 → **v1.6.0**
- Added `gogpu/gputypes` **v0.2.0** (new dependency)

**Changes**:
- Migrated all WebGPU types from `wgpu.*` to `gputypes.*`:
  - `BufferUsage`, `BufferUsageStorage`, `BufferUsageCopySrc`, `BufferUsageCopyDst`
  - `PowerPreferenceHighPerformance`
- Updated 10 files in `internal/backend/webgpu/`
- Fixed 3 prealloc warnings in linter (examples + internal/nn)

**Why This Matters**:
- Prepares codebase for **Pure Go WebGPU backend** (`gogpu/wgpu`)
- Unified type system enables future dual-backend architecture
- Build tags will allow: `go build` (Rust FFI) vs `go build -tags purego` (Pure Go)

**Links**:
- Upstream release: [go-webgpu v0.2.0](https://github.com/go-webgpu/webgpu/releases/tag/v0.2.0)
- GoGPU ecosystem: [github.com/gogpu](https://github.com/gogpu)
- Integration plan: [TASK-110](docs/dev/kanban/backlog/TASK-110-backend-strategy-gogpu.md)

---

## [0.7.7] - 2026-01-06

### 🔧 Public API Improvements

Refactored public API packages to use proper Go interfaces instead of type aliases where possible.

**Improvements**:
- `tensor/`: Added `Backend` interface with 40+ methods (was type alias)
- `nn/`: Added `Module` interface with full method definitions
- `onnx/`: Added `Model` interface for ONNX model operations
- `optim/`: Now uses public `nn.Parameter` in function signatures
- `autodiff/`: Now uses public `tensor` types
- `backend/cpu`, `backend/webgpu`: Added compile-time interface checks

**Technical Details**:
- Improves [pkg.go.dev](https://pkg.go.dev/github.com/born-ml/born) documentation by hiding internal paths
- External packages can now properly import and use the public API
- Some interfaces (`Optimizer`, `ModelReader`) remain as type aliases due to Go's type system constraints

**Fixed Issues**:
- [#25](https://github.com/born-ml/born/issues/25) — ONNX package not accessible from external packages

---

## [0.7.6] - 2026-01-03

### 🔧 ARM64 Darwin Enhancement

Comprehensive ARM64 Darwin support with enhanced struct handling, tested on M3 Pro hardware.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.1.3 → **v0.1.4**
- `go-webgpu/goffi` v0.3.6 → **v0.3.7**

**Improvements**:
- Proper layout for nested and complex struct types
- Automatic struct layout computation for integer/float combinations
- Enhanced struct return handling (9-16 bytes) utilizing X0 and X1 registers

**Fixed Issues**:
- Resolved segmentation fault in string output benchmarks on Darwin systems

**Contributors**:
- @ppoage — ARM64 Darwin implementation, Objective-C test suite, assembly verification

**Links**:
- Upstream release: [go-webgpu v0.1.4](https://github.com/go-webgpu/webgpu/releases/tag/v0.1.4)

---

## [0.7.5] - 2025-12-29

### 🔧 ARM64 Hotfix

Update GPU backend dependencies with critical ARM64 fixes for Apple Silicon.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.1.2 → **v0.1.3**
- `go-webgpu/goffi` v0.3.5 → **v0.3.6**

**Fixed Issues**:
- ARM64 HFA returns (NSRect with 4×float64 now correctly returns all values on Apple Silicon)
- Large struct returns (structs exceeding 16 bytes now properly use X8 register)
- macOS ARM64 display (blank window issue where GPU dimensions returned 0×0)

**Links**:
- Upstream release: [go-webgpu v0.1.3](https://github.com/go-webgpu/webgpu/releases/tag/v0.1.3)

---

## [0.7.4] - 2025-12-27

### ✨ New Feature: Linear Layer Without Bias

Add `WithBias` option to `nn.NewLinear` for creating Linear layers without bias term.

**New API**:
```go
// With bias (default, backwards compatible)
layer := nn.NewLinear(784, 128, backend)

// Without bias (for LLaMA-style models, LM head, etc.)
lmHead := nn.NewLinear(hiddenSize, vocabSize, backend, nn.WithBias(false))
```

**Changes**:
- Add `LinearOption` type and `WithBias(bool)` functional option
- Add `HasBias()` method for introspection
- Update `SwiGLUFFN` to use public API
- Export `WithBias` in public `nn` package

**Use Cases**:
- LM Head in language models (GPT, LLaMA, HRM)
- Attention projections (some architectures)
- SwiGLU FFN layers

**Links**:
- PR: [#22](https://github.com/born-ml/born/pull/22)

---

## [0.7.3] - 2025-12-27

### 🔧 Dependencies Update

Hotfix release updating GPU backend dependencies to latest versions.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.1.1 → **v0.1.2**
- `go-webgpu/goffi` v0.3.3 → **v0.3.5**

**Links**:
- PR: [#21](https://github.com/born-ml/born/pull/21)

---

## [0.7.2] - 2025-12-24

### 🔧 Dependencies Update

Hotfix release updating GPU backend dependencies for improved stability.

**Updated Dependencies**:
- `go-webgpu/webgpu` v0.1.0 → **v0.1.1**
- `go-webgpu/goffi` v0.3.1 → **v0.3.3**

**Documentation**:
- Updated `.claude/CLAUDE.md` to v3.0 (optimized structure, accurate project info)
- Added `TASK-110-backend-strategy-gogpu.md` for future GPU backend strategy planning

**Links**:
- PR: [#18](https://github.com/born-ml/born/pull/18)

---

## [0.7.1] - 2025-12-16

### 🔧 Code Quality Refactoring (Issue #14)

Patch release addressing cognitive complexity concerns raised by community contributor [@marcelloh](https://github.com/marcelloh). Applied [Burn framework](https://github.com/tracel-ai/burn) patterns for improved code quality and maintainability.

**Pre-Slice Bounds Elimination** (`internal/backend/cpu/conv2d.go`, `maxpool2d.go`):
- Extract row slices BEFORE inner loops to eliminate bounds checks
- Hierarchical pre-slicing for nested loop structures
- Enables Go compiler to prove safety and optimize vectorization

**Stride Specialization** (`internal/backend/cpu/conv2d.go`):
- Separate fast paths for `stride=1, padding=0` case (most common)
- Specialized functions: `conv2dFloat32Stride1NoPad`, `conv2dInputBackwardFloat32Stride1NoPad`
- Enables compiler auto-vectorization for common case

**Flash Attention CPU Refactor** (`internal/nn/flash_attention.go`):
- **Complexity reduced: 111 → <30** (removed `//nolint:gocognit` directive)
- Extracted `FlashDims`, `FlashConfig` structs for configuration
- Helper functions: `flashAttentionScoreBlock`, `flashAttentionExtractValues`, `flashAttentionProcessQuery`
- Each helper under 50 AST nodes (Go compiler inlines automatically)

**Autodiff Orchestration** (`internal/autodiff/ops/`):
- Separated orchestration from computation (Burn pattern)
- New files: `conv2d_backward.go`, `maxpool2d_backward.go` in CPU backend
- `autodiff/ops/conv2d.go`: 409 → 67 lines (delegation only)
- Extended Backend interface with backward operation methods

**Parallel Execution Utilities** (`internal/parallel/`):
- New package for reusable parallel execution patterns
- `parallel.Config` - configurable parallelism settings
- `parallel.For()` - parallel for-loop with automatic sequential fallback
- `parallel.ForBatch()` - optimized for batch×channels iteration pattern
- Ready for integration into CPU backend operations

**Backend Interface Extended** (`internal/tensor/backend.go`):
- `Conv2DInputBackward(input, kernel, grad, stride, padding)` - gradient w.r.t. input
- `Conv2DKernelBackward(input, kernel, grad, stride, padding)` - gradient w.r.t. kernel
- `MaxPool2DBackward(input, grad, maxIndices, kernelSize, stride)` - gradient propagation
- WebGPU backend updated with stub implementations

**Code Quality**:
- All files properly formatted (`go fmt ./...`)
- 0 linter issues (golangci-lint)
- All tests passing
- No performance regression

**Files Changed**: 13 files, +994/-460 lines

**Links**:
- Issue: [#14](https://github.com/born-ml/born/issues/14)
- PR: [#15](https://github.com/born-ml/born/pull/15)
- Community: Thanks to [@marcelloh](https://github.com/marcelloh) for the detailed analysis!

---

## [0.7.0] - 2025-12-10

### ⚡ Flash Attention 2 + Speculative Decoding + GGUF Import

Major release focused on inference optimization for LLM deployment.

**Flash Attention 2** (`internal/nn/flash_attention.go`, `internal/nn/online_softmax.go`):
- **O(N) Memory** - Tiled computation never materializes full N×N attention matrix
- **Online Softmax** - Incremental softmax with rescaling for numerical stability
- **WebGPU Shader** - WGSL compute shader with workgroup shared memory
- **Configurable Tiles** - Block sizes 64 and 128 supported
- **Head Dimensions** - Supports 64, 96, 128, 256
- **Causal Masking** - Built-in support for autoregressive models
- **CPU Reference** - Validation implementation for correctness testing
- **2x+ Speedup** - On sequences 8K+ vs standard attention

**Speculative Decoding** (`internal/generate/speculative.go`):
- **Draft Model** - Small model generates K candidate tokens speculatively
- **Parallel Verification** - Target model verifies all candidates in single batch
- **Modified Rejection Sampling** - Mathematically correct token acceptance
- **2-4x Speedup** - For autoregressive text generation
- **Configurable** - Draft steps (K), temperature, sampling parameters

**GGUF Import** (`internal/gguf/`):
- **Parser** - Complete GGUF v3 format parsing (types, metadata, tensor info)
- **Loader** - Memory-mapped tensor data loading
- **K-Quant Dequantization** - Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
- **Converter** - GGUF tensors to Born tensor format
- **llama.cpp Ecosystem** - Load LLaMA, Mistral, DeepSeek, Qwen models

**Code Quality**:
- Fixed 226 gosec G115 integer overflow warnings across codebase
- All files properly formatted (gofmt)
- 0 linter issues (golangci-lint)

**Tests**:
- Flash Attention: GPU vs CPU correctness validation (< 1e-4 error)
- Speculative Decoding: 11 tests, 93.1% coverage
- GGUF: 52 tests, 75% coverage

**Files Added**:
- `internal/nn/flash_attention.go` - Flash Attention module
- `internal/nn/online_softmax.go` - Online softmax implementation
- `internal/nn/flash_attention_test.go` - CPU tests
- `internal/nn/flash_attention_gpu_test.go` - GPU tests
- `internal/backend/webgpu/flash_attention.go` - GPU execution
- `internal/backend/webgpu/shaders.go` - Added flashAttentionShader
- `internal/generate/speculative.go` - Speculative decoding
- `internal/generate/speculative_test.go` - Speculative tests
- `internal/gguf/` - Complete GGUF package (types, parser, loader, dequant, convert)

---

## [0.6.0] - 2025-12-04

### 🚀 ONNX Import & Lazy GPU Mode

Major release adding ONNX model import and GPU-resident lazy evaluation for dramatically improved performance.

**ONNX Import API** (`internal/onnx/`):
- **ONNX Parser** - Parse `.onnx` model files (protobuf format)
- **Model Loader** - Load weights and construct computation graph
- **30+ Operators** - Standard ONNX operator support:
  - Activations: ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU
  - Math: MatMul, Add, Mul, Div, Sub, Sqrt, Pow, Exp, Log
  - Shape: Reshape, Transpose, Squeeze, Unsqueeze, Concat, Split
  - Utility: Gather, Slice, Cast, Constant, Identity, Flatten
- **Operator Registry** - Extensible operator registration system

**Lazy GPU Evaluation** (`internal/tensor/lazy_gpu.go`):
- **GPU-Resident Tensors** - Data stays on GPU until explicitly needed
- **LazyGPUData** - Reference to GPU buffer with lazy CPU transfer
- **Automatic Memory Management** - `runtime.SetFinalizer` for GPU buffer cleanup
- **Zero CPU Round-trips** - Chained operations stay entirely on GPU

**Command Batching** (`internal/backend/webgpu/`):
- **Batch GPU Commands** - Accumulate commands instead of immediate submit
- **Reduced Sync Overhead** - ~200 submits → 1-2 per operation chain
- **FlushCommands()** - Explicit synchronization when needed
- **Performance Impact**: ~90s/step → <5s/step for model training

**GPU-to-GPU Copy**:
- **CopyBufferToBuffer** - Direct GPU memory transfer
- **No CPU Round-trip** - Eliminated GPU→CPU→GPU transfers in lazy chains
- **~100x Speedup** - Per-operation transfer overhead eliminated

**Raw Tensor Operations** (`internal/tensor/raw_ops.go`):
- **50+ Operations** - Comprehensive tensor manipulation
- **Argmax, TopK** - Selection operations
- **Type Conversions** - Float32, Int32, Bool conversions
- **Broadcasting** - NumPy-style shape broadcasting
- **Advanced Indexing** - Gather, Scatter operations

**Bug Fixes**:
- Fixed GPU memory leak when lazy tensors go out of scope
- Fixed typed accessors (AsInt32, AsInt64, etc.) bypassing lazy realization
- Fixed Where and Sum operations missing lazy mode support

**Tests**:
- 15+ new ONNX tests (parser, loader, operators)
- Lazy mode chain tests
- Command batching tests

**Files Added**:
- `internal/onnx/` - Complete ONNX import package
- `internal/tensor/lazy_gpu.go` - Lazy GPU data structures
- `internal/tensor/raw_ops.go` - Raw tensor operations
- `internal/backend/webgpu/lazy_compute.go` - Lazy GPU operations
- `internal/backend/webgpu/gpu_*.go` - GPU tensor and autodiff support

---

## [0.5.5] - 2025-12-03

### ⚡ WebGPU Performance Hotfix

Critical performance fix for transformer training on WebGPU backend.

**Problem Fixed**:
- Multi-dimensional Transpose operations (3D+) were falling back to CPU
- Expand (broadcasting) was CPU-only
- Result: ~60s/batch for small transformer models (should be <1s)

**New GPU Operations**:
- **TransposeND shader** - N-dimensional transpose on GPU (up to 6D)
- **Expand shader** - NumPy-style broadcasting on GPU
- Both support `float32` and `int32` data types

**Performance Impact**:
- ~60x speedup for attention operations
- Transformer training now usable on WebGPU

**Tests**:
- 9 new tests: `TestTranspose3D`, `TestTranspose4D`, `TestTranspose5D`, `TestExpandBroadcast`, etc.

**Files Changed**:
- `internal/backend/webgpu/shaders.go` - Added WGSL shaders
- `internal/backend/webgpu/compute.go` - Added `runTransposeND`, `runExpand`
- `internal/backend/webgpu/ops.go` - Removed CPU fallback
- `internal/backend/webgpu/ops_extended.go` - Removed CPU fallback
- `internal/backend/webgpu/ops_nd_test.go` - New test file

## [0.5.4] - 2025-12-03

### 💾 Model Serialization

Production-ready model serialization with Format v2 best practices.

**New Features**:
- **Born Native Format v2** (`.born`) - SHA-256 checksum, security validation
- **Checkpoint API** - Save/resume training with optimizer state
- **SafeTensors Export** - HuggingFace ecosystem compatibility
- **Memory-Mapped Reader** - Efficient loading for 70GB+ models

**API**:
- `nn.Save(model, "model.born", "ModelType", metadata)` - Save model
- `nn.Load("model.born", backend, model)` - Load model
- `nn.SaveCheckpoint(path, model, optimizer, epoch, step, loss)` - Save checkpoint
- `nn.LoadCheckpoint(path, backend, model, optimizer)` - Resume training
- `serialization.WriteSafeTensors(path, tensors, metadata)` - Export for HuggingFace

**New Package**:
- `internal/serialization` - Format writer/reader, validation, mmap

**Tests**:
- 26 new tests for serialization, checkpoints, SafeTensors

## [0.5.3] - 2025-12-02

### 🐛 WebGPU Backend Fixes (HRM Compatibility)

**Bug Fixes**:
- **Comparison ops** - Now always return `float32` (0.0/1.0), even for `int32` inputs
- **Sum int32** - Added WGSL shader for int32 sum reduction
- **Sum scalar shape** - Fixed return shape from `[1]` to `[]` for proper scalar handling
- **Where int32 condition** - Added support for int32 condition tensors
- **Where broadcasting** - Added NumPy-style broadcasting (like Burn)
- **Gather backward** - Support for int32, int64, float32 index tensors

**New Functions**:
- `runComparisonOp` - Dedicated function for comparison operations
- `int32ToFloat32` - Helper for int32 to float32 conversion

**Tests**:
- 3 new Gather backward tests (int64 indices, boundary, dim0 2D)

## [0.5.2] - 2025-12-01

### ✨ Public WebGPU API

- Added public `backend/webgpu` package with `NewBackend()` function
- Windows build tag support for WebGPU
- Updated README with WebGPU API example

## [0.5.1] - 2025-12-01

### 🐛 Fixes

- Minor fixes after v0.5.0 release

## [0.5.0] - 2025-12-01

### 🚀 Phase 5: LLM Support

Major release adding complete LLM inference support! Run LLaMA, Mistral, DeepSeek, and other modern language models with Born.

### ✨ Added

**Grouped Query Attention (GQA)** (`internal/nn/gqa.go`):
- **GroupedQueryAttention** - Memory-efficient attention for LLaMA 2/3, Mistral
- **RepeatKV** - KV head broadcasting (e.g., 8 KV heads → 32 Q heads)
- **MQA helper** - Multi-Query Attention config (extreme GQA with 1 KV head)
- Full RoPE integration with KV-cache support
- 4:1 memory savings for KV-cache vs standard MHA

**SwiGLU & GLU Variants** (`internal/nn/glu.go`, `internal/nn/swiglu_ffn.go`):
- **SwiGLU** - `x * SiLU(gate)` activation (LLaMA, Mistral)
- **GeGLU** - `x * GELU(gate)` activation
- **ReGLU** - `x * ReLU(gate)` activation
- **GLU** - `x * sigmoid(gate)` (classic)
- **SwiGLUFFN** - Complete feed-forward module with gate/up/down projections
- Configurable bias (LLaMA uses no bias)

**Model Loader** (`internal/loader/`):
- **GGUF format support** - Read LLaMA, Mistral, DeepSeek model files
- **GGUFReader** - Parse metadata and tensor info
- **Weight Mappers** - Architecture-specific weight name translation
  - `LLaMAMapper` - LLaMA 1/2/3 models
  - `MistralMapper` - Mistral 7B and variants
  - `DeepSeekMapper` - DeepSeek models
- **DetectArchitecture** - Auto-detect model type from tensor names
- Support for F32, F16 dtypes (quantized types require dequant)

**Tokenizer Integration** (`internal/tokenizer/`):
- **TikToken** - OpenAI's BPE tokenizer (GPT-3.5, GPT-4)
- **BPE Tokenizer** - Generic Byte Pair Encoding
- **HuggingFace format** - Load tokenizer.json from HF models
- **Chat Templates** - Format multi-turn conversations
  - ChatML (OpenAI style)
  - LLaMA (Meta format)
  - Mistral (with [INST] tags)
- **Special tokens** - BOS, EOS, PAD, UNK handling
- **AutoLoad** - Auto-detect tokenizer type from path

**Sampling Strategies** (`internal/generate/sampling.go`):
- **Temperature** - Control randomness (0 = greedy)
- **Top-K** - Sample from top K tokens
- **Top-P (nucleus)** - Sample from smallest set with P cumulative probability
- **Min-P** - Filter tokens below P * max_prob threshold
- **Repetition Penalty** - Penalize repeated tokens
- **Frequency Penalty** - Penalize based on token frequency
- **Presence Penalty** - Penalize based on token presence
- **Configurable seed** - Reproducible sampling

**Text Generation** (`internal/generate/generator.go`):
- **TextGenerator** - High-level API for text generation
- **Streaming API** - Token-by-token generation with channels
- **Chat API** - Multi-turn conversation with templates
- **GenerateConfig** - Max tokens, min tokens, stop strings/tokens
- **GenerateResult** - Token, token ID, done flag, reason
- **KV-cache integration** - Efficient autoregressive generation
- **Echo prompt** - Optionally include prompt in output

**Multi-Output Autodiff** (`internal/autodiff/ops/`):
- **MultiOutputOperation** - Interface for ops with multiple outputs
- **BackwardMulti** - Compute gradients for multi-output ops
- **ChunkOp** - Fixed backward pass for tensor chunking
- **GatherOp** - Scatter-add gradient computation

**Public API** (`nn/`, `generate/`, `tokenizer/`, `loader/`):
- Complete public wrappers for all new types
- Type aliases for seamless internal/public integration
- Documentation with examples

### 📊 Testing

- **100+ new unit tests** across all LLM modules
- **Comprehensive sampling tests** - All strategies validated
- **Generator tests** - Streaming, stop conditions, chat
- **Tokenizer tests** - Encode/decode roundtrip, special tokens
- **0 golangci-lint issues**

### 🧪 Test Coverage

| Package | Tests | Status |
|---------|-------|--------|
| internal/nn (GQA, SwiGLU) | 35+ | ✅ |
| internal/tokenizer | 27 | ✅ |
| internal/generate | 17 | ✅ |
| internal/loader | 10+ | ✅ |
| internal/autodiff/ops | 20+ | ✅ |

### 🎯 What You Can Build Now

```go
import (
    "github.com/born-ml/born/generate"
    "github.com/born-ml/born/tokenizer"
    "github.com/born-ml/born/loader"
)

// Load tokenizer
tok, _ := tokenizer.NewTikTokenForModel("gpt-4")

// Load model
model, _ := loader.OpenModel("llama-7b.gguf")

// Create generator
gen := generate.NewTextGenerator(model, tok, generate.SamplingConfig{
    Temperature: 0.7,
    TopP:        0.9,
    TopK:        40,
})

// Generate text
result, _ := gen.Generate("Hello!", generate.GenerateConfig{MaxTokens: 100})

// Or stream tokens
stream, _ := gen.GenerateStream("Once upon", generate.GenerateConfig{MaxTokens: 50})
for chunk := range stream {
    fmt.Print(chunk.Token)
}

// Chat with templates
messages := []tokenizer.ChatMessage{
    {Role: "user", Content: "What is 2+2?"},
}
response, _ := gen.Chat(messages, tokenizer.NewChatMLTemplate(), config)
```

### 📈 Performance

| Feature | Benchmark |
|---------|-----------|
| GQA 32Q/8KV | 4x KV-cache memory savings |
| SwiGLU FFN | 2.7x expansion (vs 4x standard) |
| TikToken | ~1M tokens/sec encoding |
| Top-P sampling | O(n log n) sorting |

---

## [0.4.0] - 2025-12-01

### 🚀 Phase 4: Attention Mechanisms

Major release adding complete transformer architecture support! Build GPT, LLaMA, BERT, and modern LLM architectures with Born.

### ✨ Added

**Attention Mechanisms** (`internal/nn/`):
- **Scaled Dot-Product Attention (SDPA)** - Core attention with optional mask and dropout
- **Multi-Head Attention (MHA)** - Full implementation with WQ, WK, WV, WO projections
- **KV-Cache** - Efficient autoregressive generation (3.94x speedup for 100 tokens)

**Normalization Layers** (`internal/nn/`):
- **LayerNorm** - Classic layer normalization with learnable gamma/beta
- **RMSNorm** - Root Mean Square normalization (LLaMA style)

**Positional Encodings** (`internal/nn/`):
- **RoPE (Rotary Position Embedding)** - Used by LLaMA, Mistral, DeepSeek
- **ALiBi (Attention with Linear Biases)** - Used by BLOOM, MPT
- **Sinusoidal** - Original Transformer positional encoding
- **Learned** - Trainable position embeddings (GPT-2 style)

**Transformer Building Blocks** (`internal/nn/`):
- **TransformerBlock** - Complete transformer layer with:
  - Pre-Norm (LLaMA style) and Post-Norm (original) support
  - RMSNorm or LayerNorm selection
  - Configurable attention and FFN dimensions
- **FFN (Feed-Forward Network)** - SiLU activation (LLaMA style)
- **ForwardWithCache** - Efficient inference with KV-cache

**Tensor Operations** (`internal/tensor/`, `internal/backend/cpu/`):
- **BatchMatMul** - Native 3D/4D batched matrix multiplication
  - `[B, M, K] @ [B, K, N] → [B, M, N]` (3D)
  - `[B, H, M, K] @ [B, H, K, N] → [B, H, M, N]` (4D)
- Refactored SDPA to use BatchMatMul (-40% code)

### 🔧 Fixed

- **Scalar gradient broadcasting** - Fixed `reduceBroadcast` panic when propagating scalar gradients
- **Multi-dim Softmax backward** - Now supports 3D/4D tensors (not just 2D)

### 📊 Testing

- **70+ new unit tests** across attention modules
- **Comprehensive benchmarks** for all new components
- **0 golangci-lint issues**
- KV-Cache: 3.94x speedup verified
- Parameter counts verified (7.1M per transformer block, matching GPT-2)

### 🎯 What You Can Build Now

```go
import (
    "github.com/born-ml/born/nn"
    "github.com/born-ml/born/tensor"
)

// Create a transformer block (GPT-2 style)
config := nn.TransformerConfig{
    EmbedDim:   768,
    NumHeads:   12,
    FFNDim:     3072,
    NormFirst:  true,   // Pre-Norm (LLaMA)
    UseRMSNorm: true,   // RMSNorm (LLaMA)
    NormEps:    1e-5,
}
block := nn.NewTransformerBlock(config, backend)

// Forward pass
x := tensor.Randn[float32](tensor.Shape{1, 512, 768}, backend)
output := block.Forward(x, nil)

// With KV-Cache for generation
cache := nn.NewKVCache(1, 12, 2048, 64, backend)
for i := 0; i < 100; i++ {
    token := getNextToken()
    output := block.ForwardWithCache(token, cache)
}
```

### 📈 Performance

| Operation | Benchmark |
|-----------|-----------|
| SDPA (512 seq) | 89.2% coverage |
| MHA (768d/12h) | 2.3M params verified |
| KV-Cache (100 tokens) | **3.94x speedup** |
| TransformerBlock | ~7.1M params/block |
| RoPE (2048 seq) | Pre-computed cos/sin |

---

## [0.3.0] - 2025-11-30

### 🚀 Phase 2.5: Transformer Primitives + Public API

Major release adding essential operations for modern transformer architectures (LLaMA, Mistral, GPT), the HRM Model, and **31 type-safe public API operations**!

### ✨ Added

**Math Operations** (`internal/backend/cpu/math.go`, `internal/autodiff/ops/`):
- `Exp()` - Exponential function with gradient support
- `Sqrt()` - Square root with stable gradients
- `Rsqrt()` - Reciprocal square root (1/√x) for normalization layers
- `Cos()` - Cosine for RoPE (Rotary Position Embedding)
- `Sin()` - Sine for RoPE implementations

**Reduction Operations** (`internal/backend/cpu/reduce.go`):
- `SumDim(dim, keepDim)` - Sum along dimension with optional keepDim
- `MeanDim(dim, keepDim)` - Mean along dimension with optional keepDim
- Supports negative dimensions (-1 for last dimension)
- Broadcasting-aware for gradient computation

**Tensor Manipulation** (`internal/backend/cpu/manipulation.go`):
- `Cat(tensors, dim)` - Concatenate tensors along dimension
- `Chunk(n, dim)` - Split tensor into n equal chunks
- `Unsqueeze(dim)` - Add dimension of size 1
- `Squeeze(dim)` - Remove dimensions of size 1

**Indexing Operations** (`internal/backend/cpu/indexing.go`):
- `Gather(dim, index)` - Select elements using index tensor
- `Where(condition, x, y)` - Conditional element selection

**Neural Network Layers** (`internal/nn/`):
- **SiLU (Swish)** activation: `x * sigmoid(x)` with autodiff
- **RMSNorm** layer: Root Mean Square Normalization with learnable gamma
- **Embedding** layer: Token lookup table for NLP models

**Gradient Control** (`internal/autodiff/`):
- `NoGrad(func)` - Context manager to disable gradient recording (inference mode)
- `Detach()` - Break gradient chain while keeping tensor values

**Public API Operations** (`internal/tensor/ops_extended.go`, `tensor/`):

31 type-safe operations now available via `github.com/born-ml/born/tensor`:

- **Scalar (4)**: `MulScalar`, `AddScalar`, `SubScalar`, `DivScalar`
- **Math (6)**: `Log`, `Exp`, `Sqrt`, `Rsqrt`, `Cos`, `Sin`
- **Activation (1)**: `Softmax(dim)`
- **Comparison (12)**: `Greater`/`Gt`, `Lower`/`Lt`, `GreaterEqual`/`Ge`, `LowerEqual`/`Le`, `Equal`/`Eq`, `NotEqual`/`Ne`
- **Boolean (3)**: `Or`, `And`, `Not`
- **Reduction (2)**: `Sum`, `Argmax`
- **Type Conversion (6)**: `Int32`, `Int64`, `Float32`, `Float64`, `Uint8`, `Bool`
- **Shape (1)**: `Expand`

Example usage:
```go
import "github.com/born-ml/born/tensor"

x := tensor.Randn[float32](tensor.Shape{2, 3}, backend)
y := x.MulScalar(2.0)           // Scalar operations
mask := x.Greater(y)            // Comparison (returns Tensor[bool, B])
z := x.Softmax(-1)              // Activation
total := x.Sum()                // Reduction
i := x.Int32()                  // Type conversion
```

### 📊 Testing

- **112 new unit tests** added across all features
- **0 golangci-lint issues** (maintained strict quality standards)
- All autodiff operations validated with numerical gradient checking
- Comprehensive edge case coverage (negative dims, broadcasting, etc.)

### 🧪 Test Coverage

| Package | Coverage | Tests |
|---------|----------|-------|
| backend/cpu (math) | 79.0% | 23 |
| backend/cpu (reduce) | 80.2% | 17 |
| backend/cpu (manipulation) | - | 29 |
| backend/cpu (indexing) | - | 11 |
| autodiff/ops | 69.6% | - |
| nn (SiLU, RMSNorm, Embedding) | - | 18 |
| **Total Phase 2.5** | - | **112** |

### 🔧 Changed

- Updated `tensor.Backend` interface with new operations
- Extended `.golangci.yml` with exclusions for intentional patterns
- WebGPU backend stubs added for all new operations (CPU-only for now)

### 📦 New Files

```
internal/backend/cpu/
├── math.go              # Exp, Sqrt, Rsqrt, Cos, Sin
├── math_test.go         # 23 tests
├── reduce.go            # SumDim, MeanDim
├── reduce_test.go       # 17 tests
├── manipulation.go      # Cat, Chunk, Unsqueeze, Squeeze
├── indexing.go          # Gather, Where
└── indexing_test.go     # 11 tests

internal/autodiff/ops/
├── exp.go, sqrt.go, rsqrt.go, cos.go, sin.go
├── sumdim.go, meandim.go
├── silu.go
├── embedding.go
├── math_test.go
├── reduce_test.go
└── silu_test.go

internal/nn/
├── rmsnorm.go           # RMSNorm layer
├── rmsnorm_test.go      # 8 tests
├── embedding.go         # Embedding layer
├── embedding_test.go    # 8 tests
└── activation.go        # Added SiLU

internal/tensor/
└── ops_extended.go      # 31 public API wrappers (470 lines)

internal/backend/cpu/
├── scalar.go            # MulScalar, AddScalar, SubScalar, DivScalar
├── activation.go        # Softmax (n-dimensional, numerically stable)
├── comparison.go        # Greater, Lower, Equal, etc.
├── boolean.go           # Or, And, Not
├── conversion.go        # Cast for all dtype pairs
└── shape.go             # Expand with broadcasting

internal/backend/webgpu/
└── ops_extended.go      # Stubs + working Softmax
```

### 🎯 What This Enables

With Phase 2.5 primitives, Born can now support:

**Transformer Components:**
- ✅ **RoPE** (Rotary Position Embedding) - built from `Cos`, `Sin`, `Cat`
- ✅ **SwiGLU** activation - built from `Linear`, `SiLU`, `Chunk`
- ✅ **RMSNorm** - directly available as layer
- ✅ **Stablemax** (HRM) - built from `Where`, `SumDim`, `Gather`

**Modern LLM Architectures:**
- ✅ LLaMA (Meta)
- ✅ Mistral AI models
- ✅ GPT-style transformers
- ✅ **HRM** (Hierarchical Reasoning Model)

**Inference Capabilities:**
- ✅ Token embedding lookup
- ✅ Position encoding (RoPE)
- ✅ Layer normalization (RMSNorm)
- ✅ Modern activations (SiLU/Swish)
- ✅ Gradient control for inference (`NoGrad`, `Detach`)

### 🚀 Coming in v0.4.0

- Multi-head attention (MHA) layer
- Layer normalization variants
- More positional encodings (Absolute, Learned)
- KV-cache for efficient inference
- Linux/macOS WebGPU support

---

## [0.2.0] - 2025-11-28

### 🚀 Phase 2: WebGPU GPU Backend

Major release introducing GPU acceleration via WebGPU - the first production-ready Go ML framework with zero-CGO GPU support!

### ✨ Added

**WebGPU Backend** (`internal/backend/webgpu/`):
- **Zero-CGO GPU acceleration** via [go-webgpu](https://github.com/AlfredDobra662/webgpu) v0.1.0
- **WGSL compute shaders** for all tensor operations
- **Buffer pool** with size-based categorization for memory efficiency
- **Memory statistics** tracking (allocations, peak usage, pool hits/misses)
- **Graceful degradation** when wgpu_native.dll not available (panic recovery)

**GPU Operations**:
- Element-wise: `Add`, `Sub`, `Mul`, `Div`
- Matrix: `MatMul` (tiled algorithm, 16x16 workgroups)
- Shape: `Reshape`, `Transpose`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`

**CPU Backend Enhancements**:
- `Softmax` operation added
- Backend now implements full `tensor.Backend` interface

**Examples**:
- `examples/mnist-gpu/` - CPU vs WebGPU benchmark (~123x MatMul speedup)

**Documentation**:
- `docs/PHILOSOPHY.md` - Framework philosophy and design principles
- `docs/USE_CASES.md` - Real-world use cases and deployment scenarios
- Updated README with performance benchmarks

### 📊 Performance

**Benchmarks** (NVIDIA RTX GPU vs CPU):

| Operation | Size | CPU | WebGPU | Speedup |
|-----------|------|-----|--------|---------|
| MatMul | 1024×1024 | 847ms | 6.9ms | **123x** |
| MatMul | 512×512 | 105ms | 2.1ms | **50x** |
| MatMul | 256×256 | 13ms | 1.3ms | **10x** |
| Add | 1M elements | 1.2ms | 0.15ms | **8x** |

**MNIST MLP Inference** (batch=256):
- CPU: ~45ms/batch
- WebGPU: ~4.1ms/batch
- **Speedup: 10.9x**

### 🔧 Changed

- Build tags added for Windows-only WebGPU code (`//go:build windows`)
- `go.sum` now committed (was incorrectly in .gitignore)
- Updated all documentation for v0.2.0 milestone

### 🧪 Testing

- **13 new WebGPU operation tests** (ops_test.go)
- **7 buffer pool tests** (buffer_pool_test.go)
- **26 benchmark functions** for CPU vs GPU comparison
- All tests pass on Ubuntu, macOS, Windows
- WebGPU tests skip gracefully on systems without GPU support

### 📦 New Files

```
internal/backend/webgpu/
├── backend.go          # WebGPU backend initialization
├── ops.go              # Operation implementations
├── compute.go          # Compute pipeline management
├── shaders.go          # WGSL shader sources
├── buffer_pool.go      # GPU buffer pooling
├── *_test.go           # Tests and benchmarks
examples/mnist-gpu/
└── main.go             # GPU benchmark example
docs/
├── PHILOSOPHY.md       # Framework philosophy
└── USE_CASES.md        # Use cases
```

### ⚠️ Platform Support

- **Windows**: Full WebGPU support (requires wgpu_native.dll)
- **Linux/macOS**: CPU backend only (WebGPU builds skipped)
- WebGPU on Linux/macOS planned for future release

### 🚀 Coming in v0.3.0

- BatchNorm2D for training stability
- Dropout for regularization
- Model serialization (save/load)
- Linux WebGPU support via Vulkan
- ONNX model import

---

## [0.1.1] - 2025-11-17

### 🔥 Critical Hotfix

**BREAKING (but necessary)**: v0.1.0 had no usable public API! All packages were in `internal/` which cannot be imported by external projects. This hotfix adds proper public packages.

### ✨ Added

**Public API Packages**:
- `github.com/born-ml/born/tensor` - Type-safe tensor operations
- `github.com/born-ml/born/nn` - Neural network modules (Linear, Conv2D, MaxPool2D, etc.)
- `github.com/born-ml/born/optim` - Optimizers (SGD, Adam)
- `github.com/born-ml/born/backend/cpu` - CPU backend
- `github.com/born-ml/born/autodiff` - Automatic differentiation

**Documentation**:
- Comprehensive package documentation for pkg.go.dev
- Usage examples in each package
- API reference comments on all public types/functions

### 🔧 Changed

- Updated examples to use public API
- README updated with correct import paths

### 📦 Migration from v0.1.0

**Before (v0.1.0 - broken for external use)**:
```go
import "github.com/born-ml/born/internal/tensor"  // ❌ Cannot import!
```

**After (v0.1.1 - works!)**:
```go
import "github.com/born-ml/born/tensor"  // ✅ Public API
```

### 🧪 Testing

- All tests pass (internal tests unchanged)
- golangci-lint: 0 issues
- Public packages compile successfully
- Examples work with new imports

### 📊 Statistics

- +876 lines of public API code
- 9 new public files (doc.go + package wrappers)
- 5 public packages created

---

## [0.1.0] - 2025-11-17

### 🎉 Initial Release

First public release of Born ML Framework - a modern, type-safe machine learning framework for Go.

*Released in celebration of Go's 16th anniversary (November 10, 2009 - 2025)* 🎂

### ✨ Features

#### Core Framework
- **Tensor API** with generic type safety (`Tensor[T, B]`)
- **Shape validation** with NumPy-style broadcasting
- **Zero-copy operations** where possible
- **Device abstraction** (CPU, with GPU planned)

#### Automatic Differentiation
- **Tape-based reverse-mode autodiff**
- **Decorator pattern** (wraps any backend with autodiff)
- **Gradient tape** with operation recording
- **Backward pass** with efficient chain rule

#### Neural Network Modules
- **Linear** layers with Xavier initialization
- **Conv2D** (2D convolution) with im2col algorithm
- **MaxPool2D** (2D max pooling)
- **Activation functions**: ReLU, Sigmoid, Tanh
- **Loss functions**: CrossEntropyLoss with numerical stability
- **Parameter management** for optimization

#### Optimizers
- **SGD** with momentum
- **Adam** with bias correction

#### Backend
- **CPU Backend** with optimized implementations
- Im2col algorithm for efficient convolutions
- Float32 and Float64 support
- Batch processing

### 📊 Validated Performance

**MNIST Classification**:
- MLP (2-layer): **97.44%** accuracy (101,770 parameters)
- CNN (LeNet-5): **98.18%** accuracy (44,426 parameters)

### 📚 Examples

- **MNIST MLP** - Fully connected network example
- **MNIST CNN** - Convolutional neural network example (LeNet-5 style)

### 🧪 Testing

- **33 new tests** for Conv2D and MaxPool2D
- **Numerical gradient verification** for all autodiff operations
- **Integration tests** for end-to-end workflows
- **Overall test coverage**: 53.7%

### 🏗️ Architecture

**Zero External Dependencies** (core framework):
- Pure Go implementation
- Standard library only
- Type-safe generics (Go 1.25+)

### 📖 Documentation

- Comprehensive README with quickstart
- Example code with detailed comments
- API documentation in code

### 🔧 Technical Highlights

1. **ReshapeOp** - Enables gradient flow through reshape operations (critical for Conv2D bias)
2. **TransposeOp** - Proper gradient propagation for matrix transposes
3. **Im2col Algorithm** - Efficient convolution via matrix multiplication
4. **Max Index Tracking** - For MaxPool2D gradient routing
5. **Xavier Initialization** - For stable training

### ⚠️ Known Limitations

- CPU-only (GPU support planned for v0.2.0)
- No model save/load yet
- Limited data augmentation
- No distributed training

### 🚀 Coming in v0.2.0

- BatchNorm2D for training stability
- Dropout for regularization
- Model serialization
- Data augmentation
- GPU backend (CUDA)

---

## Release Notes

### Breaking Changes
None (initial release)

### Migration Guide
N/A (initial release)

### Contributors
- Claude Code AI Assistant
- Born ML Project Team

---

[0.7.10]: https://github.com/born-ml/born/releases/tag/v0.7.10
[0.7.9]: https://github.com/born-ml/born/releases/tag/v0.7.9
[0.7.8]: https://github.com/born-ml/born/releases/tag/v0.7.8
[0.7.15]: https://github.com/born-ml/born/releases/tag/v0.7.15
[0.7.14]: https://github.com/born-ml/born/releases/tag/v0.7.14
[0.7.13]: https://github.com/born-ml/born/releases/tag/v0.7.13
[0.7.12]: https://github.com/born-ml/born/releases/tag/v0.7.12
[0.7.11]: https://github.com/born-ml/born/releases/tag/v0.7.11
[0.7.10]: https://github.com/born-ml/born/releases/tag/v0.7.10
[0.7.9]: https://github.com/born-ml/born/releases/tag/v0.7.9
[0.7.8]: https://github.com/born-ml/born/releases/tag/v0.7.8
[0.7.7]: https://github.com/born-ml/born/releases/tag/v0.7.7
[0.7.6]: https://github.com/born-ml/born/releases/tag/v0.7.6
[0.7.5]: https://github.com/born-ml/born/releases/tag/v0.7.5
[0.7.4]: https://github.com/born-ml/born/releases/tag/v0.7.4
[0.7.3]: https://github.com/born-ml/born/releases/tag/v0.7.3
[0.7.2]: https://github.com/born-ml/born/releases/tag/v0.7.2
[0.7.1]: https://github.com/born-ml/born/releases/tag/v0.7.1
[0.7.0]: https://github.com/born-ml/born/releases/tag/v0.7.0
[0.6.0]: https://github.com/born-ml/born/releases/tag/v0.6.0
[0.5.5]: https://github.com/born-ml/born/releases/tag/v0.5.5
[0.5.4]: https://github.com/born-ml/born/releases/tag/v0.5.4
[0.5.3]: https://github.com/born-ml/born/releases/tag/v0.5.3
[0.5.2]: https://github.com/born-ml/born/releases/tag/v0.5.2
[0.5.1]: https://github.com/born-ml/born/releases/tag/v0.5.1
[0.5.0]: https://github.com/born-ml/born/releases/tag/v0.5.0
[0.4.0]: https://github.com/born-ml/born/releases/tag/v0.4.0
[0.3.0]: https://github.com/born-ml/born/releases/tag/v0.3.0
[0.2.0]: https://github.com/born-ml/born/releases/tag/v0.2.0
[0.1.1]: https://github.com/born-ml/born/releases/tag/v0.1.1
[0.1.0]: https://github.com/born-ml/born/releases/tag/v0.1.0
