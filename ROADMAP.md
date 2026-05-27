# Born ML Framework - Development Roadmap

> **Mission**: De-facto standard ML framework for Go — training and inference in a single binary
> **Architecture**: Burn-inspired (Rust), Go-idiomatic API
> **Philosophy**: Correctness → Performance → Scale

**Last Updated**: 2026-05-27 | **Current Version**: v0.9.1 | **Go**: 1.26+ | **Strategy**: Core → GPU → Models → Performance → Scale → Production → v1.0 LTS

---

## 🎯 Vision

Born = **de-facto standard ML for Go ecosystem.** Any Go service adds ML via `go get github.com/born-ml/born` — training and inference in the same binary, same toolchain, zero external dependencies.

Not a PyTorch replacement (different niche). Not an academic tool (production-first). Born is what `database/sql` is for databases: the standard Go-native interface for ML.

### Why Born

- **One stack**: train and deploy in Go, no Python sidecar, no FFI
- **One binary**: `go build` → GPU-ready binary, no Docker, no CUDA install
- **One language**: research experiments and production code share the same codebase
- **Type-safe**: `Tensor[float32, *webgpu.Backend]` — errors at compile-time, not runtime

### What Born Does Today (v0.9.0)

- ✅ Train models from scratch on CPU and GPU (HRM model validated)
- ✅ LLaMA inference from GGUF files (TinyLlama 1.1B verified)
- ✅ ONNX import (49 operators)
- ✅ GPU acceleration: WebGPU/Vulkan via gogpu/wgpu (pure Go, zero CGO)
- ✅ AVX2 SIMD, cache-tiled MatMul, parallel BatchMatMul
- ✅ GPU batched dispatch (50→1 submits per forward pass)
- ✅ Autodiff: all backward ops via forward composition (Burn pattern)

---

## 🚀 Version Strategy

### Philosophy: Validate → Iterate → Stabilize → LTS

```
v0.1.0 (INITIAL RELEASE) ✅ RELEASED (2025-11-17)
       ↓ (GPU backend development)
v0.2.0 (WebGPU GPU Backend) ✅ RELEASED (2025-11-28)
       ↓ (transformer primitives)
v0.3.0 (Transformer Primitives) ✅ RELEASED (2025-11-30)
       ↓ (attention layers)
v0.4.0 (Attention Mechanisms) ✅ RELEASED (2025-12-01)
       ↓ (LLM support)
v0.5.0 (LLM Support) ✅ RELEASED (2025-12-01)
       ↓ (model serialization)
v0.5.4 (Model Serialization) ✅ RELEASED (2025-12-03)
       ↓ (WebGPU performance)
v0.5.5 (WebGPU Performance) ✅ RELEASED (2025-12-03)
       ↓ (ONNX import + lazy GPU)
v0.6.0 (ONNX Import + Lazy GPU Mode) ✅ RELEASED (2025-12-04)
       ↓ (inference optimization)
v0.7.0 (Flash Attention, Speculative Decoding, GGUF) ✅ RELEASED (2025-12-10)
       ↓ (code quality)
v0.7.1 (Code Quality - Burn Patterns) ✅ RELEASED (2025-12-16)
       ↓ (dependency updates)
v0.7.3 (Dependencies Update) ✅ RELEASED (2025-12-27)
       ↓ (ARM64 enhancements, Linear bias option, API improvements, gogpu integration)
v0.7.8 (GoGPU Ecosystem Integration Phase 1) ✅ RELEASED (2026-01-29)
       ↓ (dependency updates)
v0.7.10 (ARM64 Callback Fix) ✅ RELEASED (2026-02-18)
       ↓ (callback reliability)
v0.7.11 (Crosscall2 Callback Integration) ✅ RELEASED (2026-02-27)
       ↓ (FFI hardening)
v0.7.12 (FFI Hardening & Library Loading) ✅ RELEASED (2026-02-27)
       ↓ (ABI compliance fixes)
v0.7.13 (ABI Compliance Fixes) ✅ RELEASED (2026-03-02)
       ↓ (first community contribution)
v0.7.14 (ONNX Equal — Community PR) ✅ RELEASED (2026-03-04)
       ↓ (erf operator — community contribution)
v0.7.15 (Erf Operator — Community PR) ✅ RELEASED (2026-04-07)
       ↓ (community contributions, bugfixes, ONNX ops)
v0.7.16 (Community PRs, ONNX 49 ops, Bugfixes) ✅ RELEASED (2026-04-10)
       ↓ (WebGPU migration to pure Go)
v0.8.0 (GoGPU/wgpu Migration — Pure Go, Zero CGO) ✅ RELEASED (2026-04-26)
       ↓ (LLaMA inference, GGUF loading, reproducibility)
v0.8.1 (LLaMA Inference, GGUF Model Loading) ✅ RELEASED (2026-05-15)
       ↓ (tokenizer bug fix)
v0.8.2 (Tokenizer Fix, Backward Ops Migration, Scalar Gradient Fix) ✅ RELEASED (2026-05-16)
       ↓ (GPU scatter-add shaders)
v0.8.3 (GPU SelectAdd/ScatterAdd Shaders) ✅ RELEASED (2026-05-16)
       ↓ (CPU parallel + GPU batching + SIMD)
v0.9.0 (CPU Parallel, GPU Batching, AVX2 SIMD) ✅ RELEASED (2026-05-17)
       ↓ (distributed multi-GPU)
v0.10.0 (Distributed Multi-GPU, Data Parallelism) → Q3 2026
       ↓ (quantization, production serving)
v0.11.0 (Quantization, INT8/GPTQ, Model Zoo) → Q4 2026
       ↓ (multi-node, production serving)
v0.12.0 (Multi-Node, PD-Disaggregation, Production Serving) → Q1 2027
       ↓ (API stabilization)
v1.0.0 LTS → API freeze, stability guarantees
```

### Critical Milestones

**v0.1.0** = Initial Release with Core Features ✅ RELEASED
- Tensor API with type safety
- Tape-based autodiff
- NN modules: Linear, Conv2D, MaxPool2D
- Optimizers: SGD, Adam
- CPU backend (im2col convolutions)
- **Validated**: MNIST MLP 97.44%, CNN 98.18%

**v0.2.0** = WebGPU GPU Backend ✅ RELEASED
- Zero-CGO GPU acceleration (originally via go-webgpu, migrated to gogpu/wgpu in v0.8.0)
- GPU operations: MatMul, Add, Sub, Mul, Div, Transpose
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Buffer pool for memory efficiency
- **Performance**: 123x MatMul speedup, 10.9x inference speedup
- Windows support (Linux/macOS planned for v0.5.0)

**v0.3.0** = Transformer Primitives + Public API ✅ RELEASED
- Math ops: Exp, Sqrt, Rsqrt, Cos, Sin, Log
- Reductions: SumDim, MeanDim (with keepDim), Sum, Argmax
- Manipulation: Cat, Chunk, Unsqueeze, Squeeze, Expand
- Indexing: Gather, Where
- Layers: SiLU, RMSNorm, Embedding, Softmax
- Gradient control: NoGrad, Detach
- **31 public API operations**: MulScalar, Greater/Gt, Int32, etc.
- **Enables**: LLaMA, Mistral, GPT, HRM architectures

**v0.4.0** = Attention Mechanisms ✅ RELEASED
- Multi-Head Attention (MHA) with Q, K, V projections
- Scaled Dot-Product Attention (SDPA)
- KV-Cache for efficient inference (3.94x speedup)
- Layer Normalization variants
- Positional Encodings: RoPE, ALiBi, Sinusoidal, Learned
- Transformer Block with FFN
- BatchMatMul for 3D/4D tensors

**v0.5.0** = LLM Support ✅ RELEASED
- Grouped Query Attention (GQA) - LLaMA 2/3, Mistral
- SwiGLU FFN with GLU variants (GeGLU, ReGLU)
- GGUF Model Loader (v3 format)
- Tokenizers: TikToken, BPE, HuggingFace
- Sampling: Temperature, Top-K, Top-P, Min-P, Repetition Penalty
- TextGenerator with streaming API
- **Production-ready LLM inference pipeline**

**v0.5.4** = Model Serialization ✅ RELEASED
- Born Native Format v2 (.born) with SHA-256 checksum
- Security validation (offset overlap, bounds check)
- Memory-mapped reader for 70GB+ models
- Checkpoint API for training resume
- SafeTensors export for HuggingFace

**v0.5.5** = WebGPU Performance ✅ RELEASED
- Multi-dimensional Transpose on GPU (3D/4D/5D/6D)
- Expand (broadcasting) on GPU
- ~60x speedup for attention operations
- Eliminated CPU fallback for transformer training

**v0.6.0** = ONNX Import + Lazy GPU Mode ✅ RELEASED
- ONNX model import (parser, loader, 30+ operators)
- Lazy GPU evaluation (GPU-resident tensors)
- Command batching (~90s → <5s/step for training)
- GPU-to-GPU copy (no CPU round-trips)
- GPU memory management (automatic cleanup)
- 50+ raw tensor operations

**v0.7.0** = Inference Optimization ✅ RELEASED
- Flash Attention 2 (O(N) memory, 2x+ speedup, 128K+ context)
- Speculative Decoding (2-4x inference speedup)
- GGUF Import (llama.cpp ecosystem, K-quant dequantization)
- WebGPU WGSL Flash Attention shader
- Online softmax for numerical stability

**v0.7.1** = Code Quality Refactoring ✅ CURRENT
- Burn framework patterns applied (Issue #14)
- Flash Attention CPU complexity: 111 → <30
- Pre-slice bounds elimination
- Stride specialization for auto-vectorization
- New `internal/parallel` package
- Extended Backend interface with backward methods

**v0.8.0** = GoGPU/wgpu Migration ✅ RELEASED (2026-04-26)
- WebGPU backend migrated from go-webgpu (Rust FFI) to gogpu/wgpu (pure Go)
- Zero CGO, zero runtime deps — `go build` produces GPU-ready binary
- Vulkan primary compute backend

**v0.8.1** = LLaMA Inference & GGUF Loading ✅ RELEASED (2026-05-15)
- `models/llama`: Full LLaMA model (GQA, RoPE, SwiGLU FFN, KV cache)
- `loader`: Public GGUF/SafeTensors loading API
- `LoadGGUF`: Auto-dequantize Q4_K, Q5_K, Q6_K, Q8_0, F16, F32
- Injectable attention (`WithAttentionFunc`) for research experiments
- `nn.SetSeed()` for reproducible weight initialization
- Tested: TinyLlama 1.1B Q8_0 — "Paris" top-1 for "The capital of France is"
- Fixed: RoPE rotate-half, GGML naming, Q4_K/Q5_K scales, F16 subnormals, tied embeddings

**v0.8.2** = Tokenizer Fix + Backward Ops Migration ✅ RELEASED (2026-05-16)
- Fixed: HF tokenizer normalizer (SentencePiece Prepend+Replace). PPL 1887 → 230.
- Fixed: Scalar ops on gradient tape. Embedding weights were getting zero gradients.
- Refactored: 7 backward ops migrated from CPU-fallback to forward composition (ADR-009).
- Added: SelectAdd, ScatterAdd backend ops for Embedding/Gather backward.

**v0.8.3** = GPU SelectAdd/ScatterAdd Shaders ✅ RELEASED (2026-05-16)
- WGSL compute shaders for scatter-add (no f32 atomics)
- 27K readbacks → 1 GPU dispatch. HRM: minutes → seconds

**v0.9.0** = CPU Parallel + GPU Batching + AVX2 SIMD ✅ RELEASED (2026-05-17)
- CPU: parallel BatchMatMul (2-4x), cache-tiled blocked MatMul (3-5x), AVX2 SIMD (3.5x)
- GPU: batched dispatch (50→1 submits), all resource keepAlive fixes
- Go 1.26 minimum. Combined CPU: ~20-70x for large batch MatMul.
- HRM GPU training validated: 160+ steps, loss converging

**v0.10.0** = Distributed Multi-GPU → Q3 2026 (ADR-011)
- Device struct with Index (CUDA:0, CUDA:1, ...)
- Ring AllReduce for single-node data parallelism (port from Burn)
- Distributed data loader (automatic batch sharding)
- Device registry + capability reporting

**v0.11.0** = Quantization + Model Zoo → Q4 2026
- INT8 symmetric quantization
- GPTQ/AWQ 4-bit
- Pre-trained model loading
- PagedAttention for LLM serving

**v0.12.0** = Multi-Node + Production Serving → Q1 2027
- Multi-node orchestrator (gRPC, port from Burn)
- PD-disaggregation (prefill/decode split)
- Continuous batching
- Fault tolerance + checkpointing

**v1.0.0** = LTS
- API freeze (community feedback period)
- Stable API guarantees, semantic versioning
- Production hardening, comprehensive documentation
- Goal: any Go service adds ML via `go get github.com/born-ml/born`

---

## 📊 Current Status (v0.7.1)

**Phase**: 🚀 Code Quality + Community Contributions
**Focus**: Maintainability & Developer Experience
**Quality**: Production-ready

**What Works**:
- ✅ Tensor API (creation, operations, broadcasting)
- ✅ Shape validation with NumPy-style rules
- ✅ Zero-copy operations where possible
- ✅ Tape-based reverse-mode autodiff
- ✅ Gradient computation with chain rule
- ✅ NN Modules: Linear, Conv2D, MaxPool2D
- ✅ Activations: ReLU, Sigmoid, Tanh, Softmax, **SiLU**
- ✅ Normalization: **RMSNorm**
- ✅ Embeddings: **Token lookup tables**
- ✅ Loss: CrossEntropyLoss (numerically stable)
- ✅ Optimizers: SGD (momentum), Adam (bias correction)
- ✅ CPU Backend with im2col algorithm
- ✅ **WebGPU Backend** with zero-CGO GPU acceleration
- ✅ GPU Operations: MatMul, Add, Sub, Mul, Div, Transpose
- ✅ **Math ops**: Exp, Sqrt, Rsqrt, Cos, Sin, Log
- ✅ **Reductions**: SumDim, MeanDim (keepDim), Sum, Argmax
- ✅ **Manipulation**: Cat, Chunk, Unsqueeze, Squeeze, Expand
- ✅ **Indexing**: Gather, Where
- ✅ **Gradient control**: NoGrad, Detach
- ✅ **31 Public API operations**: MulScalar, Greater/Gt, Int32, etc.
- ✅ Buffer pool for GPU memory management
- ✅ Batch processing
- ✅ Float32/Float64 support
- ✅ **ONNX Import**: Parser, loader, 30+ operators (v0.6.0)
- ✅ **Lazy GPU Mode**: GPU-resident tensors, deferred CPU transfer (v0.6.0)
- ✅ **Command Batching**: Reduced GPU sync overhead (v0.6.0)
- ✅ **50+ Raw Ops**: Argmax, TopK, type conversions, broadcasting (v0.6.0)
- ✅ **Flash Attention 2**: O(N) memory, WebGPU shader, 2x+ speedup (v0.7.0)
- ✅ **Speculative Decoding**: 2-4x inference speedup (v0.7.0)
- ✅ **GGUF Import**: llama.cpp models, K-quant dequantization (v0.7.0)
- ✅ **Burn Patterns**: Pre-slicing, stride specialization (v0.7.1)
- ✅ **Parallel Utils**: `internal/parallel` package (v0.7.1)

**Performance** (v0.2.0):
- ✅ **MatMul 1024×1024**: 123x speedup (GPU vs CPU)
- ✅ **MNIST Inference**: 10.9x speedup (batch=256)
- ✅ **Throughput**: 62,000+ samples/sec (GPU)

**Performance** (v0.6.0 - Lazy GPU Mode):
- ✅ **Training Step**: ~90s → <5s (~18x speedup)
- ✅ **GPU Submits**: ~200 → 1-2 per chain (~100x reduction)
- ✅ **GPU Memory**: Automatic cleanup via finalizers

**Transformer Support** (v0.3.0):
- ✅ **LLaMA** architectures ready
- ✅ **Mistral AI** models ready
- ✅ **GPT-style** transformers ready
- ✅ **HRM** (Hierarchical Reasoning Model) ready

**Validation**:
- ✅ **MNIST MLP**: 97.44% accuracy (101,770 params)
- ✅ **MNIST CNN**: 98.18% accuracy (44,426 params)
- ✅ **Test Coverage**: 53.7%
- ✅ **golangci-lint**: 0 issues
- ✅ **Race Detector**: Clean

**Architecture**:
- ✅ **Zero CGO** (including GPU backend!)
- ✅ **Pure Go** implementation
- ✅ **Type-safe** generics (Go 1.25+)
- ✅ **Backend abstraction** (CPU + WebGPU)
- ✅ **Decorator pattern** (autodiff wraps any backend)

**Platform Support**:
- ✅ **Windows**: Full GPU support (WebGPU)
- ✅ **Linux/macOS**: CPU backend (GPU in v0.4.0)

**History**: See [CHANGELOG.md](CHANGELOG.md) for complete release notes

---

## 📅 Roadmap

### **v0.2.0 - WebGPU GPU Backend** ✅ RELEASED (2025-11-28)

**Goal**: GPU acceleration without CGO dependencies

**Delivered**:
- ✅ WebGPU backend (zero-CGO, migrated to gogpu/wgpu in v0.8.0)
- ✅ GPU operations: MatMul, Add, Sub, Mul, Div, Transpose
- ✅ Activations: ReLU, Sigmoid, Tanh, Softmax
- ✅ Buffer pool for memory efficiency
- ✅ Memory statistics tracking
- ✅ Graceful degradation on systems without GPU
- ✅ 123x MatMul speedup, 10.9x inference speedup
- ✅ Windows support (Linux/macOS in v0.5.0)

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

### **v0.3.0 - Transformer Primitives + Public API** ✅ RELEASED (2025-11-30)

**Goal**: Enable modern transformer architectures (LLaMA, Mistral, GPT, HRM) + type-safe public API

**Delivered**:
- ✅ Math operations: Exp, Sqrt, Rsqrt, Cos, Sin, Log
- ✅ Reductions: SumDim, MeanDim (with keepDim), Sum, Argmax
- ✅ Manipulation: Cat, Chunk, Unsqueeze, Squeeze, Expand
- ✅ Indexing: Gather, Where
- ✅ Layers: SiLU, RMSNorm, Embedding, Softmax
- ✅ Gradient control: NoGrad, Detach
- ✅ **31 public API operations**:
  - Scalar: MulScalar, AddScalar, SubScalar, DivScalar
  - Comparison: Greater/Gt, Lower/Lt, Equal/Eq, etc.
  - Boolean: Or, And, Not
  - Type conversion: Int32, Int64, Float32, Float64, Uint8, Bool
- ✅ 112 new unit tests, 0 linter issues
- ✅ All autodiff operations with numerical gradient validation

**Impact**:
- ✅ LLaMA architectures fully supported
- ✅ Mistral AI models supported
- ✅ GPT-style transformers supported
- ✅ HRM (Hierarchical Reasoning Model) ready
- ✅ External projects can use full typed tensor API

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

### **v0.7.0 - Inference Optimization** ✅ RELEASED (2025-12-10)

**Goal**: State-of-the-art inference performance

**Delivered**:
- ✅ **Flash Attention 2** - O(N) memory, WebGPU WGSL shader, 2x+ speedup on long sequences
- ✅ **Speculative Decoding** - Draft model + verification, 2-4x inference speedup
- ✅ **GGUF Import** - llama.cpp format, K-quant dequantization (Q4_K, Q5_K, Q6_K, Q8_0)
- ✅ **Online Softmax** - Numerical stability for long sequences
- ✅ **128K+ Context** - Extended context length support
- ✅ 0 linter issues, all tests passing

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

### **v0.7.1 - Code Quality Refactoring** ✅ RELEASED (2025-12-16)

**Goal**: Improved code maintainability via Burn framework patterns

**Delivered**:
- ✅ **Pre-Slice Bounds Elimination** - Conv2D/MaxPool2D optimization
- ✅ **Stride Specialization** - Fast paths for common stride=1, padding=0 case
- ✅ **Flash Attention Refactor** - Complexity 111 → <30
- ✅ **Autodiff Orchestration** - Separated orchestration from computation
- ✅ **Parallel Utilities** - New `internal/parallel` package
- ✅ **Extended Backend Interface** - Backward operation methods
- ✅ 0 linter issues, all tests passing

**Community**: Thanks to [@marcelloh](https://github.com/marcelloh) for Issue #14!

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

### **v0.8.0 - Quantization & Efficiency** (February 2026)

**Goal**: Production-ready quantization and developer experience

**Duration**: ~8 weeks

**Key Features**:
1. **Post-Training Quantization** (CRITICAL)
   - GPTQ algorithm (4-bit, 8-bit)
   - AWQ (Activation-aware quantization)
   - 4x model size reduction
   - Minimal accuracy loss (<1%)

2. **KV Cache Compression** (CRITICAL)
   - 4-bit/8-bit quantized KV cache
   - 2-4x memory savings
   - On-the-fly compression/decompression

3. **Jupyter Kernel** (HIGH)
   - Interactive ML development
   - go-jupyter integration
   - Rich output (plots, tensors)

4. **Model Zoo** (HIGH)
   - 10+ pre-trained models
   - Automatic download and caching
   - Version management

**Target**: February 2026

---

### **v0.9.0 - Production Serving** (March 2026)

**Goal**: vLLM-class serving infrastructure

**Duration**: ~10 weeks

**Key Features**:
1. **PagedAttention** (CRITICAL)
   - OS-style paged KV cache
   - >90% GPU utilization
   - Near-zero memory waste

2. **Continuous Batching** (CRITICAL)
   - Iteration-level request scheduling
   - 10-23x throughput improvement
   - Dynamic batch size adjustment

3. **Kernel Fusion** (CRITICAL)
   - Automatic operation graph optimization
   - 30-50% speedup
   - Burn-style fusion patterns

4. **MoE Support** (HIGH)
   - Mixture of Experts architecture
   - Mixtral, DeepSeek models
   - Expert routing and load balancing

5. **API Server** (HIGH)
   - OpenAI-compatible REST API
   - Streaming responses
   - Multi-model serving

**Target**: March 2026

---

### **v0.10.0 - Scale & Stability** (April 2026)

**Goal**: Enterprise-ready scale and production hardening

**Duration**: ~11 weeks

**Key Features**:
1. **Multi-GPU Data Parallelism** (CRITICAL)
   - Pure Go implementation (no NCCL)
   - Gradient all-reduce
   - Linear scaling to 8+ GPUs

2. **CPU SIMD Optimization** (HIGH)
   - AVX2 (x86-64) optimized kernels
   - Neon (ARM64/Apple Silicon)
   - 10-50x CPU speedup

3. **Gradient Checkpointing** (HIGH)
   - 80% memory reduction
   - Recompute activations during backward
   - Automatic checkpointing strategy

4. **Training Dashboard** (MEDIUM)
   - Terminal-based TUI
   - Live loss/accuracy curves
   - GPU/CPU utilization monitoring

5. **Comprehensive Documentation** (MEDIUM)
   - Complete API reference
   - 5+ tutorials
   - 10+ examples
   - Migration guides

**Target**: April 2026

---

### **v1.0.0 - Long-Term Support Release** (After API Freeze)

**Goal**: Production LTS with stability guarantees

**Prerequisites** (STRICT):
- v0.10.0 stable and battle-tested
- API freeze period (2-4 weeks community feedback)
- Zero critical bugs
- Complete documentation
- Multiple production deployments

**LTS Guarantees**:
- ✅ API stability (no breaking changes in v1.x.x)
- ✅ Long-term support (3+ years)
- ✅ Semantic versioning strictly followed
- ✅ Security updates and bug fixes
- ✅ Performance improvements (non-breaking)
- ✅ Backward compatibility within v1.x.x

**Success Metrics**:
- Production deployments in multiple companies
- >1000 GitHub stars
- Active community contributions
- Complete documentation and tutorials
- Benchmark suite vs PyTorch/TensorFlow

---

## 🔬 Development Principles

### Code Quality
- **Test Coverage**: >70% for core modules
- **Linting**: 0 issues (34+ linters via golangci-lint)
- **Race Detector**: Always clean
- **Benchmarks**: Performance regression tests
- **Documentation**: Every public API documented

### Architecture Decisions
- **Zero Dependencies**: Core framework stays pure Go
- **Backend Abstraction**: Easy to add GPU/TPU/WebGPU
- **Type Safety**: Generics for compile-time checks
- **Memory Safety**: No `unsafe` in tensor operations
- **Numerical Stability**: Proven algorithms (LogSumExp, etc.)

### Performance Philosophy
1. **Correctness First**: Get it right, then make it fast
2. **Profile Before Optimize**: Data-driven optimization
3. **Benchmark Everything**: Regression detection
4. **Real-World Validation**: MNIST, ImageNet, etc.

### Community Driven
- GitHub Issues for feature requests
- Discussions for design decisions
- Pull requests welcome
- Transparent roadmap updates

---

## 📚 Resources

**Inspiration**:
- PyTorch: https://pytorch.org/ (API design)
- Burn (Rust): https://github.com/tracel-ai/burn (architecture)
- Gorgonia: https://github.com/gorgonia/gorgonia (Go ML)

**Documentation**:
- README.md - Quick start
- CONTRIBUTING.md - How to contribute
- docs/guides/ - User guides
- CHANGELOG.md - Release history

**Development**:
- Go 1.25+ required
- golangci-lint for quality
- gogpu/wgpu for WebGPU (pure Go, zero runtime dependencies)

---

## 📞 Support

**Bug Reports**:
- GitHub Issues: https://github.com/born-ml/born/issues
- Security: See SECURITY.md

**Questions**:
- GitHub Discussions: https://github.com/born-ml/born/discussions
- Stack Overflow: Tag `born-ml`

**Contributing**:
- See CONTRIBUTING.md
- Good first issues labeled
- Code reviews welcome

---

## 🚦 Feature Request Process

1. **Open GitHub Issue** with feature proposal
2. **Community Discussion** (upvotes, comments)
3. **Roadmap Review** (quarterly)
4. **Prioritization** based on:
   - Community demand (upvotes)
   - Implementation complexity
   - Alignment with vision
   - Maintainability

5. **Assignment** to milestone
6. **Implementation** with tests + docs
7. **Release** with migration guide if needed

---

## 🔒 Stability Guarantees

### v0.x.x (Current)
- ⚠️ API may change between minor versions
- ⚠️ Deprecation warnings provided
- ✅ Migration guides for breaking changes
- ✅ Semantic versioning followed

### v1.x.x (Future LTS)
- ✅ API stability guaranteed
- ✅ No breaking changes in minor/patch
- ✅ 3+ years support
- ✅ Security updates
- ✅ Performance improvements (non-breaking)

---

## 🎯 Success Metrics

**Technical**:
- [ ] >70% test coverage
- [ ] <1ms tensor creation (CPU)
- [ ] <100ms MNIST training/epoch (GPU)
- [ ] Zero memory leaks (validated)
- [ ] 0 golangci-lint issues

**Community**:
- [ ] >1000 GitHub stars
- [ ] >10 contributors
- [ ] >5 production deployments
- [ ] >100 weekly downloads (pkg.go.dev)

**Documentation**:
- [ ] Complete API documentation
- [ ] 10+ tutorials
- [ ] 5+ example projects
- [ ] Migration guides for all versions

---

*Version 5.0 (2025-12-16)*
*Current: v0.7.1 (Code Quality) | Next: v0.8.0 (Quantization, Feb 2026) | Target: v1.0.0 LTS (After API Freeze)*
