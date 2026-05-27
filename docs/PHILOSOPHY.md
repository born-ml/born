# Born ML Framework - Philosophy & Design Principles

**Status**: Living Document
**Last Updated**: 2026-05-27

---

## Core Philosophy: "Born Production-Ready"

Born ML Framework follows a **production-first** philosophy where models are "born" ready for deployment, not as an afterthought.

### Key Principles

#### 1. Zero Dependencies (Pure Go)

```go
// ✅ Born: No CGO, no external dependencies
import "github.com/born-ml/born/tensor"

// ❌ Others: CGO dependencies
// OpenXLA, CUDA libraries, Python runtime
```

**Why it matters:**
- **Trivial cross-compilation**: GOOS/GOARCH just works
- **Single binary deployment**: No containers required
- **Fast cold start**: < 100ms startup time
- **Small memory footprint**: Ideal for edge devices

#### 2. Type Safety First

```go
// Compile-time guarantees via generics
type Tensor[T DType, B Backend] struct

// Invalid operations caught at compile-time, not runtime!
```

**Advantages:**
- Entire class of bugs eliminated before runtime
- Better IDE autocomplete and refactoring
- Self-documenting APIs
- Modern Go 1.25+ idioms

#### 3. Decorator Pattern for Composability

Inspired by Burn (Rust), Born uses decorator composition:

```go
base := cpu.New()                    // Base backend
withAutodiff := autodiff.New(base)   // Add autodiff capability
optimized := fusion.New(withAutodiff) // Add kernel fusion
```

**Benefits:**
- Swappable backends (CPU, CUDA, Vulkan, WebGPU)
- Layered functionality (autodiff, fusion, quantization)
- Testable components
- Flexible architecture

##### 4. Production-First, Research-Capable

```
Traditional ML workflow:
Research (Python) → Rewrite (Go/C++) → Production
                   ↑ Lost details, bugs introduced

Born workflow:
Research (Go) → Production (Go)
             ↑ Same codebase, same behavior!
```

**Use cases:**
- ✅ Go microservices + ML inference
- ✅ Edge deployment (IoT, embedded)
- ✅ Cloud-native ML serving (Kubernetes)
- ✅ ML Systems research (distributed learning, federated ML)
- ✅ Integration with Go ecosystem

#### 5. Injectable Attention for Research

Born separates the *model architecture* from the *attention implementation*. Pass a custom attention function at `LoadGGUF` time to experiment with sparse, linear, or custom attention without touching the model weights:

```go
model, _ := llama.LoadGGUF("model.gguf", backend,
    llama.WithAttention(myCustomAttentionFn),
)
```

This keeps research modifications local and reproducible.

#### 6. Reproducibility via nn.SetSeed()

`nn.SetSeed(seed)` sets the global random seed before weight initialization, ensuring identical starting conditions across runs — critical for debugging and fair comparisons.

#### 7. Backward = Forward Ops Composition

Following Burn (Rust), Born computes gradients by composing forward operations — not via handwritten backward kernels. This guarantees that backward runs on the same device as forward (CPU or GPU) with no data transfers:

```go
// SiLU backward — composed from Sigmoid, Mul, Add, Sub
sig := backend.Sigmoid(x)
oneMinusSig := backend.Sub(ones, sig)
deriv := backend.Mul(sig, backend.Add(ones, backend.Mul(x, oneMinusSig)))
gradInput := backend.Mul(outputGrad, deriv)
```

No `AsFloat32()`, no GPU→CPU readback, no Go loops. Tensors stay on the compute device throughout the entire forward+backward pass.

---

## Design Decisions

### Why Go, Not Python?

**Python problems for production:**
- 🐌 Slow startup (import torch takes seconds)
- 📦 Dependency hell (pip, conda, virtualenv)
- 🐳 Large Docker images (GB sizes)
- 🔧 Integration friction with Go backends
- 🧵 GIL limitations for concurrency

**Go advantages:**
- ⚡ Fast startup (< 100ms)
- 📦 Single binary deployment
- 🐳 Minimal Docker images (from scratch)
- 🔧 Native integration with Go services
- 🧵 Excellent concurrency primitives

### Why Burn-Inspired Architecture?

Burn (Rust ML framework) proved that:
1. Backend abstraction works well
2. Decorator pattern enables flexibility
3. Type safety doesn't hurt expressiveness
4. Production-focused design is viable

Born adapts these concepts for Go ecosystem.

### Why Not Just Use PyTorch?

**PyTorch is excellent for:**
- ❌ Research prototyping (if you're Python-first)
- ❌ Large-scale distributed training (with Python infrastructure)
- ❌ Access to massive pre-trained model zoo

**Born is better for:**
- ✅ **Production deployment** (single binary)
- ✅ **Go-native integration** (no FFI overhead)
- ✅ **Edge inference** (low resource usage)
- ✅ **Reproducible research** (deterministic builds)
- ✅ **Type-safe ML** (compile-time checks)

---

## Competitive Positioning

### Born vs GoMLX

| Feature | Born | GoMLX |
|---------|------|-------|
| **Dependencies** | Pure Go ✅ | OpenXLA/PJRT (C++) ❌ |
| **Cross-compilation** | Trivial ✅ | Complex ⚠️ |
| **Startup time** | < 100ms ✅ | Slower ⚠️ |
| **Generics** | Go 1.25+ ✅ | Go 1.18+ ✅ |
| **Maturity** | Early development ⚠️ | More mature ✅ |

### Born vs Gorgonia

| Feature | Born | Gorgonia |
|---------|------|----------|
| **Generics** | Type-safe ✅ | Pre-generics ❌ |
| **API Design** | Modern ✅ | Legacy ⚠️ |
| **Backend Abstraction** | Decorator pattern ✅ | Limited ⚠️ |
| **Active Development** | Active ✅ | Slower ⚠️ |

### Born vs PyTorch/TensorFlow (via ONNX)

**Hybrid approach:**

```
PyTorch/TF (training) → ONNX export → Born (deployment)
```

**Advantages:**
- Use Python ecosystem for training (if preferred)
- Deploy as Go binary (production benefits)
- Best of both worlds

---

## Target Use Cases

### ✅ Ideal for Born

**1. Go Microservices + ML**
```go
// Microservice with embedded ML model
func handler(w http.ResponseWriter, r *http.Request) {
    prediction := model.Predict(parseRequest(r))
    json.NewEncoder(w).Encode(prediction)
}
// One binary, no Python sidecar!
```

**2. Edge Deployment**
- Raspberry Pi, IoT devices
- Limited resources (RAM, CPU)
- No internet connectivity
- Fast inference required

**3. Kubernetes Operators**
- ML model serving in K8s
- Native Go integration
- Cloud-native observability
- HPA integration

**4. ML Systems Research**
- Distributed learning algorithms
- Federated learning
- Systems + ML intersection
- Production-critical research

### ❌ Not Ideal for Born (Yet)

**1. Large-Scale Distributed Training**
- Multi-GPU data parallelism planned (v0.10.0)
- Multi-node training planned (v0.12.0)

**2. Pure Algorithm Research**
- If you're Python-first ecosystem
- If you need latest transformers/diffusion models
- If ecosystem size > all else

---

## Roadmap Alignment

### Phase 1: Core Framework ✅ COMPLETE
- Pure Go tensor operations
- CPU backend
- Autodiff engine
- Basic NN modules (Linear, Conv2D, Activations)
- SGD/Adam optimizers

### Phase 2: GPU Acceleration ✅ COMPLETE
- WebGPU backend (zero-CGO via [gogpu/wgpu](https://github.com/gogpu/wgpu), pure Go)
- WGSL compute shaders
- GPU buffer pooling & memory management
- 123x MatMul speedup, 10.9x inference speedup

### Phase 2.5: Transformer Primitives ✅ COMPLETE
- Math operations (Exp, Sqrt, Rsqrt, Cos, Sin)
- Reductions (SumDim, MeanDim)
- Manipulation (Cat, Chunk, Unsqueeze, Squeeze)
- Modern layers (SiLU, RMSNorm, Embedding)
- LLaMA/GPT/Mistral architecture support

### Phase 3: LLM Inference ✅ COMPLETE
- Multi-head attention (MHA), GQA, KV-cache (3.94x speedup)
- Flash Attention 2, speculative decoding
- `models/llama.LoadGGUF()` — end-to-end LLaMA inference (verified: TinyLlama 1.1B Q8_0, Q4_K_M)
- Injectable attention for research experiments (swap implementation at model load time)
- `nn.SetSeed()` for reproducible weight initialization
- Tokenizers (TikToken, BPE), streaming text generation

### Phase 4: Performance ✅ COMPLETE
- CPU parallel MatMul, cache-tiled blocking, AVX2 SIMD (Go 1.26 `goexperiment.simd`)
- GPU shared encoder (ADR-012): utilization 55→80%
- GPU memory pool (ADR-016/017): TieredPool from device.Limits(), explicit Release lifecycle
- GPU buffer leak eliminated: 0 GC warnings (was 43K/step)
- Tensor.Persist()/Unpersist() for cross-step GPU tensor lifecycle
- GPU training example: MLP 77.8 steps/sec

### Phase 5: Scale — In Progress
- Multi-GPU data parallelism (v0.10.0)
- Distributed multi-node training (v0.12.0)
- Resource budget enforcement (GPU/CPU memory limits)

**See [ROADMAP.md](../ROADMAP.md) for detailed timeline and milestones.**

---

## Why Born Will Succeed

### 1. ✅ Right Time
- Go generics available (1.18+, mature in 1.25+)
- Cloud-native deployment critical
- Python dependency hell is real problem
- gogpu/wgpu (pure Go WebGPU) enabling zero-CGO GPU inference

### 2. ✅ Right Problem
Production ML deployment is painful:
- Complex dependencies
- Large container images
- Slow startup times
- Integration friction

Born solves these problems.

### 3. ✅ Right Inspiration
Burn (Rust) proved the concept works.
Born adapts proven patterns for Go ecosystem.

### 4. ✅ Right Ecosystem
- Go dominates cloud-native (Kubernetes, Docker, etc.)
- Microservices architecture (Go's strength)
- Edge computing growth (IoT, embedded)
- ML inference > training in production

---

## Vision: Born as De-Facto Standard

**Goal:** Born becomes the **default choice** for:

1. **ML deployment in Go ecosystem**
   - Every Go service that needs ML uses Born
   - "Train anywhere, deploy Born"

2. **Edge ML inference**
   - Low-resource devices
   - Fast startup required
   - Offline inference

3. **ML Systems research**
   - Distributed learning
   - Federated ML
   - Production-critical experiments

**Not replacing PyTorch for everything** - but becoming **the standard for production ML in Go**.

---

## Contributing to Born Philosophy

When contributing to Born, prioritize:

1. **Production-readiness** > Feature count
2. **Type safety** > Dynamic flexibility
3. **Zero dependencies** > Convenience
4. **Performance** > Ease of implementation
5. **Composability** > Monolithic design

Every feature must answer: **"Does this help production deployment?"**

If yes → implement.
If no → reconsider.

---

**"Born Production-Ready"** - это не слоган, это архитектурный принцип! 🚀
