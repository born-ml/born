# Born - Production-Ready ML for Go

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/born-ml/born.svg)](https://pkg.go.dev/github.com/born-ml/born)
[![Go Report Card](https://goreportcard.com/badge/github.com/born-ml/born)](https://goreportcard.com/report/github.com/born-ml/born)
[![Pure Go](https://img.shields.io/badge/100%25-Pure_Go-00ADD8)](https://golang.org/)
[![Release](https://img.shields.io/github/v/release/born-ml/born?include_prereleases&label=version)](https://github.com/born-ml/born/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Test Status](https://github.com/born-ml/born/actions/workflows/test.yml/badge.svg)](https://github.com/born-ml/born/actions/workflows/test.yml)
[![Codecov](https://codecov.io/gh/born-ml/born/branch/main/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/gh/born-ml/born)
[![Discussions](https://img.shields.io/github/discussions/born-ml/born?logo=github&label=Discussions)](https://github.com/born-ml/born/discussions)

> **"Models are born production-ready"**

Born is a modern deep learning framework for Go, inspired by [Burn](https://github.com/tracel-ai/burn) (Rust). Build ML models in pure Go and deploy as single binaries - no Python runtime, no complex dependencies.

**Project Status**: 🎉 **v0.5.0 Released!** (Complete LLM Support!)
**Latest**: ⚡ Phase 5 complete - GQA, SwiGLU, Tokenizers, Model Loading, Text Generation

*Pure Go ML with GPU acceleration - no CGO required!*

---

## Why Born?

### The Problem
Deploying ML models is hard:
- Python runtime required
- Complex dependency management
- Large Docker images
- Slow startup times
- Integration friction with Go backends

### The Born Solution
```go
import "github.com/born-ml/born"

// Models "born" ready for production
model := born.Load("resnet50.born")
prediction := model.Predict(image)

// That's it. No Python. No containers. Just Go.
```

**Benefits**:
- Single binary deployment
- Fast startup (< 100ms)
- Small memory footprint
- Native Go integration
- Cross-platform out of the box

---

## Features

### Core
- **Pure Go** - No CGO dependencies, trivial cross-compilation
- **Type Safe** - Generics-powered API for compile-time guarantees
- **GPU Acceleration** - WebGPU backend (zero-CGO, 123x speedup)
- **Autodiff** - Automatic differentiation via decorator pattern
- **Production Ready** - Single binary deployment, fast startup
- **WebAssembly** - Run inference in browsers natively

### LLM Support (v0.5.0) 🆕
- **Grouped Query Attention (GQA)** - Memory-efficient attention (LLaMA 2/3, Mistral)
- **SwiGLU FFN** - Modern FFN with gated activations (+ GeGLU, ReGLU, GLU)
- **Model Loading** - GGUF format support, weight mapping for LLaMA/Mistral/DeepSeek
- **Tokenizers** - TikToken, BPE, HuggingFace format, chat templates
- **Sampling** - Temperature, Top-K, Top-P (nucleus), Min-P, repetition penalty
- **Text Generation** - Streaming API, KV-cache integration, stop sequences

### Transformer Architecture (v0.4.0)
- **Multi-Head Attention (MHA)** - Full implementation with Q, K, V projections
- **Scaled Dot-Product Attention** - Core attention with optional mask/dropout
- **KV-Cache** - Efficient autoregressive generation (3.94x speedup)
- **Positional Encodings** - RoPE, ALiBi, Sinusoidal, Learned
- **TransformerBlock** - Complete Pre-Norm/Post-Norm support
- **Normalizations** - LayerNorm, RMSNorm (LLaMA style)
- **FFN** - Feed-Forward Networks with SiLU activation

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/born-ml/born.git
cd born

# Build
make build

# Or install CLI
make install
```

### Development Setup

**Requirements**:
- Go 1.25+
- Make (optional, but recommended)
- golangci-lint (for linting)

**Build**:
```bash
make build          # Build all binaries
make test           # Run tests
make lint           # Run linter
make bench          # Run benchmarks
```

### Example: MNIST Classification

**Working example included!** See `examples/mnist/` for complete implementation.

```go
package main

import (
    "github.com/born-ml/born/autodiff"
    "github.com/born-ml/born/backend/cpu"
    "github.com/born-ml/born/nn"
    "github.com/born-ml/born/optim"
)

func main() {
    // Create backend with autodiff
    backend := autodiff.New(cpu.New())

    // Define model (784 → 128 → 10)
    model := NewMNISTNet(backend)

    // Create loss and optimizer
    criterion := nn.NewCrossEntropyLoss(backend)
    optimizer := optim.NewAdam(model.Parameters(), optim.AdamConfig{
        LR:    0.001,
        Betas: [2]float32{0.9, 0.999},
    }, backend)

    // Training loop
    for epoch := range 10 {
        // Forward pass
        logits := model.Forward(batch.ImagesTensor)
        loss := criterion.Forward(logits, batch.LabelsTensor)

        // Backward pass
        optimizer.ZeroGrad()
        grads := backend.Backward(loss.Raw())
        optimizer.Step(grads)

        // Log progress
        acc := nn.Accuracy(logits, batch.LabelsTensor)
        fmt.Printf("Epoch %d: Loss=%.4f, Accuracy=%.2f%%\n",
            epoch, loss.Raw().AsFloat32()[0], acc*100)
    }
}
```

**Run it:** `cd examples/mnist && go run .`

### Example: LLM Text Generation (v0.5.0)

```go
package main

import (
    "fmt"
    "github.com/born-ml/born/generate"
    "github.com/born-ml/born/tokenizer"
    "github.com/born-ml/born/loader"
)

func main() {
    // Load tokenizer
    tok, _ := tokenizer.NewTikTokenForModel("gpt-4")

    // Load model (GGUF format)
    model, _ := loader.OpenModel("llama-7b.gguf")

    // Create generator with sampling config
    gen := generate.NewTextGenerator(model, tok, generate.SamplingConfig{
        Temperature: 0.7,
        TopP:        0.9,
        TopK:        40,
    })

    // Generate text
    result, _ := gen.Generate("Hello, world!", generate.GenerateConfig{
        MaxTokens: 100,
    })
    fmt.Println(result)

    // Or use streaming
    stream, _ := gen.GenerateStream("Once upon a time", generate.GenerateConfig{
        MaxTokens: 50,
        Stream:    true,
    })
    for chunk := range stream {
        fmt.Print(chunk.Token)
    }
}
```

**Core Features:**
- ✅ Tensor operations (Add, MatMul, Reshape, Exp, Sqrt, Cat, etc.)
- ✅ **31 type-safe public API operations** (MulScalar, Greater, Softmax, Int32, etc.)
- ✅ Automatic differentiation with gradient tape
- ✅ Neural network modules (Linear, Conv2D, ReLU, SiLU, RMSNorm, Embedding)
- ✅ Optimizers (SGD with momentum, Adam with bias correction)
- ✅ Losses (CrossEntropyLoss with numerical stability)
- ✅ GPU acceleration (WebGPU - 123x speedup)
- ✅ Transformer primitives (for LLaMA, GPT, Mistral architectures)

---

## Architecture

### Backend Abstraction

Born uses a backend interface for device independence:

```go
type Backend interface {
    Add(a, b *RawTensor) *RawTensor
    MatMul(a, b *RawTensor) *RawTensor
    // ... other operations
}
```

**Available Backends:**

| Backend | Status | Description |
|---------|--------|-------------|
| CPU | ✅ **Available** | Pure Go implementation (v0.1.1) |
| WebGPU | ✅ **Available** | Zero-CGO GPU via [go-webgpu](https://github.com/go-webgpu/webgpu) (v0.2.0) |
| Vulkan | 📋 Q3 2025 | Cross-platform GPU compute |
| CUDA | 📋 Q3 2025 | NVIDIA GPU via zero-CGO |
| Metal | 📋 Q4 2025 | Apple GPU (macOS/iOS) |

### Decorator Pattern

Functionality composed via decorators (inspired by Burn):

```go
// Basic backend
base := cpu.New()

// Add autodiff
withAutodiff := autodiff.New(base)

// Add kernel fusion
optimized := fusion.New(withAutodiff)

// Your code works with any backend!
model := createModel(optimized)
```

### Type Safety with Generics

```go
type Tensor[T DType, B Backend] struct {
    raw     *RawTensor
    backend B
}

// Compile-time type checking
func (t *Tensor[float32, B]) MatMul(other *Tensor[float32, B]) *Tensor[float32, B]
```

---

## Roadmap

### Phase 1: Core (v0.1) - ✅ COMPLETE (Nov 2025)
- [x] Tensor API with generics
- [x] CPU backend (pure Go)
- [x] Autodiff decorator with gradient tape
- [x] NN modules (Linear, ReLU, Sigmoid, Tanh, Sequential)
- [x] SGD/Adam optimizers with momentum/bias correction
- [x] CrossEntropyLoss with numerical stability
- [x] MNIST classification example

**Status**: All 7 core tasks complete. 132 unit tests, 83.8% average coverage, 0 linter issues.

### Phase 2: GPU Backends (v0.2) - ✅ COMPLETE (Nov 2025)
- [x] WebGPU backend (zero-CGO via go-webgpu)
- [x] WGSL compute shaders (12 operations)
- [x] GPU buffer pooling & memory management
- [x] MNIST GPU inference (10.9x speedup)

**Status**: All 5 GPU tasks complete. 123x MatMul speedup, ~16000 samples/sec throughput.

### Phase 2.5: Transformer Primitives (v0.3) - ✅ COMPLETE (Nov 2025)
- [x] Math operations (Exp, Sqrt, Rsqrt, Cos, Sin, Log)
- [x] Reductions (SumDim, MeanDim with keepDim, Sum, Argmax)
- [x] Tensor manipulation (Cat, Chunk, Unsqueeze, Squeeze, Expand)
- [x] Indexing (Gather, Where)
- [x] Modern layers (SiLU, RMSNorm, Embedding, Softmax)
- [x] Gradient control (NoGrad, Detach)
- [x] **31 public API operations** (MulScalar, Greater/Gt, Int32, etc.)

**Status**: All 7 tasks complete. 112 new tests, 0 linter issues.

### Phase 4: Attention Mechanisms (v0.4.0) - December 2025 ✅ COMPLETE
- [x] Multi-head attention (MHA)
- [x] Scaled dot-product attention (SDPA)
- [x] KV-cache for inference (3.94x speedup)
- [x] Layer normalization (LayerNorm + RMSNorm)
- [x] Positional encodings (RoPE, ALiBi, Sinusoidal, Learned)
- [x] Transformer block with FFN
- [x] BatchMatMul for 3D/4D tensors

**Status**: All 8 tasks complete. 80+ new tests, 0 linter issues. **Full Transformer architecture ready!**

### Phase 5: LLM Support (v0.5.0) - December 2025 ✅ COMPLETE
- [x] Grouped Query Attention (GQA) - LLaMA 2/3, Mistral style
- [x] SwiGLU + GLU variants (GeGLU, ReGLU)
- [x] Model Loader (GGUF format, weight mappers)
- [x] Tokenizer integration (TikToken, BPE, chat templates)
- [x] Sampling strategies (Top-K, Top-P, Min-P, temperature, penalties)
- [x] Inference Pipeline (TextGenerator, streaming, stop sequences)

**Status**: All 6 LLM tasks complete. 100+ new tests, 0 linter issues. **Ready for LLM inference!**

### Phase 6: ONNX & Cross-Platform (v0.6.0) - Q1 2026
- [ ] Linux/macOS WebGPU support
- [ ] ONNX import/export
- [ ] Model quantization (INT8, FP16)
- [ ] Pre-trained model hub integration

### Long-Term: v1.0 LTS - 2026
- [ ] Distributed training
- [ ] Flash Attention
- [ ] Model zoo with pre-trained weights
- [ ] Production optimizations (SIMD, memory pooling)

**Full roadmap**: See [ROADMAP.md](ROADMAP.md)

---

## Documentation

### For Users

- **[Philosophy](docs/PHILOSOPHY.md)** - Production-first design principles
- **[Use Cases](docs/USE_CASES.md)** - When to use Born (and when not)
- **[Getting Started](docs/getting-started.md)** - Installation and first steps *(coming soon)*
- **[API Reference](https://pkg.go.dev/github.com/born-ml/born)** - Complete API documentation
- **[Examples](examples/)** - Sample code (MNIST MLP, CNN, GPU inference)

### For Contributors

- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[GitHub Issues](https://github.com/born-ml/born/issues)** - Report bugs or request features

---

## Philosophy

### "Born Ready"

Models trained anywhere (PyTorch, TensorFlow) are **imported** and **born** production-ready:

```
Training → Birth → Production
 (Burn)    (Born)    (Run)

PyTorch trains  →  Born imports  →  Born deploys
TensorFlow trains → Born imports → Born deploys
Born trains    →  Born ready   →  Born serves
```

### Production First

- **Single Binary**: Entire model in one executable
- **No Runtime**: No Python, no dependencies
- **Fast Startup**: < 100ms cold start
- **Small Memory**: Minimal footprint
- **Cloud Native**: Natural fit for Go services

### Developer Experience

- **Type Safe**: Catch errors at compile time
- **Clean API**: Intuitive and ergonomic
- **Great Docs**: Comprehensive documentation
- **Easy Deploy**: `go build` and you're done

---

## Performance

**Actual Benchmarks** (AMD Ryzen 9 5950X, NVIDIA RTX 3080):

### Matrix Operations (WebGPU vs CPU)

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| MatMul 1024x1024 | 7143ms | 58ms | **123x** |
| MatMul 512x512 | 499ms | 12ms | **41x** |
| MatMul 256x256 | 56ms | 3.7ms | **15x** |

### Neural Network Inference

| Batch Size | CPU | GPU | Speedup | Throughput |
|------------|-----|-----|---------|------------|
| 64 | 48ms | 19ms | 2.5x | 3,357/s |
| 256 | 182ms | 21ms | **8.5x** | 11,883/s |
| 512 | 348ms | 32ms | **10.9x** | 15,973/s |

*Note: CPU backend uses naive O(n³) MatMul. SIMD optimizations planned for future releases.*

---

## Inspiration

Born is inspired by and learns from:

- **[Burn](https://github.com/tracel-ai/burn)** - Architecture patterns, decorator design
- **[PyTorch](https://pytorch.org/)** - API ergonomics
- **[TinyGrad](https://github.com/geohot/tinygrad)** - Simplicity principles
- **[Gonum](https://github.com/gonum/gonum)** - Go numerical computing
- **[HDF5 for Go](https://github.com/scigolib/hdf5)** - Model serialization, dataset storage (planned)

---

## Acknowledgments

Special thanks to the projects that made Born possible:

### 🙏 [go-webgpu](https://github.com/AlfredDobra662/webgpu)

Born's GPU acceleration is powered by **go-webgpu** - a remarkable pure Go binding for WebGPU via wgpu-native.

**Why go-webgpu is special:**
- **Zero CGO** - Pure Go bindings using [goffi](https://github.com/AlfredDobra662/goffi) for FFI
- **Cross-platform** - Works on Windows, Linux, macOS
- **Modern API** - Clean, idiomatic Go interface to WebGPU
- **Active development** - Maintained and improving

Without go-webgpu, Born would need CGO for GPU support, making cross-compilation complex and defeating our "pure Go" goal. This library enables us to offer **production-ready GPU acceleration** while maintaining the simplicity of `go build`.

Thank you to [Alfred Dobra](https://github.com/AlfredDobra662) and all contributors!

---

## Community

**Project is in early development**. Star the repo to follow progress!

- **GitHub Org**: [github.com/born-ml](https://github.com/born-ml)
- **Main Repo**: [github.com/born-ml/born](https://github.com/born-ml/born)
- **Discussions**: [GitHub Discussions](https://github.com/born-ml/born/discussions)
  - [Announcements](https://github.com/born-ml/born/discussions/2)
  - [Q&A](https://github.com/born-ml/born/discussions/3)
  - [Feature Requests](https://github.com/born-ml/born/discussions/4)
- **Issues**: [Report bugs or request features](https://github.com/born-ml/born/issues)

---

## License

Licensed under the **Apache License, Version 2.0**.

**Why Apache 2.0?**
- ✅ **Patent protection** - Critical for ML algorithms and production use
- ✅ **Enterprise-friendly** - Clear legal framework for commercial adoption
- ✅ **Industry standard** - Same as TensorFlow, battle-tested in ML ecosystem
- ✅ **Contributor protection** - Explicit patent grant and termination clauses

See [LICENSE](LICENSE) file for full terms.

---

## FAQ

**Q: Why not use Gorgonia?**
A: Gorgonia is great but uses a different approach. Born focuses on modern Go (generics), pure Go (no CGO), and production-first design inspired by Burn.

**Q: Can I run LLMs with Born?**
A: Yes! v0.5.0 includes full LLM support - GGUF model loading, tokenizers, sampling strategies, and text generation with streaming. Load LLaMA, Mistral, or DeepSeek models directly.

**Q: When will it be ready?**
A: Core features (v0.1-v0.5) are RELEASED! Includes CPU/GPU backends, transformer architecture, and LLM support. ONNX import targeted for v0.6.0 (Q1 2026).

**Q: Can I use PyTorch models?**
A: Yes! Via ONNX import (v0.6.0, Q1 2026). Train in PyTorch, deploy with Born. Currently GGUF models are supported.

**Q: WebAssembly support?**
A: Yes! Pure Go compiles to WASM natively. Inference in browsers out of the box.

**Q: What LLM architectures are supported?**
A: LLaMA 2/3, Mistral, DeepSeek, and compatible architectures. GQA, RoPE, SwiGLU are all supported.

**Q: How can I help?**
A: Check our [Contributing Guide](CONTRIBUTING.md) and [GitHub Issues](https://github.com/born-ml/born/issues)!

---

<div align="center">

**Born for Production. Ready from Day One.**

Made with ❤️ by the Born ML team

[Documentation](docs/) • [Contributing](CONTRIBUTING.md) • [Community](#community)

</div>
