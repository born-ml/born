# Born - Production-Ready ML for Go

<p align="center">
  <img src="assets/born.png" alt="Born ML Framework - Inspired by Burn" width="800">
</p>

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

**Project Status**: 🚀 **v0.7.0 Released!** (Flash Attention 2 + Speculative Decoding + GGUF!)
**Latest**: 🔥 Flash Attention 2 (WebGPU), Speculative Decoding (2-4x speedup), GGUF Import (K-quants)

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
- **Autodiff** - Automatic differentiation via decorator pattern
- **Production Ready** - Single binary deployment, fast startup
- **WebAssembly** - Run inference in browsers natively

### GPU Acceleration
- **WebGPU Backend** - Zero-CGO GPU via [go-webgpu](https://github.com/go-webgpu/webgpu), 123x MatMul speedup
- **38+ GPU Operations** - MatMul, BatchMatMul, Conv2D, MaxPool2D, Softmax, and more
- **Lazy Evaluation** - GPU-resident tensors, command batching (~90s → <5s/step)
- **Multi-dim Transpose** - GPU-accelerated 3D/4D/5D/6D tensors
- **Automatic Memory** - `runtime.SetFinalizer` for GPU buffer cleanup

### LLM & Transformers
- **Flash Attention 2** - O(N) memory, WebGPU WGSL shader, 2x+ speedup on long sequences
- **Speculative Decoding** - Draft model + verification, 2-4x inference speedup
- **Multi-Head Attention** - MHA, SDPA, Grouped Query Attention (GQA)
- **KV-Cache** - Efficient autoregressive generation (3.94x speedup)
- **Positional Encodings** - RoPE, ALiBi, Sinusoidal, Learned
- **Modern FFN** - SwiGLU, GeGLU, ReGLU with gated activations
- **Normalizations** - LayerNorm, RMSNorm (LLaMA style)
- **Tokenizers** - TikToken, BPE, HuggingFace format, chat templates
- **Sampling** - Temperature, Top-K, Top-P, Min-P, repetition penalty
- **Text Generation** - Streaming API, stop sequences

### Model Import & Export
- **ONNX Import** - Load PyTorch/TensorFlow models via `.onnx` (30+ operators)
- **GGUF Import** - llama.cpp format with K-quant dequantization (Q4_K, Q5_K, Q6_K, Q8_0)
- **Native Format** - `.born` format with `nn.Save()` / `nn.Load()`
- **Checkpoints** - Resume training with optimizer state preservation
- **SafeTensors** - HuggingFace compatible export

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

### Example: LLM Text Generation

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
- ✅ **35+ GPU operations** (BatchMatMul, Conv2D, MaxPool2D, Comparisons, Reductions)
- ✅ **31 type-safe public API operations** (MulScalar, Greater, Softmax, Int32, etc.)
- ✅ Automatic differentiation with gradient tape
- ✅ Neural network modules (Linear, Conv2D, ReLU, SiLU, RMSNorm, Embedding)
- ✅ Optimizers (SGD with momentum, Adam with bias correction)
- ✅ Losses (CrossEntropyLoss with numerical stability)
- ✅ **Complete WebGPU backend** (zero-CGO, 123x MatMul speedup)
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
| CPU | ✅ **Available** | Pure Go implementation, all operations |
| WebGPU | ✅ **Available** | Zero-CGO GPU via [go-webgpu](https://github.com/go-webgpu/webgpu) |
| Vulkan | 📋 Planned | Cross-platform GPU compute (Linux focus) |
| CUDA | 📋 Planned | NVIDIA GPU via zero-CGO |
| Metal | 📋 Planned | Apple GPU (macOS/iOS) |

**WebGPU Operation Support** 🎉

| Category | Operations | Backend |
|----------|------------|---------|
| **Math** | Add, Sub, Mul, Div (float32 + int32), Exp, Sqrt, Rsqrt, Log, Cos, Sin | ✅ GPU |
| **Matrix** | MatMul, **BatchMatMul** (3D/4D), Transpose, Reshape | ✅ GPU |
| **CNN** | **Conv2D**, **MaxPool2D** | ✅ GPU |
| **Activation** | ReLU, Sigmoid, Tanh, Softmax | ✅ GPU |
| **Scalar** | MulScalar, AddScalar, SubScalar, DivScalar | ✅ GPU |
| **Reduction** | **Sum**, SumDim, MeanDim, **Argmax** | ✅ GPU/CPU hybrid |
| **Compare** | **Greater**, **Lower**, GreaterEqual, LowerEqual, **Equal**, NotEqual | ✅ GPU |
| **Boolean** | **And**, **Or**, **Not** | ✅ GPU |
| **Shape** | Cat, Chunk, Unsqueeze, Squeeze, **Expand** | ✅ CPU (efficient) |
| **Selection** | **Where**, **Gather**, **Embedding** | ✅ GPU |
| **Type** | **Cast** (float32, int32) | ✅ CPU |

**Total: 38+ GPU-accelerated operations!**

*All operations required for LLM inference (Attention, RoPE, LayerNorm, etc.) are fully supported on GPU.*

**GPU Backend Setup:**

WebGPU requires the `wgpu_native` library. Download from [wgpu-native releases](https://github.com/gfx-rs/wgpu-native/releases):

**Windows (x64):**
```bash
# Download latest release
curl -LO https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-windows-x86_64-msvc-release.zip
unzip wgpu-windows-x86_64-msvc-release.zip

# Install DLL system-wide (requires admin)
copy lib\wgpu_native.dll C:\Windows\System32\

# Or place next to your executable
copy lib\wgpu_native.dll .\your-app\
```

**Linux (x64):**
```bash
curl -LO https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip
sudo cp lib/libwgpu_native.so /usr/local/lib/
sudo ldconfig
```

**macOS (ARM64):**
```bash
curl -LO https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-macos-aarch64-release.zip
unzip wgpu-macos-aarch64-release.zip
sudo cp lib/libwgpu_native.dylib /usr/local/lib/
```

**Usage:**
```go
import (
    "github.com/born-ml/born/autodiff"
    "github.com/born-ml/born/backend/cpu"
    "github.com/born-ml/born/backend/webgpu"
)

// Automatic GPU/CPU selection with graceful fallback
var backend tensor.Backend
if webgpu.IsAvailable() {
    gpu, err := webgpu.New()
    if err == nil {
        backend = autodiff.New(gpu)
        defer gpu.Release() // Don't forget to release GPU resources
    }
}
if backend == nil {
    backend = autodiff.New(cpu.New())
}
```

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

### ✅ What's Working

**Core Framework**
- Tensor API with generics, autodiff, NN modules (Linear, Conv2D, ReLU, etc.)
- Optimizers (SGD, Adam), losses (CrossEntropyLoss)
- MNIST: 97.44% MLP, 98.18% CNN accuracy

**GPU Acceleration**
- WebGPU backend with 38+ operations (123x MatMul speedup)
- Lazy evaluation, command batching (~90s → <5s/step)
- CNN support (Conv2D, MaxPool2D, BatchMatMul)

**LLM & Transformers**
- Multi-Head Attention, GQA, KV-Cache (3.94x speedup)
- RoPE, ALiBi, RMSNorm, SwiGLU
- Tokenizers (TikToken, BPE), text generation with streaming

**Model Import & Export**
- ONNX import (30+ operators)
- GGUF loading (LLaMA, Mistral, DeepSeek)
- Native `.born` format, SafeTensors export

### 🚀 Upcoming

**Inference Optimization** - Flash Attention 2, Speculative Decoding, GGUF import

**Quantization** - GPTQ/AWQ (4x smaller), KV Cache compression, Model Zoo

**Production Serving** - PagedAttention, Continuous Batching, OpenAI-compatible API

**Scale & Stability** - Multi-GPU, CPU SIMD (AVX2/Neon), Gradient Checkpointing

**v1.0 LTS** - API freeze, 3+ years support, production hardening

**Full roadmap & changelog**: See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md)

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

### WebGPU WGSL Shaders

Born includes **30+ optimized WGSL compute shaders**:

| Shader | Workgroup | Description |
|--------|-----------|-------------|
| `addShader` | 256 | Element-wise addition |
| `subShader` | 256 | Element-wise subtraction |
| `mulShader` | 256 | Element-wise multiplication |
| `divShader` | 256 | Element-wise division |
| `matmulShader` | 16x16 | Matrix multiplication (2D) |
| `batchMatMulShader` | 8x8x1 | Batched matmul (3D/4D) |
| `conv2dShader` | 8x8x1 | 2D convolution with padding |
| `maxPool2dShader` | 8x8x1 | 2D max pooling |
| `transposeShader` | 16x16 | Matrix transpose |
| `reluShader` | 256 | ReLU activation |
| `sigmoidShader` | 256 | Sigmoid activation |
| `tanhShader` | 256 | Tanh activation |
| `softmaxShader` | 256 | Softmax (numerically stable) |
| `expShader` | 256 | Element-wise exp |
| `sqrtShader` | 256 | Element-wise sqrt |
| `rsqrtShader` | 256 | Reciprocal sqrt (1/√x) |
| `cosShader` | 256 | Element-wise cosine |
| `sinShader` | 256 | Element-wise sine |
| `greaterShader` | 256 | Greater-than comparison |
| `lowerShader` | 256 | Less-than comparison |
| `equalShader` | 256 | Equality comparison |
| `andShader` | 256 | Logical AND |
| `orShader` | 256 | Logical OR |
| `notShader` | 256 | Logical NOT |
| `argmaxShader` | 256 | Argmax along dimension |
| `globalSumShader` | 256 | Parallel sum reduction |
| `scalarMulShader` | 256 | Scalar multiplication |
| `scalarAddShader` | 256 | Scalar addition |
| `addShaderInt32` | 256 | Int32 element-wise addition |
| `subShaderInt32` | 256 | Int32 element-wise subtraction |
| `mulShaderInt32` | 256 | Int32 element-wise multiplication |
| `divShaderInt32` | 256 | Int32 element-wise division |

All shaders use **workgroup shared memory** for optimal performance and support **bounds checking** for safety.

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

### 🙏 [go-webgpu](https://github.com/AlfredDobra662/webgpu) & [wgpu-native](https://github.com/gfx-rs/wgpu-native)

Born's GPU acceleration is powered by **go-webgpu** - a remarkable pure Go binding for WebGPU via **wgpu-native**.

**Why this stack is special:**
- **Zero CGO** - Pure Go bindings using [goffi](https://github.com/AlfredDobra662/goffi) for FFI
- **Cross-platform** - Works on Windows (D3D12), Linux (Vulkan), macOS (Metal)
- **Modern API** - Clean, idiomatic Go interface to WebGPU
- **wgpu-native** - Battle-tested Rust implementation of WebGPU by [gfx-rs](https://github.com/gfx-rs)
- **Active development** - Both projects are actively maintained

Without go-webgpu and wgpu-native, Born would need CGO for GPU support, making cross-compilation complex and defeating our "pure Go" goal. This stack enables us to offer **production-ready GPU acceleration** while maintaining the simplicity of `go build`.

Thank you to [Alfred Dobra](https://github.com/AlfredDobra662), [gfx-rs team](https://github.com/gfx-rs), and all contributors!

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
A: Yes! Full LLM support included - GGUF model loading, tokenizers, sampling strategies, and text generation with streaming. Load LLaMA, Mistral, or DeepSeek models directly.

**Q: When will it be ready?**
A: Core features are released! CPU/GPU backends, transformers, LLM support, and ONNX import all work. See [ROADMAP.md](ROADMAP.md) for upcoming features.

**Q: Can I use PyTorch models?**
A: Yes! Via ONNX import. Train in PyTorch, export to ONNX, deploy with Born. GGUF models are also supported.

**Q: WebAssembly support?**
A: Yes! Pure Go compiles to WASM natively. Inference in browsers out of the box.

**Q: What LLM architectures are supported?**
A: LLaMA 2/3, Mistral, DeepSeek, and compatible architectures. GQA, RoPE, SwiGLU are all supported.

**Q: How do I enable GPU acceleration?**
A: Install `wgpu_native` library from [wgpu-native releases](https://github.com/gfx-rs/wgpu-native/releases), then use `webgpu.IsAvailable()` to check GPU support. See [Architecture](#backend-abstraction) for setup instructions. **38+ GPU operations** included - everything needed for LLM inference!

**Q: What GPU operations are supported?**
A: **All operations needed for production ML!** Math (Add, Mul, Exp, etc.), Matrix (MatMul, BatchMatMul, Conv2D), Activations (ReLU, Softmax), Comparisons (Greater, Equal), Boolean (And, Or, Not), Reductions (Sum, Argmax), and more. See the [WebGPU Operation Table](#backend-abstraction).

**Q: How can I help?**
A: Check our [Contributing Guide](CONTRIBUTING.md) and [GitHub Issues](https://github.com/born-ml/born/issues)!

---

<div align="center">

**Born for Production. Ready from Day One.**

Made with ❤️ by the Born ML team

[Documentation](docs/) • [Contributing](CONTRIBUTING.md) • [Community](#community)

</div>
