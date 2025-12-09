# Changelog

All notable changes to the Born ML Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.0]: https://github.com/born-ml/born/releases/tag/v0.5.0
[0.4.0]: https://github.com/born-ml/born/releases/tag/v0.4.0
[0.3.0]: https://github.com/born-ml/born/releases/tag/v0.3.0
[0.2.0]: https://github.com/born-ml/born/releases/tag/v0.2.0
[0.1.1]: https://github.com/born-ml/born/releases/tag/v0.1.1
[0.1.0]: https://github.com/born-ml/born/releases/tag/v0.1.0
