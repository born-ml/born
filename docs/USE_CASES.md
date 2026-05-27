# Born ML Framework - Use Cases Guide

**Status**: Living Document
**Last Updated**: 2026-05-27

---

## When to Use Born

This guide helps you decide if Born is the right choice for your ML project.

---

## ✅ Ideal Use Cases

### 1. Go Microservices with ML Inference

**Scenario:**
You have a Go backend and need to add ML capabilities (classification, recommendations, NLP).

**Why Born:**
```go
// Traditional approach: Python sidecar
// - Complex deployment (2 runtimes)
// - Network overhead (gRPC/HTTP)
// - Coordination issues

// Born approach: Single binary
import "github.com/born-ml/born/tensor"

func handler(w http.ResponseWriter, r *http.Request) {
    input := parseRequest(r)
    prediction := model.Predict(input)
    json.NewEncoder(w).Encode(prediction)
}
// No Python, no containers, just Go!
```

**Benefits:**
- ✅ Single binary deployment
- ✅ No network overhead
- ✅ Simple debugging
- ✅ Type-safe integration

**Example applications:**
- Fraud detection in payment services
- Content moderation in social platforms
- Recommendation systems in e-commerce
- NLP in chat applications

---

### 2. Edge Deployment

**Scenario:**
Deploy ML models on resource-constrained devices (IoT, embedded, Raspberry Pi).

**Why Born:**
- **Small binary size**: < 10 MB (vs GB for Python stack)
- **Low memory footprint**: Efficient memory usage
- **Fast startup**: < 100ms cold start
- **Cross-compilation**: `GOOS=linux GOARCH=arm64 go build`

**Example applications:**
- Face recognition on security cameras
- Anomaly detection in industrial sensors
- Voice commands on smart devices
- Object detection on drones

**Comparison:**

| Requirement | Born | PyTorch + Python |
|-------------|------|------------------|
| Binary size | < 10 MB | > 1 GB |
| Startup time | < 100ms | 2-5 seconds |
| Memory usage | < 50 MB | > 500 MB |
| Cross-compile | ✅ Trivial | ❌ Complex |

---

### 3. Kubernetes ML Serving

**Scenario:**
Deploy ML models in Kubernetes with auto-scaling, observability, and cloud-native features.

**Why Born:**
```yaml
# Born deployment: FROM scratch!
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  template:
    spec:
      containers:
      - name: born-model
        image: myregistry/born-model:v1.0
        resources:
          limits:
            memory: "128Mi"  # Minimal resource usage!
            cpu: "500m"
```

**Benefits:**
- ✅ Minimal Docker images (from scratch)
- ✅ Fast scale-up (cold start < 100ms)
- ✅ Native Go observability (Prometheus, OpenTelemetry)
- ✅ No Python GIL issues for concurrency

**Example applications:**
- Real-time inference APIs
- Batch prediction jobs
- A/B testing model endpoints
- Multi-tenant ML serving

---

### 4. ML Systems Research

**Scenario:**
Research at the intersection of systems and ML (distributed learning, federated ML, edge AI).

**Why Born:**
- **Reproducibility**: Deterministic builds, no dependency hell
- **Type safety**: Catch bugs at compile-time
- **Systems integration**: Native Go concurrency, networking
- **Production-ready code**: Research code = production code

**Example research areas:**
- Federated learning algorithms
- Distributed training optimization
- Edge-cloud collaboration
- Resource-efficient inference
- Model compression techniques

**Advantages over Python:**

| Aspect | Born (Go) | PyTorch (Python) |
|--------|-----------|------------------|
| Reproducibility | Deterministic builds ✅ | Dependency hell ❌ |
| Concurrency | Go routines ✅ | GIL limitations ❌ |
| Deployment | Same code ✅ | Rewrite needed ❌ |
| Type safety | Compile-time ✅ | Runtime ❌ |

---

### 5. High-Performance Inference

**Scenario:**
Low-latency inference with high throughput (recommendation systems, real-time analytics).

**Why Born:**
```go
// Born: concurrent inference via goroutines
func processBatch(inputs []Tensor) []Prediction {
    results := make([]Prediction, len(inputs))
    var wg sync.WaitGroup

    for i, input := range inputs {
        wg.Add(1)
        go func(idx int, in Tensor) {
            defer wg.Done()
            results[idx] = model.Predict(in)
        }(i, input)
    }

    wg.Wait()
    return results
}
// No GIL, true parallelism!
```

**Benefits:**
- ✅ True parallelism (no GIL)
- ✅ Efficient memory sharing
- ✅ Low-latency (< 1ms overhead)
- ✅ High throughput (concurrent goroutines)

---

### 6. LLM & Transformer Inference

**Scenario:**
Run transformer-based models (LLaMA, Mistral, DeepSeek) with efficient autoregressive generation directly from GGUF files.

**Why Born:**
- **`models/llama.LoadGGUF()`** — end-to-end LLaMA inference, no manual assembly required
- **KV-Cache** for 3.94x faster text generation
- **GGUF quantization**: Q4_K, Q5_K, Q6_K, Q8_0, F16, F32
- **Injectable attention** — swap attention implementation at load time for research

**Example:**
```go
import (
    "fmt"
    "github.com/born-ml/born/backend/cpu"
    "github.com/born-ml/born/models/llama"
    "github.com/born-ml/born/generate"
    "github.com/born-ml/born/tokenizer"
)

func main() {
    backend := cpu.New()

    // Load from GGUF — Q4_K_M, Q8_0, F16, F32 all supported
    model, _ := llama.LoadGGUF("tinyllama-1.1b.Q8_0.gguf", backend)
    defer model.Release()

    tok, _ := tokenizer.NewTikTokenForModel("gpt-4")

    gen := generate.NewTextGenerator(model, tok, generate.SamplingConfig{
        Temperature: 0.7,
        TopP:        0.9,
        TopK:        40,
    })

    result, _ := gen.Generate("Once upon a time", generate.GenerateConfig{
        MaxTokens: 200,
    })
    fmt.Println(result)
}
```

Verified: TinyLlama 1.1B Q8_0 and Q4_K_M.

**Benefits:**
- ✅ Single binary LLM inference — no Python, no runtime libraries
- ✅ GGUF loading with K-quant dequantization
- ✅ KV-cache (3.94x speedup over recompute)
- ✅ Injectable attention for research experiments

**Example applications:**
- Local LLM inference
- Edge AI assistants
- Privacy-preserving text generation
- Embedded chatbots

---

### 7. ONNX Model Deployment

**Scenario:**
Train models in PyTorch/TensorFlow, deploy with Born.

**Workflow:**
```python
# 1. Train in PyTorch
model = train_pytorch_model()
torch.onnx.export(model, "model.onnx")
```

```go
// 2. Deploy with Born (49 ONNX operators supported)
import "github.com/born-ml/born/internal/onnx"

registry := onnx.NewRegistry(backend)
model, _ := onnx.LoadModel("model.onnx", registry)
output := model.Run(input)
```

**Benefits:**
- ✅ Use Python ecosystem for training
- ✅ Deploy as Go binary — 49 ONNX operators supported
- ✅ Best of both worlds

---

## ⚠️ Use with Caution

### 1. Large-Scale Distributed Training

**Current limitation:**
Born doesn't yet support distributed training (planned for future releases).

**Alternative:**
- Train in PyTorch/TensorFlow (distributed)
- Export via ONNX (when available)
- Deploy with Born

**When Born will be ready:**
- Distributed training support planned for future releases

---

### 2. Latest Pre-Trained Models

**Current limitation:**
No model zoo yet (planned future releases).

**Workaround:**
- Use ONNX import (ONNX import (49 ops))
- Port models manually (if simple)

**When Born will be ready:**
- future releases : Model zoo with popular architectures

---

## ❌ Not Recommended (Choose Alternatives)

### 1. Pure Algorithm Research (If Python-First)

**If you are:**
- Deep learning researcher
- Focused on novel architectures
- Need latest transformer/diffusion models
- Python ecosystem is critical

**Use instead:**
- PyTorch (research standard)
- JAX (functional approach)
- TensorFlow (production + research)

**Consider Born when:**
- Deployment is part of research question
- Systems aspects are critical
- Type safety helps (reproducibility)

---

### 2. Computer Vision with Complex Models

**Current limitation:**
- Conv2D basic (no optimizations yet)
- No pre-trained ResNet/EfficientNet (future releases)
- No data augmentation library

**Use Born when:**
- Simple CNNs (MNIST-level)
- Custom architectures
- After upcoming releases (ONNX import + model zoo)

---

### 3. NLP with Pre-Trained Transformers (Large-Scale)

**Current limitation:**
- No model zoo — you need to supply your own GGUF or ONNX file
- Distributed / multi-GPU inference not yet supported

**What works today:**
- ✅ LLaMA-compatible models via `models/llama.LoadGGUF()`
- ✅ Tokenizers (TikToken, BPE), streaming generation
- ✅ ONNX import (49 operators) for encoder-style models (BERT, etc.)

**Use Born when:**
- You have a GGUF or ONNX file and need single-binary deployment
- Edge inference or privacy-preserving local inference

**Wait or use alternatives when:**
- You need access to hundreds of pre-trained models without GGUF conversion
- Multi-GPU or distributed serving at scale is required

---

## Decision Matrix

### Choose Born if:

| Criterion | Born Score |
|-----------|------------|
| Go-native integration required | ✅✅✅ |
| Production deployment priority | ✅✅✅ |
| Edge/IoT deployment | ✅✅✅ |
| Type safety critical | ✅✅ |
| Small binary/memory required | ✅✅✅ |
| Fast cold start needed | ✅✅✅ |

### Avoid Born if:

| Criterion | Born Score |
|-----------|------------|
| Need latest pre-trained models NOW | ❌❌ |
| Large-scale distributed training NOW | ❌❌ |
| Python ecosystem critical | ❌❌❌ |
| Complex CV/NLP NOW (wait upcoming releases) | ⚠️ |

---

## Migration Paths

### From PyTorch to Born

**Option 1: ONNX Import (ONNX import (49 ops))**
```python
# PyTorch
torch.onnx.export(model, "model.onnx")
```
```go
// Born (ONNX import (49 ops))
model := born.LoadONNX("model.onnx")
```

**Option 2: Manual Port (Now)**
- Reimplement model in Born
- Use same hyperparameters
- Validate outputs match

---

### From TensorFlow to Born

**Option 1: ONNX Import (ONNX import (49 ops))**
```python
# TensorFlow
tf2onnx.convert.from_keras(model, output_path="model.onnx")
```
```go
// Born (ONNX import (49 ops))
model := born.LoadONNX("model.onnx")
```

---

### From Gorgonia to Born

**Migration benefits:**
- Modern generics API
- Better type safety
- Active development
- Cleaner codebase

**Migration steps:**
1. Identify model architecture
2. Reimplement using Born modules
3. Transfer weights (manual for now)
4. Validate outputs

---

## Real-World Success Stories (Planned)

### Case Study 1: Payment Fraud Detection

**Before:**
- Python service (Flask + PyTorch)
- 500 MB Docker image
- 3s cold start
- Complex deployment

**After (Born):**
- Go service with embedded Born model
- 15 MB binary
- 50ms cold start
- Single binary deployment

**Results:**
- 97% faster startup
- 10x smaller deployment
- Simpler operations

---

### Case Study 2: Edge Face Recognition

**Before:**
- TensorFlow Lite on Raspberry Pi
- 200 MB dependencies
- 1s startup
- Python runtime required

**After (Born):**
- Born binary on ARM64
- 8 MB binary
- 80ms startup
- No runtime dependencies

**Results:**
- 25x smaller footprint
- 12x faster startup
- Battery life improved (less overhead)

---

## Frequently Asked Questions

### Q: Can I use Born for production NOW?

**A:** Yes, for:
- Simple models (MLP, basic CNN)
- CPU and GPU inference (WebGPU)
- Go-native integration

**Wait for upcoming releases** if you need:
- ONNX import
- Pre-trained models
- Vulkan/CUDA backends

---

### Q: Will Born replace PyTorch?

**A:** No, different goals:
- **PyTorch**: Research, training, experimentation
- **Born**: Production deployment, Go integration

**Best practice:** Train in PyTorch → Deploy with Born (when ONNX import available)

---

### Q: Is Born stable enough?

**A:** Current status (v0.9.1):
- ✅ Core API stable (tensor, nn, optim, autodiff)
- ✅ Production-tested (MNIST 97%+, GPU 123x speedup, MLP GPU training 77.8 steps/sec)
- ✅ LLM inference: LLaMA via `models/llama.LoadGGUF()`, verified on TinyLlama 1.1B Q8_0 and Q4_K_M
- ✅ GPU training: all forward+backward ops on GPU, explicit buffer lifecycle, TieredPool memory management
- ✅ ONNX import (49 operators)
- ⚠️ API may evolve before v1.0

**Recommendation:** Ready for production inference and GPU training workloads.

---

## Get Started

If Born fits your use case:

```bash
# Install (latest release)
go get github.com/born-ml/born@latest

# Example
package main

import (
    "github.com/born-ml/born/tensor"
    "github.com/born-ml/born/backend/cpu"
    "github.com/born-ml/born/nn"
)

func main() {
    backend := cpu.New()

    // Create model
    model := nn.NewSequential(
        nn.NewLinear(784, 128, backend),
        nn.NewReLU(),
        nn.NewLinear(128, 10, backend),
    )

    // Inference
    x := tensor.Randn[float32](tensor.Shape{1, 784}, backend)
    output := model.Forward(x)
}
```

**Next steps:**
- Check [README.md](../README.md)
- Read [PHILOSOPHY.md](PHILOSOPHY.md)
- Explore examples in `/examples`

---

**Need help deciding?** Open an issue on GitHub with your use case! 🚀
