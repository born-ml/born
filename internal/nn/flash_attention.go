// Package nn provides neural network modules and layers for building deep learning models.
package nn

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// FlashAttentionConfig configures the Flash Attention module.
type FlashAttentionConfig struct {
	NumHeads   int  // Number of attention heads.
	HeadDim    int  // Dimension per head.
	MaxSeqLen  int  // Maximum sequence length.
	CausalMask bool // Whether to use causal (autoregressive) masking.
	BlockSize  int  // Tile size for blocked computation (default: 64).
}

// FlashAttention implements Flash Attention 2 algorithm.
//
// Memory complexity: O(N) instead of O(N²) for standard attention.
//
// Flash Attention achieves O(N) memory by:
//  1. Tiling the computation into blocks (size B)
//  2. Processing Q blocks sequentially
//  3. For each Q block, iterating over K,V blocks
//  4. Using online softmax to accumulate results incrementally
//  5. Never materializing the full N×N attention matrix
//
// This Week 1 implementation is a CPU reference for correctness validation.
// GPU kernels will be added in later weeks.
//
// Reference: "Flash Attention 2: Faster Attention with Better Parallelism"
// Dao et al., 2023 (https://arxiv.org/abs/2307.08691)
type FlashAttention[T tensor.DType, B tensor.Backend] struct {
	config  FlashAttentionConfig
	backend B
	scale   float32 // 1/sqrt(headDim)
}

// NewFlashAttention creates a new Flash Attention module.
//
// Parameters:
//   - config: Configuration specifying heads, dimensions, block size, etc.
//   - backend: Tensor backend (CPU, GPU, etc.)
//
// Returns:
//   - *FlashAttention: Initialized Flash Attention module.
//
// Example:
//
//	config := nn.FlashAttentionConfig{
//	    NumHeads:   8,
//	    HeadDim:    64,
//	    MaxSeqLen:  2048,
//	    CausalMask: true,
//	    BlockSize:  64,
//	}
//	fa := nn.NewFlashAttention[float32](config, backend)
func NewFlashAttention[T tensor.DType, B tensor.Backend](
	config FlashAttentionConfig,
	backend B,
) *FlashAttention[T, B] {
	// Default block size if not specified
	if config.BlockSize == 0 {
		config.BlockSize = 64
	}

	scale := float32(1.0 / math.Sqrt(float64(config.HeadDim)))

	return &FlashAttention[T, B]{
		config:  config,
		backend: backend,
		scale:   scale,
	}
}

// Forward computes attention output using Flash Attention algorithm.
//
// This method implements the tiled Flash Attention algorithm:
//  1. For each query block Qi (size blockSize × headDim):
//  2. Initialize online softmax accumulator
//  3. For each key/value block (Kj, Vj):
//  4. Compute scores Sij = Qi @ Kj^T / sqrt(d)
//  5. Apply causal mask if needed
//  6. Update online softmax with (Sij, Vj)
//  7. Normalize accumulated output
//
// GPU Acceleration (Week 2): When using WebGPU backend, automatically dispatches
// to GPU-optimized Flash Attention kernel. Falls back to CPU for other backends.
//
// Parameters:
//   - q: Query tensor [batch, seqLen, numHeads, headDim]
//   - k: Key tensor [batch, kvLen, numHeads, headDim]
//   - v: Value tensor [batch, kvLen, numHeads, headDim]
//   - mask: Optional attention mask [batch, seqLen, kvLen] (currently unused, causal mask is config-driven)
//
// Returns:
//   - *tensor.Tensor: Output tensor [batch, seqLen, numHeads, headDim]
//
// Example:
//
//	Q := tensor.Randn[float32](tensor.Shape{2, 128, 8, 64}, backend)
//	K := tensor.Randn[float32](tensor.Shape{2, 128, 8, 64}, backend)
//	V := tensor.Randn[float32](tensor.Shape{2, 128, 8, 64}, backend)
//	output := fa.Forward(Q, K, V, nil)
func (fa *FlashAttention[T, B]) Forward(
	q, k, v, _ *tensor.Tensor[T, B],
) *tensor.Tensor[T, B] {
	// Validate inputs
	if len(q.Shape()) != 4 || len(k.Shape()) != 4 || len(v.Shape()) != 4 {
		panic("FlashAttention: Q, K, V must be 4D [batch, seq, numHeads, headDim]")
	}

	batch := q.Shape()[0]
	seqLen := q.Shape()[1]
	kvLen := k.Shape()[1]
	numHeads := q.Shape()[2]
	headDim := q.Shape()[3]

	if numHeads != fa.config.NumHeads || headDim != fa.config.HeadDim {
		panic("FlashAttention: numHeads or headDim mismatch with config")
	}

	// Week 2: GPU path - detect WebGPU backend and use GPU kernel
	if fa.backend.Name() == "webgpu" {
		// Type assert to WebGPU backend interface
		type webgpuBackend interface {
			FlashAttentionGPU(
				q, k, v *tensor.RawTensor,
				scale float32,
				causal bool,
				blockSize int,
			) (*tensor.RawTensor, error)
		}

		if gpuBackend, ok := any(fa.backend).(webgpuBackend); ok {
			// Execute on GPU
			outputRaw, err := gpuBackend.FlashAttentionGPU(
				q.Raw(), k.Raw(), v.Raw(),
				fa.scale,
				fa.config.CausalMask,
				fa.config.BlockSize,
			)
			if err != nil {
				panic("FlashAttention: GPU execution failed: " + err.Error())
			}

			// Wrap in typed tensor
			return tensor.New[T, B](outputRaw, fa.backend)
		}
	}

	// CPU fallback path
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()

	// Convert T to float32 for CPU computation
	qFloat := convertToFloat32(qData)
	kFloat := convertToFloat32(kData)
	vFloat := convertToFloat32(vData)

	outputFloat := flashAttentionCPU(
		qFloat, kFloat, vFloat,
		batch, seqLen, kvLen, numHeads, headDim,
		fa.scale,
		fa.config.CausalMask,
		fa.config.BlockSize,
	)

	// Convert back to T and wrap in tensor
	outputData := convertFromFloat32[T](outputFloat)
	outputTensor, err := tensor.FromSlice[T](
		outputData,
		tensor.Shape{batch, seqLen, numHeads, headDim},
		fa.backend,
	)
	if err != nil {
		panic("FlashAttention: failed to create output tensor: " + err.Error())
	}

	return outputTensor
}

// flashAttentionCPU is the CPU reference implementation.
//
// Uses tiled computation with online softmax to achieve O(N) memory.
//
// Algorithm outline:
//
//	For each query position i:
//	  1. Initialize OnlineSoftmax accumulator
//	  2. Divide K,V into blocks of size blockSize
//	  3. For each block j:
//	     - Compute scores[j] = Q[i] @ K[j]^T / sqrt(d)
//	     - Apply causal mask if i < j (for causal attention)
//	     - Update online softmax with (scores[j], V[j])
//	  4. Normalize and store output[i]
//
// Parameters:
//   - q: [batch * seqLen * numHeads * headDim] flattened query.
//   - k: [batch * kvLen * numHeads * headDim] flattened key.
//   - v: [batch * kvLen * numHeads * headDim] flattened value.
//   - batch, seqLen, kvLen, numHeads, headDim: Shape parameters.
//   - scale: 1/sqrt(headDim) scaling factor.
//   - causal: Whether to apply causal masking.
//   - blockSize: Tile size for blocked computation.
//
// Returns:
//   - []float32: [batch * seqLen * numHeads * headDim] flattened output.
//
//nolint:gocognit // High complexity is inherent to Flash Attention tiled algorithm with nested loops for batch/head/query/kv blocks
func flashAttentionCPU(
	q, k, v []float32,
	batch, seqLen, kvLen, numHeads, headDim int,
	scale float32,
	causal bool,
	blockSize int,
) []float32 {
	output := make([]float32, batch*seqLen*numHeads*headDim)

	// Process each batch and head independently
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Process query positions in blocks
			for qStart := 0; qStart < seqLen; qStart += blockSize {
				qEnd := min(qStart+blockSize, seqLen)
				qBlockSize := qEnd - qStart

				// For each query in this block
				for qIdx := 0; qIdx < qBlockSize; qIdx++ {
					i := qStart + qIdx

					// Initialize online softmax for this query
					softmax := NewOnlineSoftmax(headDim)

					// Iterate over K,V blocks
					for kvStart := 0; kvStart < kvLen; kvStart += blockSize {
						kvEnd := min(kvStart+blockSize, kvLen)
						kvBlockSize := kvEnd - kvStart

						// Compute attention scores for this K,V block
						scores := make([]float32, kvBlockSize)
						for kvIdx := 0; kvIdx < kvBlockSize; kvIdx++ {
							j := kvStart + kvIdx

							// Apply causal mask: future positions get -inf
							if causal && j > i {
								scores[kvIdx] = float32(math.Inf(-1))
								continue
							}

							// Compute Q[i] @ K[j]^T
							score := float32(0)
							for d := 0; d < headDim; d++ {
								qOffset := b*seqLen*numHeads*headDim + i*numHeads*headDim + h*headDim + d
								kOffset := b*kvLen*numHeads*headDim + j*numHeads*headDim + h*headDim + d
								score += q[qOffset] * k[kOffset]
							}
							scores[kvIdx] = score * scale
						}

						// Extract V block: [kvBlockSize, headDim]
						values := make([]float32, kvBlockSize*headDim)
						for kvIdx := 0; kvIdx < kvBlockSize; kvIdx++ {
							j := kvStart + kvIdx
							for d := 0; d < headDim; d++ {
								vOffset := b*kvLen*numHeads*headDim + j*numHeads*headDim + h*headDim + d
								values[kvIdx*headDim+d] = v[vOffset]
							}
						}

						// Update online softmax with this block
						softmax.Update(scores, values)
					}

					// Normalize and store output for this query
					result := softmax.Normalize()
					for d := 0; d < headDim; d++ {
						outOffset := b*seqLen*numHeads*headDim + i*numHeads*headDim + h*headDim + d
						output[outOffset] = result[d]
					}
				}
			}
		}
	}

	return output
}

// Helper functions for type conversion.

// convertToFloat32 converts generic DType slice to float32.
func convertToFloat32[T tensor.DType](data []T) []float32 {
	result := make([]float32, len(data))
	for i, v := range data {
		// Use type assertion to handle different types
		switch val := any(v).(type) {
		case float32:
			result[i] = val
		case float64:
			result[i] = float32(val)
		case int32:
			result[i] = float32(val)
		case int64:
			result[i] = float32(val)
		case uint8:
			result[i] = float32(val)
		case bool:
			if val {
				result[i] = 1.0
			} else {
				result[i] = 0.0
			}
		}
	}
	return result
}

// convertFromFloat32 converts float32 slice back to generic DType.
func convertFromFloat32[T tensor.DType](data []float32) []T {
	result := make([]T, len(data))
	var zero T
	for i, v := range data {
		// Use type assertion to handle different types
		switch any(zero).(type) {
		case float32:
			result[i] = any(v).(T)
		case float64:
			result[i] = any(float64(v)).(T)
		case int32:
			result[i] = any(int32(v)).(T)
		case int64:
			result[i] = any(int64(v)).(T)
		case uint8:
			result[i] = any(uint8(v)).(T)
		case bool:
			result[i] = any(v >= 0.5).(T)
		}
	}
	return result
}
