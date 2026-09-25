package nn

import (
	"fmt"
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// RotaryEncoding implements Rotary Position Embedding (RoPE).
//
// RoPE is a modern positional encoding used in LLaMA, Mistral, DeepSeek, and other
// state-of-the-art LLMs. It applies a rotation to query and key embeddings based on
// their position, allowing the model to capture relative position information.
//
// Mathematical formulation:
//
//	For position m and dimension pair (2i, 2i+1):
//	  θ_i = base^(-2i/d)  (typically base=10000)
//
//	  [q'_{2i}  ]   [cos(m·θ_i)  -sin(m·θ_i)] [q_{2i}  ]
//	  [q'_{2i+1}] = [sin(m·θ_i)   cos(m·θ_i)] [q_{2i+1}]
//
// Architecture:
//   - Pre-computes cos and sin values for all positions and dimensions
//   - Applies rotation by splitting input into even/odd pairs
//   - Supports both training (full sequence) and inference (with offset for KV-cache)
//
// Example:
//
//	config := nn.RotaryEncodingConfig{
//	    DModel:    64,     // Head dimension (typically 64-128)
//	    MaxSeqLen: 2048,   // Maximum sequence length
//	    Theta:     10000.0,
//	}
//	rope := nn.NewRotaryEncoding(config, backend)
//
//	// During training: apply to full sequence
//	q := tensor.Randn[float32](tensor.Shape{batch, heads, seq, 64}, backend)
//	q_rotated := rope.Forward(q)
//
//	// During inference with KV-cache: apply with position offset
//	q_new := tensor.Randn[float32](tensor.Shape{batch, heads, 1, 64}, backend)
//	q_rotated := rope.ForwardWithOffset(q_new, currentPosition)
type RotaryEncoding[B tensor.Backend] struct {
	FreqCos   *tensor.Tensor[float32, B] // [max_seq_len, d_model/2] - cosine values
	FreqSin   *tensor.Tensor[float32, B] // [max_seq_len, d_model/2] - sine values
	MaxSeqLen int                        // Maximum sequence length
	DModel    int                        // Model dimension (must be even)
	backend   B
}

// RotaryEncodingConfig configures a RotaryEncoding layer.
type RotaryEncodingConfig struct {
	DModel    int     // Dimension per head (typically 64-128, must be even)
	MaxSeqLen int     // Maximum sequence length (e.g., 2048, 4096)
	Theta     float64 // Base frequency for rotation (default: 10000.0)
}

// NewRotaryEncoding creates a new RotaryEncoding layer.
//
// Pre-computes cosine and sine values for all positions and dimension pairs.
//
// Parameters:
//   - cfg: Configuration for RoPE (dimension, max sequence length, theta base)
//   - backend: Computation backend
//
// Returns a new RotaryEncoding layer with pre-computed rotation matrices.
//
// Panics if DModel is not even (RoPE requires pairing dimensions).
func NewRotaryEncoding[B tensor.Backend](cfg RotaryEncodingConfig, backend B) *RotaryEncoding[B] {
	if cfg.DModel%2 != 0 {
		panic(fmt.Sprintf("RotaryEncoding: DModel must be even, got %d", cfg.DModel))
	}
	if cfg.MaxSeqLen <= 0 {
		panic(fmt.Sprintf("RotaryEncoding: MaxSeqLen must be positive, got %d", cfg.MaxSeqLen))
	}
	if cfg.Theta <= 0 {
		cfg.Theta = 10000.0 // Default theta
	}

	// Compute frequencies for each dimension pair
	// θ_i = base^(-2i/d) for i in [0, d/2)
	halfDim := cfg.DModel / 2
	freqs := make([]float32, halfDim)
	for i := 0; i < halfDim; i++ {
		// θ_i = theta^(-2i/d)
		exponent := -2.0 * float64(i) / float64(cfg.DModel)
		freqs[i] = float32(math.Pow(cfg.Theta, exponent))
	}

	// Pre-compute cos and sin for all positions
	cosData := make([]float32, cfg.MaxSeqLen*halfDim)
	sinData := make([]float32, cfg.MaxSeqLen*halfDim)

	for pos := 0; pos < cfg.MaxSeqLen; pos++ {
		for i := 0; i < halfDim; i++ {
			angle := float64(pos) * float64(freqs[i])
			idx := pos*halfDim + i
			cosData[idx] = float32(math.Cos(angle))
			sinData[idx] = float32(math.Sin(angle))
		}
	}

	// Create tensors
	freqCos, err := tensor.FromSlice[float32, B](cosData, tensor.Shape{cfg.MaxSeqLen, halfDim}, backend)
	if err != nil {
		panic(fmt.Sprintf("failed to create cos tensor: %v", err))
	}

	freqSin, err := tensor.FromSlice[float32, B](sinData, tensor.Shape{cfg.MaxSeqLen, halfDim}, backend)
	if err != nil {
		panic(fmt.Sprintf("failed to create sin tensor: %v", err))
	}

	return &RotaryEncoding[B]{
		FreqCos:   freqCos,
		FreqSin:   freqSin,
		MaxSeqLen: cfg.MaxSeqLen,
		DModel:    cfg.DModel,
		backend:   backend,
	}
}

// Forward applies rotary position embeddings to the input tensor.
//
// Supports both 3D and 4D input tensors:
//   - 3D: [batch, seq_len, d_model] - applies RoPE to entire sequence
//   - 4D: [batch, n_heads, seq_len, d_k] - applies RoPE per head (typical for attention)
//
// The rotation is applied to dimension pairs (2i, 2i+1) using pre-computed cos/sin values.
//
// Parameters:
//   - x: Input tensor [batch, seq_len, d_model] or [batch, n_heads, seq_len, d_k]
//
// Returns tensor with same shape as input, with rotary embeddings applied.
//
// Panics if sequence length exceeds MaxSeqLen or if last dimension doesn't match DModel.
func (r *RotaryEncoding[B]) Forward(x *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B] {
	return r.ForwardWithOffset(x, 0)
}

// ForwardWithOffset applies rotary embeddings with a position offset.
//
// This is useful for incremental decoding with KV-cache, where new tokens are generated
// one at a time but need position embeddings that account for previous tokens.
//
// Parameters:
//   - x: Input tensor [batch, seq_len, d_model] or [batch, n_heads, seq_len, d_k]
//   - offset: Position offset (e.g., current position in KV-cache)
//
// Returns tensor with rotary embeddings applied at positions [offset, offset+seq_len).
//
// Example (KV-cache inference):
//
//	// Initial prompt: positions [0, prompt_len)
//	q_prompt := rope.Forward(q_prompt_tokens)
//
//	// Generate token 1: position [prompt_len]
//	q_new := rope.ForwardWithOffset(q_new_token, prompt_len)
//
//	// Generate token 2: position [prompt_len + 1]
//	q_new := rope.ForwardWithOffset(q_new_token, prompt_len + 1)
//
// Panics if offset + seq_len exceeds MaxSeqLen.
func (r *RotaryEncoding[B]) ForwardWithOffset(x *tensor.Tensor[float32, B], offset int) *tensor.Tensor[float32, B] {
	shape := x.Shape()
	var seqLen, dModel int

	// Determine shape format
	switch len(shape) {
	case 3:
		// [batch, seq_len, d_model]
		seqLen = shape[1]
		dModel = shape[2]
	case 4:
		// [batch, n_heads, seq_len, d_k]
		seqLen = shape[2]
		dModel = shape[3]
	default:
		panic(fmt.Sprintf("RotaryEncoding: input must be 3D or 4D, got shape %v", shape))
	}

	// Validate dimensions
	if dModel != r.DModel {
		panic(fmt.Sprintf("RotaryEncoding: expected last dimension %d, got %d", r.DModel, dModel))
	}
	if offset+seqLen > r.MaxSeqLen {
		panic(fmt.Sprintf("RotaryEncoding: offset + seq_len (%d) exceeds MaxSeqLen (%d)", offset+seqLen, r.MaxSeqLen))
	}

	// Extract cos/sin for the relevant positions [offset, offset+seqLen)
	// cos/sin shape: [max_seq_len, d_model/2]
	// Extract rows [offset:offset+seqLen] -> [seq_len, d_model/2]
	halfDim := r.DModel / 2
	cosData := r.FreqCos.Data()
	sinData := r.FreqSin.Data()

	posCosSinShape := tensor.Shape{seqLen, halfDim}

	// Extract cos for positions
	posCosData := make([]float32, seqLen*halfDim)
	posSinData := make([]float32, seqLen*halfDim)
	for pos := 0; pos < seqLen; pos++ {
		srcIdx := (offset + pos) * halfDim
		dstIdx := pos * halfDim
		copy(posCosData[dstIdx:dstIdx+halfDim], cosData[srcIdx:srcIdx+halfDim])
		copy(posSinData[dstIdx:dstIdx+halfDim], sinData[srcIdx:srcIdx+halfDim])
	}

	posCos, err := tensor.FromSlice[float32, B](posCosData, posCosSinShape, r.backend)
	if err != nil {
		panic(fmt.Sprintf("failed to create position cos tensor: %v", err))
	}

	posSin, err := tensor.FromSlice[float32, B](posSinData, posCosSinShape, r.backend)
	if err != nil {
		panic(fmt.Sprintf("failed to create position sin tensor: %v", err))
	}

	// Apply rotation to input
	return r.applyRotation(x, posCos, posSin)
}

// applyRotation applies the rotation matrix using cos/sin values.
//
// Uses the rotate-half convention (LLaMA/GPT-NeoX standard):
// pairs (x[i], x[i+d/2]) rather than interleaved (x[2i], x[2i+1]).
//
// Rotation formula implemented via backend tensor ops so every operation
// is recorded on the autodiff tape and gradients flow through RoPE:
//
//  1. Split x into first half and second half along the last dimension.
//  2. Broadcast cos/sin from [seq, d/2] to match x's full shape.
//  3. rotated_first  = first_half  * cos - second_half * sin
//  4. rotated_second = second_half * cos + first_half  * sin
//  5. Concatenate rotated_first and rotated_second along the last dimension.
//
// Supports both 3D [batch, seq, d] and 4D [batch, heads, seq, d] inputs.
func (r *RotaryEncoding[B]) applyRotation(
	x *tensor.Tensor[float32, B],
	posCos *tensor.Tensor[float32, B],
	posSin *tensor.Tensor[float32, B],
) *tensor.Tensor[float32, B] {
	shape := x.Shape()
	is3D := len(shape) == 3

	// Split x into [first_half, second_half] along the last dimension.
	// Chunk requires the split dimension to be divisible by n=2.
	// x shape: [batch, seq, d] → each half is [batch, seq, d/2]
	// x shape: [batch, heads, seq, d] → each half is [batch, heads, seq, d/2]
	halves := x.Chunk(2, -1)
	firstHalf := halves[0]  // [..., :d/2]
	secondHalf := halves[1] // [..., d/2:]

	// posCos/posSin have shape [seq, d/2].
	// Broadcast to match x's shape so element-wise Mul works correctly.
	var cos, sin *tensor.Tensor[float32, B]
	if is3D {
		// Target: [batch, seq, d/2]
		batchSize := shape[0]
		seqLen := shape[1]
		halfDim := r.DModel / 2
		// [seq, d/2] → [1, seq, d/2] → [batch, seq, d/2]
		cos = posCos.Unsqueeze(0).Expand(tensor.Shape{batchSize, seqLen, halfDim})
		sin = posSin.Unsqueeze(0).Expand(tensor.Shape{batchSize, seqLen, halfDim})
	} else {
		// Target: [batch, heads, seq, d/2]
		batchSize := shape[0]
		numHeads := shape[1]
		seqLen := shape[2]
		halfDim := r.DModel / 2
		// [seq, d/2] → [1, seq, d/2] → [1, 1, seq, d/2] → [batch, heads, seq, d/2]
		cos = posCos.Unsqueeze(0).Unsqueeze(0).Expand(tensor.Shape{batchSize, numHeads, seqLen, halfDim})
		sin = posSin.Unsqueeze(0).Unsqueeze(0).Expand(tensor.Shape{batchSize, numHeads, seqLen, halfDim})
	}

	// Apply rotate-half: recorded on tape via Mul, Sub, Add.
	//
	// Each half is used twice (once for rotatedFirst, once for rotatedSecond).
	// The CPU backend short-circuits to in-place mutation when a.IsUnique() is
	// true, which would corrupt the raw buffer before the second use reads it.
	// ForceNonUnique increments the refcount so IsUnique() returns false for the
	// duration of this function, forcing the backend to allocate fresh output
	// tensors. The deferred restores drop the extra refcount when we return.
	restoreFirst := firstHalf.Raw().ForceNonUnique()
	defer restoreFirst()
	restoreSecond := secondHalf.Raw().ForceNonUnique()
	defer restoreSecond()

	rotatedFirst := firstHalf.Mul(cos).Sub(secondHalf.Mul(sin))
	rotatedSecond := secondHalf.Mul(cos).Add(firstHalf.Mul(sin))

	// Concatenate along last dimension to restore original shape.
	return tensor.Cat([]*tensor.Tensor[float32, B]{rotatedFirst, rotatedSecond}, -1)
}
