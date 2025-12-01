// Package nn provides neural network modules and layers for building deep learning models.
// It includes activations, attention mechanisms, normalization layers, and more.
package nn

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// ScaledDotProductAttention computes attention scores using the scaled dot-product mechanism.
//
// This is the core attention mechanism used in transformers, implementing:
//
//	Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//
// Where:
//   - Q (query): what information we're looking for [batch, heads, seq_q, head_dim]
//   - K (key): what information is available [batch, heads, seq_k, head_dim]
//   - V (value): the actual information to retrieve [batch, heads, seq_k, head_dim]
//   - mask: optional attention mask (additive, -inf for masked positions)
//   - scale: scaling factor (typically 1/sqrt(head_dim)), 0 for auto-compute
//
// Parameters:
//   - query: Query tensor [batch, heads, seq_q, head_dim]
//   - key: Key tensor [batch, heads, seq_k, head_dim]
//   - value: Value tensor [batch, heads, seq_k, head_dim]
//   - mask: Optional attention mask [batch, 1, seq_q, seq_k] or nil (additive mask, -inf for masked)
//   - scale: Scaling factor (0 for auto-compute as 1/sqrt(head_dim))
//
// Returns:
//   - output: Attended values [batch, heads, seq_q, head_dim]
//   - weights: Attention weights [batch, heads, seq_q, seq_k]
//
// Example:
//
//	backend := autodiff.New(cpu.New())
//	Q := tensor.Randn[float32](tensor.Shape{2, 8, 10, 64}, backend)  // batch=2, heads=8, seq=10, dim=64
//	K := tensor.Randn[float32](tensor.Shape{2, 8, 10, 64}, backend)
//	V := tensor.Randn[float32](tensor.Shape{2, 8, 10, 64}, backend)
//	output, weights := nn.ScaledDotProductAttention(Q, K, V, nil, 0)  // auto-scale
func ScaledDotProductAttention[B tensor.Backend](
	query, key, value *tensor.Tensor[float32, B],
	mask *tensor.Tensor[float32, B],
	scale float32,
) (*tensor.Tensor[float32, B], *tensor.Tensor[float32, B]) {
	validateAttentionInputs(query, key, value)

	// Auto-compute scale if not provided
	qHeadDim := query.Shape()[3]
	if scale == 0 {
		scale = float32(1.0 / math.Sqrt(float64(qHeadDim)))
	}

	// Extract dimensions
	batch := query.Shape()[0]
	heads := query.Shape()[1]
	seqQ := query.Shape()[2]
	seqK := key.Shape()[2]

	// 1. Compute attention scores: QK^T
	scores := batchedMatMulQK(query, key, batch, heads, seqQ, seqK, qHeadDim)

	// 2. Scale
	scores = scores.MulScalar(scale)

	// 3. Apply mask (if provided)
	if mask != nil {
		scores = scores.Add(mask)
	}

	// 4. Softmax along last dimension (over keys)
	weights := scores.Softmax(-1)

	// 5. Compute output: weights @ V
	output := batchedMatMulWV(weights, value, batch, heads, seqQ, seqK, qHeadDim)

	return output, weights
}

// validateAttentionInputs validates the input tensors for attention.
func validateAttentionInputs[B tensor.Backend](
	query, key, value *tensor.Tensor[float32, B],
) {
	if len(query.Shape()) != 4 {
		panic("ScaledDotProductAttention: query must be 4D [batch, heads, seq_q, head_dim]")
	}
	if len(key.Shape()) != 4 {
		panic("ScaledDotProductAttention: key must be 4D [batch, heads, seq_k, head_dim]")
	}
	if len(value.Shape()) != 4 {
		panic("ScaledDotProductAttention: value must be 4D [batch, heads, seq_k, head_dim]")
	}

	// Q and K must have same head_dim
	qHeadDim := query.Shape()[3]
	kHeadDim := key.Shape()[3]
	if qHeadDim != kHeadDim {
		panic("ScaledDotProductAttention: query and key must have same head_dim")
	}

	// K and V must have same seq length
	kSeqLen := key.Shape()[2]
	vSeqLen := value.Shape()[2]
	if kSeqLen != vSeqLen {
		panic("ScaledDotProductAttention: key and value must have same seq length")
	}
}

// batchedMatMulQK computes batched matrix multiplication for Q @ K^T.
// Returns scores of shape [batch, heads, seq_q, seq_k].
func batchedMatMulQK[B tensor.Backend](
	query, key *tensor.Tensor[float32, B],
	batch, heads, seqQ, seqK, headDim int,
) *tensor.Tensor[float32, B] {
	backend := query.Backend()

	// Reshape to [batch*heads, seq, dim]
	qReshaped := query.Reshape(batch*heads, seqQ, headDim)
	kReshaped := key.Reshape(batch*heads, seqK, headDim)

	scoresData := make([]float32, batch*heads*seqQ*seqK)

	// Compute matmul for each batch*head
	for bh := 0; bh < batch*heads; bh++ {
		// Extract Q and K slices for this batch*head
		qSlice := extractSlice2D(qReshaped, bh, seqQ, headDim, backend)
		kSlice := extractSlice2D(kReshaped, bh, seqK, headDim, backend)

		// K^T and MatMul
		kSliceT := kSlice.T()
		scoresSlice := qSlice.MatMul(kSliceT)

		// Copy to output
		copySliceData(scoresSlice.Data(), scoresData, bh*seqQ*seqK)
	}

	scores, err := tensor.FromSlice[float32](scoresData, tensor.Shape{batch, heads, seqQ, seqK}, backend)
	if err != nil {
		panic("batchedMatMulQK: failed to create scores tensor")
	}

	return scores
}

// batchedMatMulWV computes batched matrix multiplication for weights @ V.
// Returns output of shape [batch, heads, seq_q, head_dim].
func batchedMatMulWV[B tensor.Backend](
	weights, value *tensor.Tensor[float32, B],
	batch, heads, seqQ, seqK, headDim int,
) *tensor.Tensor[float32, B] {
	backend := value.Backend()

	// Reshape to [batch*heads, seq, dim]
	vReshaped := value.Reshape(batch*heads, seqK, headDim)
	wReshaped := weights.Reshape(batch*heads, seqQ, seqK)

	outputData := make([]float32, batch*heads*seqQ*headDim)

	// Compute matmul for each batch*head
	for bh := 0; bh < batch*heads; bh++ {
		// Extract W and V slices for this batch*head
		wSlice := extractSlice2D(wReshaped, bh, seqQ, seqK, backend)
		vSlice := extractSlice2D(vReshaped, bh, seqK, headDim, backend)

		// MatMul
		outSlice := wSlice.MatMul(vSlice)

		// Copy to output
		copySliceData(outSlice.Data(), outputData, bh*seqQ*headDim)
	}

	output, err := tensor.FromSlice[float32](outputData, tensor.Shape{batch, heads, seqQ, headDim}, backend)
	if err != nil {
		panic("batchedMatMulWV: failed to create output tensor")
	}

	return output
}

// extractSlice2D extracts a 2D slice from a 3D tensor at the given index.
func extractSlice2D[B tensor.Backend](
	t *tensor.Tensor[float32, B],
	index, dim1, dim2 int,
	backend B,
) *tensor.Tensor[float32, B] {
	slice := tensor.Zeros[float32](tensor.Shape{dim1, dim2}, backend)
	tData := t.Data()
	sliceData := slice.Data()

	offset := index * dim1 * dim2
	for i := 0; i < dim1*dim2; i++ {
		sliceData[i] = tData[offset+i]
	}

	return slice
}

// copySliceData copies data from source to destination starting at offset.
func copySliceData(src, dst []float32, offset int) {
	for i := 0; i < len(src); i++ {
		dst[offset+i] = src[i]
	}
}

// CausalMask creates a causal (autoregressive) attention mask.
//
// In causal attention, each position can only attend to earlier positions
// (including itself). This is used in autoregressive models like GPT.
//
// Returns a mask tensor where:
//   - Upper triangle (future positions) = -inf (masked out)
//   - Lower triangle + diagonal (past + current) = 0 (allowed)
//
// The mask is applied additively to attention scores before softmax:
//
//	scores = QK^T / sqrt(d_k) + mask
//
// Shape: [1, 1, seq_len, seq_len] (broadcastable to [batch, heads, seq, seq])
//
// Example:
//
//	// For seq_len=4:
//	// [[0,   -inf, -inf, -inf],
//	//  [0,   0,    -inf, -inf],
//	//  [0,   0,    0,    -inf],
//	//  [0,   0,    0,    0   ]]
//
//	backend := cpu.New()
//	mask := nn.CausalMask(10, backend)  // [1, 1, 10, 10]
func CausalMask[B tensor.Backend](seqLen int, backend B) *tensor.Tensor[float32, B] {
	// Create a mask of shape [1, 1, seq_len, seq_len]
	mask := tensor.Zeros[float32](tensor.Shape{1, 1, seqLen, seqLen}, backend)

	// Fill upper triangle with -inf
	negInf := float32(math.Inf(-1))
	data := mask.Data()

	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			// Index in flattened array: [0, 0, i, j]
			// For shape [1, 1, seq_len, seq_len]:
			// index = 0*1*seq_len*seq_len + 0*seq_len*seq_len + i*seq_len + j
			idx := i*seqLen + j
			data[idx] = negInf
		}
	}

	return mask
}
