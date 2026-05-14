package llama

import (
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReshapeTranspose4D verifies that Reshape+Transpose for attention heads
// produces the correct layout: [batch, seq, heads, dim] → [batch, heads, seq, dim].
func TestReshapeTranspose4D(t *testing.T) {
	backend := autodiff.New(cpu.New())

	// [1, 2, 6] — batch=1, seq=2, dim=6 (3 heads × 2 head_dim)
	data := []float32{
		// seq pos 0: head0=[1,2], head1=[3,4], head2=[5,6]
		1, 2, 3, 4, 5, 6,
		// seq pos 1: head0=[7,8], head1=[9,10], head2=[11,12]
		7, 8, 9, 10, 11, 12,
	}
	x, err := tensor.FromSlice[float32](data, tensor.Shape{1, 2, 6}, backend)
	require.NoError(t, err)

	// Reshape to [1, 2, 3, 2] = [batch, seq, heads, head_dim]
	r := x.Reshape(1, 2, 3, 2)
	assert.Equal(t, tensor.Shape{1, 2, 3, 2}, r.Shape())

	// Transpose to [1, 3, 2, 2] = [batch, heads, seq, head_dim]
	tr := r.Transpose(0, 2, 1, 3)
	assert.Equal(t, tensor.Shape{1, 3, 2, 2}, tr.Shape())

	d := tr.Data()

	// Expected layout after transpose:
	// Head 0: seq0=[1,2], seq1=[7,8]
	// Head 1: seq0=[3,4], seq1=[9,10]
	// Head 2: seq0=[5,6], seq1=[11,12]
	expected := []float32{
		1, 2, 7, 8, // head 0
		3, 4, 9, 10, // head 1
		5, 6, 11, 12, // head 2
	}

	require.Equal(t, len(expected), len(d), "data length")
	for i := range expected {
		assert.Equal(t, expected[i], d[i], "position %d: got %f, want %f", i, d[i], expected[i])
	}
}

// TestAttentionSingleToken verifies that self-attention on a single token
// produces correct output (Q@K^T = scalar, softmax([x]) = [1.0], output = V).
func TestAttentionSingleToken(t *testing.T) {
	backend := autodiff.New(cpu.New())

	// Q = K = V = [1, 1, 1, 4] (batch=1, heads=1, seq=1, dim=4)
	data := []float32{1, 0, 0, 0}
	q, _ := tensor.FromSlice[float32](data, tensor.Shape{1, 1, 1, 4}, backend)
	k, _ := tensor.FromSlice[float32](data, tensor.Shape{1, 1, 1, 4}, backend)
	v, _ := tensor.FromSlice[float32]([]float32{0.5, 0.5, 0.5, 0.5}, tensor.Shape{1, 1, 1, 4}, backend)

	scale := float32(0.5) // 1/sqrt(4)
	out, weights := nn.ScaledDotProductAttention(q, k, v, nil, scale)

	// Single token: weights = softmax([q@k^T * scale]) = softmax([0.5]) = [1.0]
	wData := weights.Data()
	assert.InDelta(t, 1.0, float64(wData[0]), 1e-5, "single token weight should be 1.0")

	// Output = 1.0 * V = V
	oData := out.Data()
	for i, want := range []float32{0.5, 0.5, 0.5, 0.5} {
		assert.InDelta(t, float64(want), float64(oData[i]), 1e-5, "output[%d]", i)
	}
}

// TestCausalMaskPreventsAttentionToFuture verifies causal masking works.
func TestCausalMaskPreventsAttentionToFuture(t *testing.T) {
	backend := autodiff.New(cpu.New())

	// 3 tokens, 1 head, dim=2
	// Q[0] should attend only to K[0]
	// Q[1] should attend to K[0], K[1]
	// Q[2] should attend to K[0], K[1], K[2]
	q, _ := tensor.FromSlice[float32]([]float32{
		1, 0, // token 0
		0, 1, // token 1
		1, 1, // token 2
	}, tensor.Shape{1, 1, 3, 2}, backend)
	k, _ := tensor.FromSlice[float32]([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, tensor.Shape{1, 1, 3, 2}, backend)
	v, _ := tensor.FromSlice[float32]([]float32{
		10, 0,
		0, 20,
		30, 30,
	}, tensor.Shape{1, 1, 3, 2}, backend)

	mask := nn.CausalMask(3, backend)
	_, weights := nn.ScaledDotProductAttention(q, k, v, mask, 1.0)
	w := weights.Data()

	// Token 0 should have weight 1.0 on position 0, 0.0 on 1 and 2.
	assert.InDelta(t, 1.0, float64(w[0]), 1e-4, "w[0→0] should be 1.0")
	assert.InDelta(t, 0.0, float64(w[1]), 1e-4, "w[0→1] should be 0.0 (masked)")
	assert.InDelta(t, 0.0, float64(w[2]), 1e-4, "w[0→2] should be 0.0 (masked)")

	// Token 1: can attend to 0 and 1, not 2.
	assert.InDelta(t, 0.0, float64(w[5]), 1e-4, "w[1→2] should be 0.0 (masked)")
	assert.Greater(t, float64(w[3]+w[4]), 0.99, "w[1→0]+w[1→1] should sum to ~1.0")

	// Token 2: can attend to all 3.
	total := float64(w[6] + w[7] + w[8])
	assert.InDelta(t, 1.0, total, 1e-4, "w[2→*] should sum to 1.0")
}
