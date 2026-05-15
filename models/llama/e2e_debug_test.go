package llama

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestE2E_KnownWeights builds a minimal LLaMA (1 layer, dim=4, 2 heads)
// with hand-set weights, runs forward, and verifies each intermediate step.
func TestE2E_KnownWeights(t *testing.T) {
	backend := autodiff.New(cpu.New())

	cfg := Config{
		VocabSize:   4,
		HiddenSize:  4,
		NumLayers:   1,
		NumHeads:    2,
		NumKVHeads:  2,
		HeadDim:     2,
		FFNSize:     8,
		MaxSeqLen:   8,
		RopeTheta:   10000,
		NormEpsilon: 1e-5,
	}

	nn.SetSeed(0)
	model := NewModel(cfg, backend)

	// ── Set all weights to identity-like values for traceability ──

	// Embedding: each token i → [i+1, i+1, i+1, i+1] * 0.5
	setWeightData(t, model.Embed.Weight, []float32{
		0.0, 0.0, 0.0, 0.0, // token 0
		0.5, 0.5, 0.5, 0.5, // token 1
		1.0, 1.0, 1.0, 1.0, // token 2
		1.5, 1.5, 1.5, 1.5, // token 3
	})

	// RMSNorm gamma = [1, 1, 1, 1] (already default, but explicit)
	setWeightData(t, model.Layers[0].AttnNorm.Gamma, []float32{1, 1, 1, 1})
	setWeightData(t, model.Layers[0].FFNNorm.Gamma, []float32{1, 1, 1, 1})
	setWeightData(t, model.Norm.Gamma, []float32{1, 1, 1, 1})

	// Q, K, V projections = identity [4, 4]
	identity4 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	setLinearWeight(t, model.Layers[0].QProj, identity4)
	setLinearWeight(t, model.Layers[0].KProj, identity4)
	setLinearWeight(t, model.Layers[0].VProj, identity4)
	setLinearWeight(t, model.Layers[0].OProj, identity4)

	// LM Head = identity [4, 4]
	setLinearWeight(t, model.Head, identity4)

	// ── Step 1: Embedding ──
	input, err := tensor.FromSlice[int32]([]int32{1, 2}, tensor.Shape{1, 2}, backend)
	require.NoError(t, err)

	embedded := model.Embed.Forward(input)
	embData := embedded.Data()
	t.Logf("Embedding output: %v", embData)

	// Token 1 → [0.5, 0.5, 0.5, 0.5], Token 2 → [1.0, 1.0, 1.0, 1.0]
	assert.InDelta(t, 0.5, float64(embData[0]), 1e-4, "embed[0][0]")
	assert.InDelta(t, 1.0, float64(embData[4]), 1e-4, "embed[1][0]")

	// ── Step 2: RMSNorm on embedding ──
	// RMS of [0.5, 0.5, 0.5, 0.5] = 0.5, normalized = [1, 1, 1, 1]
	// RMS of [1.0, 1.0, 1.0, 1.0] = 1.0, normalized = [1, 1, 1, 1]
	flat := embedded.Reshape(2, 4)
	normed := model.Layers[0].AttnNorm.Forward(flat)
	normData := normed.Data()
	t.Logf("AttnNorm output: %v", normData)

	assert.InDelta(t, 1.0, float64(normData[0]), 1e-3, "norm[0][0] should be ~1.0")
	assert.InDelta(t, 1.0, float64(normData[4]), 1e-3, "norm[1][0] should be ~1.0")

	// ── Step 3: Full forward pass ──
	logits := model.Forward(input.Raw(), nil, 0)
	logitsData := logits.AsFloat32()

	t.Logf("Logits shape: %v", logits.Shape())
	t.Logf("Logits: %v", logitsData)

	// With identity projections, the output should be traceable.
	// Key: logits should NOT be all zeros or all same value.
	var logitMax float32
	for _, v := range logitsData {
		if v > logitMax {
			logitMax = v
		}
	}
	t.Logf("Max logit: %.4f", logitMax)
	assert.Greater(t, float64(logitMax), 0.0, "logits should have positive values")
}

// TestE2E_SelfAttention_IdentityProjection verifies attention with identity Q/K/V projections.
func TestE2E_SelfAttention_IdentityProjection(t *testing.T) {
	backend := autodiff.New(cpu.New())

	cfg := Config{
		VocabSize:   4,
		HiddenSize:  4,
		NumLayers:   1,
		NumHeads:    2,
		NumKVHeads:  2,
		HeadDim:     2,
		FFNSize:     8,
		MaxSeqLen:   8,
		RopeTheta:   10000,
		NormEpsilon: 1e-5,
	}

	nn.SetSeed(0)
	model := NewModel(cfg, backend)
	layer := model.Layers[0]

	// Identity projections.
	identity4 := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	setLinearWeight(t, layer.QProj, identity4)
	setLinearWeight(t, layer.KProj, identity4)
	setLinearWeight(t, layer.VProj, identity4)
	setLinearWeight(t, layer.OProj, identity4)

	// Input: [1, 1, 4] — single token, normalized (RMS=1).
	x, err := tensor.FromSlice[float32](
		[]float32{0.5, -0.5, 0.5, -0.5}, // RMS = 0.5
		tensor.Shape{1, 1, 4}, backend)
	require.NoError(t, err)

	// Self-attention on single token: Q@K^T = scalar, softmax = 1.0, output = V.
	// With identity projection: V = input, so attention output = input.
	// But RoPE rotates Q and K, so Q@K^T may differ from input@input^T.
	out, attn, ffn := layer.DebugForward(x, nil, 0)

	attnData := attn.Data()
	ffnData := ffn.Data()
	outData := out.Data()

	t.Logf("Input:       %v", x.Data())
	t.Logf("Attn output: %v (RMS=%.4f)", attnData, rmsFloat32(attnData))
	t.Logf("FFN output:  %v (RMS=%.4f)", ffnData, rmsFloat32(ffnData))
	t.Logf("Layer out:   %v (RMS=%.4f)", outData, rmsFloat32(outData))

	// Attn output should be ~same magnitude as input (identity projection + single token).
	attnRMS := rmsFloat32(attnData)
	inputRMS := rmsFloat32(x.Data())
	assert.InDelta(t, inputRMS, attnRMS, inputRMS*2,
		"attention output RMS should be similar magnitude to input with identity projections")
}

// TestE2E_LinearForward_IdentityWeight verifies Linear with identity weight does nothing.
func TestE2E_LinearForward_IdentityWeight(t *testing.T) {
	backend := autodiff.New(cpu.New())

	linear := nn.NewLinear[*autodiff.AutodiffBackend[*cpu.CPUBackend]](4, 4, backend, nn.WithBias(false))
	setLinearWeight(t, linear, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})

	x, _ := tensor.FromSlice[float32]([]float32{1, 2, 3, 4}, tensor.Shape{1, 4}, backend)
	out := linear.Forward(x)
	d := out.Data()

	// Identity: output should equal input.
	for i, want := range []float32{1, 2, 3, 4} {
		assert.InDelta(t, float64(want), float64(d[i]), 1e-5, "Linear identity [%d]", i)
	}
}

// TestE2E_LinearForward_KnownMultiplication verifies Linear produces correct MatMul.
func TestE2E_LinearForward_KnownMultiplication(t *testing.T) {
	backend := autodiff.New(cpu.New())

	// W = [[2, 0], [0, 3]] stored as [out=2, in=2]
	// Forward: x @ W^T = [1, 1] @ [[2, 0], [0, 3]]^T = [1, 1] @ [[2, 0], [0, 3]] = [2, 3]
	linear := nn.NewLinear[*autodiff.AutodiffBackend[*cpu.CPUBackend]](2, 2, backend, nn.WithBias(false))
	setLinearWeight(t, linear, []float32{2, 0, 0, 3})

	x, _ := tensor.FromSlice[float32]([]float32{1, 1}, tensor.Shape{1, 2}, backend)
	out := linear.Forward(x)
	d := out.Data()

	assert.InDelta(t, 2.0, float64(d[0]), 1e-5, "Linear known [0]")
	assert.InDelta(t, 3.0, float64(d[1]), 1e-5, "Linear known [1]")
}

// ── Helpers ──

func setWeightData[B tensor.Backend](t *testing.T, param *nn.Parameter[B], data []float32) {
	t.Helper()
	dst := param.Tensor().Raw().AsFloat32()
	require.Equal(t, len(data), len(dst), "weight data length mismatch")
	copy(dst, data)
}

func setLinearWeight[B tensor.Backend](t *testing.T, linear *nn.Linear[B], data []float32) {
	t.Helper()
	dst := linear.Weight().Tensor().Raw().AsFloat32()
	require.Equal(t, len(data), len(dst), "linear weight data length mismatch")
	copy(dst, data)
}

func rmsFloat32(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(data)))
}
