package llama

import (
	"fmt"
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func adBackend() *autodiff.AutodiffBackend[*cpu.CPUBackend] {
	return autodiff.New(cpu.New())
}

// TestRMSNorm3D_OutputMagnitude verifies RMSNorm normalizes 3D input correctly.
// After RMSNorm with gamma=1, output RMS should be approximately 1.
func TestRMSNorm3D_OutputMagnitude(t *testing.T) {
	backend := adBackend()

	norm := nn.NewRMSNorm[*autodiff.AutodiffBackend[*cpu.CPUBackend]](64, 1e-5, backend)

	// Input: [1, 1, 64] with known values — std=10 (intentionally large).
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) - 32 // range [-32, 31]
	}
	x, err := tensor.FromSlice[float32](data, tensor.Shape{1, 1, 64}, backend)
	require.NoError(t, err)

	out := norm.Forward(x)
	outData := out.Data()

	// RMS of output should be approximately 1.0 (gamma=1).
	var sumSq float64
	for _, v := range outData {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / float64(len(outData)))

	t.Logf("Input RMS: %.4f", math.Sqrt(sqMean(data)))
	t.Logf("Output RMS: %.4f (expect ~1.0)", rms)
	assert.InDelta(t, 1.0, rms, 0.1, "RMSNorm output RMS should be ~1.0")
}

// TestLinearForward_OutputMagnitude verifies Linear output scale for typical LLaMA weights.
func TestLinearForward_OutputMagnitude(t *testing.T) {
	backend := adBackend()

	nn.SetSeed(42)
	linear := nn.NewLinear[*autodiff.AutodiffBackend[*cpu.CPUBackend]](64, 64, backend, nn.WithBias(false))

	// Input: normalized vector (RMS=1), which is what Linear receives after RMSNorm.
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(math.Sin(float64(i))) / float32(math.Sqrt(32)) // RMS≈1
	}
	x, err := tensor.FromSlice[float32](data, tensor.Shape{1, 64}, backend)
	require.NoError(t, err)

	out := linear.Forward(x)
	outData := out.Data()

	outRMS := math.Sqrt(sqMean(outData))
	t.Logf("Xavier Linear(64→64) output RMS: %.4f", outRMS)

	// Xavier init std ≈ sqrt(2/(in+out)) = sqrt(2/128) ≈ 0.125
	// Output RMS ≈ input_RMS * weight_std * sqrt(in) = 1.0 * 0.125 * 8 ≈ 1.0
	assert.Less(t, outRMS, 10.0, "Linear output RMS should be moderate, not explosive")
}

// TestSwiGLUFFN_OutputMagnitude verifies SwiGLU FFN doesn't amplify signal.
func TestSwiGLUFFN_OutputMagnitude(t *testing.T) {
	backend := adBackend()

	nn.SetSeed(42)
	ffn := nn.NewSwiGLUFFN[*autodiff.AutodiffBackend[*cpu.CPUBackend]](nn.SwiGLUFFNConfig{
		EmbedDim: 64,
		FFNDim:   128,
	}, backend)

	// Normalized input (RMS≈1), 3D [batch, seq, dim].
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(math.Cos(float64(i))) * 0.125 // small magnitude
	}
	x, err := tensor.FromSlice[float32](data, tensor.Shape{1, 1, 64}, backend)
	require.NoError(t, err)

	out := ffn.Forward(x)
	outData := out.Data()

	outRMS := math.Sqrt(sqMean(outData))
	t.Logf("SwiGLU FFN output RMS: %.4f (input RMS: %.4f)", outRMS, math.Sqrt(sqMean(data)))
	assert.Less(t, outRMS, 10.0, "FFN should not amplify signal dramatically")
}

// TestSingleLayerForward_Magnitude traces one transformer layer with known input.
func TestSingleLayerForward_Magnitude(t *testing.T) {
	backend := adBackend()

	cfg := Config{
		VocabSize:   100,
		HiddenSize:  64,
		NumLayers:   1,
		NumHeads:    4,
		NumKVHeads:  4,
		HeadDim:     16,
		FFNSize:     128,
		MaxSeqLen:   32,
		RopeTheta:   10000,
		NormEpsilon: 1e-5,
	}

	nn.SetSeed(42)
	model := NewModel(cfg, backend)

	// Input: single token.
	input, err := tensor.FromSlice[int32]([]int32{5}, tensor.Shape{1, 1}, backend)
	require.NoError(t, err)

	// Get embedding.
	embedded := model.Embed.Forward(input)
	embData := embedded.Data()
	embRMS := math.Sqrt(sqMean(embData))
	t.Logf("Embedding RMS: %.4f", embRMS)

	// Full forward.
	logits := model.Forward(input.Raw(), nil, 0)
	logitsData := logits.AsFloat32()

	var maxLogit float32
	for _, v := range logitsData {
		if v > maxLogit || maxLogit == 0 {
			maxLogit = v
		}
	}
	logitRMS := math.Sqrt(sqMean(logitsData))

	t.Logf("Logits: max=%.2f, RMS=%.4f, len=%d", maxLogit, logitRMS, len(logitsData))
	assert.Less(t, float64(maxLogit), 100.0,
		"1-layer model with random Xavier weights should not produce logits > 100")
}

// TestFullModel_LogitRange tests that a small model produces logits in reasonable range.
func TestFullModel_LogitRange(t *testing.T) {
	backend := adBackend()

	cfg := Config{
		VocabSize:   100,
		HiddenSize:  64,
		NumLayers:   4, // 4 layers like a tiny LLM
		NumHeads:    4,
		NumKVHeads:  2,
		HeadDim:     16,
		FFNSize:     128,
		MaxSeqLen:   32,
		RopeTheta:   10000,
		NormEpsilon: 1e-5,
	}

	nn.SetSeed(42)
	model := NewModel(cfg, backend)

	input, err := tensor.FromSlice[int32]([]int32{5}, tensor.Shape{1, 1}, backend)
	require.NoError(t, err)

	logits := model.Forward(input.Raw(), nil, 0)
	logitsData := logits.AsFloat32()

	logitRMS := math.Sqrt(sqMean(logitsData))
	var maxAbs float32
	for _, v := range logitsData {
		if v > maxAbs {
			maxAbs = v
		}
		if -v > maxAbs {
			maxAbs = -v
		}
	}

	t.Logf("4-layer model logits: maxAbs=%.2f, RMS=%.4f", maxAbs, logitRMS)

	// With random Xavier weights, 4 layers, logits should be O(1)-O(10), not O(100)+.
	assert.Less(t, float64(maxAbs), 200.0,
		"4-layer random model should not produce logits > 200")
}

// TestResidualAccumulation checks that residual connections don't explode through layers.
func TestResidualAccumulation(t *testing.T) {
	backend := adBackend()

	cfg := Config{
		VocabSize:   100,
		HiddenSize:  64,
		NumLayers:   8,
		NumHeads:    4,
		NumKVHeads:  2,
		HeadDim:     16,
		FFNSize:     128,
		MaxSeqLen:   32,
		RopeTheta:   10000,
		NormEpsilon: 1e-5,
	}

	nn.SetSeed(42)
	model := NewModel(cfg, backend)

	// Run forward and check that each layer's output doesn't grow unbounded.
	input, err := tensor.FromSlice[int32]([]int32{5}, tensor.Shape{1, 1}, backend)
	require.NoError(t, err)

	embedded := model.Embed.Forward(input)

	hidden := embedded
	for i, layer := range model.Layers {
		hidden = layer.Forward(hidden, nil, 0)
		rms := math.Sqrt(sqMean(hidden.Data()))
		t.Logf("After layer %d: RMS=%.4f", i, rms)

		assert.Less(t, rms, 100.0,
			fmt.Sprintf("hidden state RMS after layer %d should not explode", i))
	}
}

func sqMean(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return sum / float64(len(data))
}
