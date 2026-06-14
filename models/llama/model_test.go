// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package llama

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/generate"
	"github.com/born-ml/born/internal/tensor"
)

// Compile-time interface satisfaction check.
// If Model does not implement LLMModel, this line fails to compile.
var _ generate.LLMModel = (*Model[cpuBackend])(nil)

// cpuBackend is the concrete type used in forward-pass tests.
// SiLU (used by SwiGLU FFN) requires autodiff.AutodiffBackend wrapper.
type cpuBackend = *autodiff.AutodiffBackend[*cpu.CPUBackend]

// newCPUBackend creates the backend used in tests.
func newCPUBackend() cpuBackend {
	return autodiff.New(cpu.New())
}

// tinyConfig returns a tiny LLaMA config suitable for unit tests.
// All dimensions are deliberately small to keep test runtime fast.
func tinyConfig() Config {
	return Config{
		VocabSize:   128,
		HiddenSize:  16,
		NumLayers:   2,
		NumHeads:    4,
		NumKVHeads:  2,
		HeadDim:     4,
		FFNSize:     32,
		MaxSeqLen:   64,
		RopeTheta:   10000.0,
		NormEpsilon: 1e-5,
	}
}

// TestNewModel_ForwardShape checks that Model.Forward returns the correct shape.
func TestNewModel_ForwardShape(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()

	model := NewModel(cfg, backend)

	tests := []struct {
		name      string
		batchSize int
		seqLen    int
		wantShape []int
	}{
		{"single_token", 1, 1, []int{1, 1, cfg.VocabSize}},
		{"short_sequence", 1, 4, []int{1, 4, cfg.VocabSize}},
		{"batch_2_seq_3", 2, 3, []int{2, 3, cfg.VocabSize}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputRaw := makeInt32Input(tt.batchSize, tt.seqLen, cfg.VocabSize)

			logits := model.Forward(inputRaw, nil, 0)
			if logits == nil {
				t.Fatal("Forward returned nil")
			}

			gotShape := logits.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("output ndim = %d, want %d", len(gotShape), len(tt.wantShape))
			}
			for i, d := range tt.wantShape {
				if gotShape[i] != d {
					t.Errorf("output shape[%d] = %d, want %d", i, gotShape[i], d)
				}
			}
		})
	}
}

// TestNewModel_VocabSize verifies VocabSize() satisfies the LLMModel interface.
func TestNewModel_VocabSize(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	if got := model.VocabSize(); got != cfg.VocabSize {
		t.Errorf("VocabSize() = %d, want %d", got, cfg.VocabSize)
	}
}

// TestNewModel_ParameterCount verifies the model has the expected number of parameters.
func TestNewModel_ParameterCount(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	params := model.Parameters()
	if len(params) == 0 {
		t.Error("model has no parameters")
	}

	// Rough lower bound:
	// - embedding: 1 param
	// - each layer: AttnNorm(1) + QProj(1) + KProj(1) + VProj(1) + OProj(1) + FFNNorm(1) + FFN(3) = 9
	// - final norm: 1
	// - lm_head: 1
	minExpected := 1 + cfg.NumLayers*9 + 1 + 1
	if len(params) < minExpected {
		t.Errorf("Parameters() = %d, want at least %d", len(params), minExpected)
	}
}

// TestNewModel_Validate checks that validate() catches bad configs.
func TestNewModel_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"valid", tinyConfig(), false},
		{
			"zero_vocab",
			func() Config { c := tinyConfig(); c.VocabSize = 0; return c }(),
			true,
		},
		{
			"non_divisible_heads",
			func() Config { c := tinyConfig(); c.NumKVHeads = 3; return c }(), // 4 % 3 != 0
			true,
		},
		{
			"zero_layers",
			func() Config { c := tinyConfig(); c.NumLayers = 0; return c }(),
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// validate() is a pure config check — no backend ops involved.
			model := &Model[cpuBackend]{Config: tt.cfg}
			err := model.validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestWithAttentionFunc checks that a custom attention function is actually invoked.
func TestWithAttentionFunc(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()

	called := false
	customAttn := func(
		q, _ /*k*/, _ /*v*/ *tensor.Tensor[float32, cpuBackend],
		_ /*mask*/ *tensor.Tensor[float32, cpuBackend],
		_ /*scale*/ float32,
	) (*tensor.Tensor[float32, cpuBackend], *tensor.Tensor[float32, cpuBackend]) {
		called = true
		// Return zeros with the same shape as regular SDPA would produce.
		qShape := q.Shape()
		numElems := qShape[0] * qShape[1] * qShape[2] * qShape[3]
		out, err := tensor.FromSlice[float32, cpuBackend](
			make([]float32, numElems),
			tensor.Shape{qShape[0], qShape[1], qShape[2], qShape[3]},
			backend,
		)
		if err != nil {
			t.Errorf("custom attn: create output: %v", err)
		}
		return out, nil
	}

	model := NewModel(cfg, backend, WithAttentionFunc[cpuBackend](customAttn))

	inputRaw, _ := tensor.NewRaw(tensor.Shape{1, 2}, tensor.Int32, tensor.CPU)
	model.Forward(inputRaw, nil, 0)

	if !called {
		t.Error("custom AttentionFunc was not called during Forward")
	}
}

// TestRelease verifies that Release frees parameters and is safe to call multiple times.
func TestRelease(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	params := model.Parameters()
	if len(params) == 0 {
		t.Fatal("model has no parameters before Release")
	}

	// First Release must not panic.
	model.Release()

	// Second Release (double-free) must not panic.
	model.Release()
}

// TestForward_WithKVCache verifies incremental decoding via KV cache.
// First call processes a prompt (multiple tokens), second call decodes one token
// using cached keys/values. Output shapes must match expectations.
func TestForward_WithKVCache(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)
	cache := NewModelCache(cfg, cfg.MaxSeqLen, backend)

	promptLen := 4
	prompt := makeInt32Input(1, promptLen, cfg.VocabSize)

	// Prefill: full prompt, startPos=0.
	logits := model.Forward(prompt, cache, 0)
	if logits == nil {
		t.Fatal("prefill Forward returned nil")
	}
	wantShape := tensor.Shape{1, promptLen, cfg.VocabSize}
	if !logits.Shape().Equal(wantShape) {
		t.Fatalf("prefill shape = %v, want %v", logits.Shape(), wantShape)
	}

	// Decode: one token, startPos=promptLen (cached).
	nextToken := makeInt32Input(1, 1, cfg.VocabSize)
	logits2 := model.Forward(nextToken, cache, promptLen)
	if logits2 == nil {
		t.Fatal("decode Forward returned nil")
	}
	wantShape2 := tensor.Shape{1, 1, cfg.VocabSize}
	if !logits2.Shape().Equal(wantShape2) {
		t.Fatalf("decode shape = %v, want %v", logits2.Shape(), wantShape2)
	}
}

// TestForward_Deterministic verifies that two forward passes with the same
// input and no cache produce identical logits.
func TestForward_Deterministic(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	input := makeInt32Input(1, 3, cfg.VocabSize)

	logits1 := model.Forward(input, nil, 0)
	logits2 := model.Forward(input, nil, 0)

	data1 := logits1.AsFloat32()
	data2 := logits2.AsFloat32()

	if len(data1) != len(data2) {
		t.Fatalf("logit lengths differ: %d vs %d", len(data1), len(data2))
	}
	for i := range data1 {
		if data1[i] != data2[i] {
			t.Errorf("logits[%d] = %f vs %f (not deterministic)", i, data1[i], data2[i])
			break
		}
	}
}

// TestForward_LogitsFinite verifies that forward pass produces no NaN/Inf values.
func TestForward_LogitsFinite(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	input := makeInt32Input(1, 4, cfg.VocabSize)
	logits := model.Forward(input, nil, 0)
	data := logits.AsFloat32()

	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] = %f (NaN or Inf)", i, v)
		}
	}
}

// TestModelCache_Clear verifies that cache.Clear() does not panic
// and that forward works correctly after clearing.
func TestModelCache_Clear(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)
	cache := NewModelCache(cfg, cfg.MaxSeqLen, backend)

	// Fill cache with a forward pass.
	input := makeInt32Input(1, 3, cfg.VocabSize)
	model.Forward(input, cache, 0)

	// Clear must not panic.
	cache.Clear()

	// Forward after clear must succeed (fresh cache).
	logits := model.Forward(input, cache, 0)
	if logits == nil {
		t.Fatal("Forward after cache.Clear returned nil")
	}
}

// TestNewModelCache_LayerCount verifies that the cache has one entry per layer.
func TestNewModelCache_LayerCount(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	cache := NewModelCache(cfg, cfg.MaxSeqLen, backend)

	if len(cache.layers) != cfg.NumLayers {
		t.Errorf("cache has %d layers, want %d", len(cache.layers), cfg.NumLayers)
	}
}

// makeInt32Input creates a RawTensor of int32 token IDs of shape [batch, seq].
func makeInt32Input(batch, seq, vocabSize int) *tensor.RawTensor {
	raw, _ := tensor.NewRaw(tensor.Shape{batch, seq}, tensor.Int32, tensor.CPU)
	data := raw.AsInt32()
	for i := range data {
		data[i] = int32(i % vocabSize)
	}
	return raw
}
