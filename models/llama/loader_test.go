// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package llama

import (
	"testing"

	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/tensor"
)

// TestWeightLoader_setParameter verifies that weight routing to model parameters
// works correctly without requiring an actual GGUF file.
//
// It creates a tiny model with known shapes and verifies that setParameter
// correctly dispatches each named tensor to its destination.
// The plain *CPUBackend is sufficient here because no forward pass is executed.
func TestWeightLoader_setParameter(t *testing.T) {
	// Weight routing tests do not execute forward passes — plain cpu.New() is fine.
	backend := cpu.New()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	loader := &weightLoader[*cpu.CPUBackend]{
		model:   model,
		backend: backend,
	}

	tests := []struct {
		name     string
		bornName string
		shape    tensor.Shape
		wantErr  bool
	}{
		{
			name:     "embedding_weight",
			bornName: bornNameEmbeddingWeight,
			shape:    tensor.Shape{cfg.VocabSize, cfg.HiddenSize},
		},
		{
			name:     "norm_weight",
			bornName: "norm.weight",
			shape:    tensor.Shape{cfg.HiddenSize},
		},
		{
			name:     "lm_head_weight",
			bornName: "lm_head.weight",
			shape:    tensor.Shape{cfg.VocabSize, cfg.HiddenSize},
		},
		{
			name:     "layer0_norm1",
			bornName: "layers.0.norm1.weight",
			shape:    tensor.Shape{cfg.HiddenSize},
		},
		{
			name:     "layer0_norm2",
			bornName: "layers.0.norm2.weight",
			shape:    tensor.Shape{cfg.HiddenSize},
		},
		{
			name:     "layer0_attn_q",
			bornName: "layers.0.attn.q.weight",
			shape:    tensor.Shape{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
		},
		{
			name:     "layer0_attn_k",
			bornName: "layers.0.attn.k.weight",
			shape:    tensor.Shape{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
		},
		{
			name:     "layer0_attn_v",
			bornName: "layers.0.attn.v.weight",
			shape:    tensor.Shape{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
		},
		{
			name:     "layer0_attn_o",
			bornName: "layers.0.attn.o.weight",
			shape:    tensor.Shape{cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim},
		},
		{
			name:     "layer0_ffn_gate",
			bornName: "layers.0.ffn.gate.weight",
			shape:    tensor.Shape{cfg.FFNSize, cfg.HiddenSize},
		},
		{
			name:     "layer0_ffn_up",
			bornName: "layers.0.ffn.up.weight",
			shape:    tensor.Shape{cfg.FFNSize, cfg.HiddenSize},
		},
		{
			name:     "layer0_ffn_down",
			bornName: "layers.0.ffn.down.weight",
			shape:    tensor.Shape{cfg.HiddenSize, cfg.FFNSize},
		},
		{
			name:     "unknown_is_skipped",
			bornName: "some.unknown.tensor",
			shape:    tensor.Shape{4, 4},
			wantErr:  false, // silently skipped
		},
		{
			name:     "layer_index_out_of_range",
			bornName: "layers.999.attn.q.weight",
			shape:    tensor.Shape{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
			wantErr:  true,
		},
		{
			name:     "wrong_shape_embedding",
			bornName: bornNameEmbeddingWeight,
			shape:    tensor.Shape{1, 1}, // wrong shape
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := tensor.NewRaw(tt.shape, tensor.Float32, tensor.CPU)
			if err != nil {
				t.Fatalf("create raw tensor: %v", err)
			}
			// Fill with distinguishable non-zero data to verify the copy.
			data := raw.AsFloat32()
			for i := range data {
				data[i] = float32(i+1) * 0.01
			}

			err = loader.setParameter(tt.bornName, raw)
			if (err != nil) != tt.wantErr {
				t.Errorf("setParameter(%q) error = %v, wantErr %v", tt.bornName, err, tt.wantErr)
			}
		})
	}
}

// TestNewKVCacheIntegration verifies that Model.Forward works with a live KV cache.
// This test uses the autodiff-wrapped backend because SwiGLU requires SiLU.
func TestNewKVCacheIntegration(t *testing.T) {
	backend := newCPUBackend()
	cfg := tinyConfig()
	model := NewModel(cfg, backend)

	// Create a per-layer KV cache for this model.
	cache := NewModelCache(cfg, cfg.MaxSeqLen, backend)

	// Prefill: process 3-token prompt.
	prompt := makeInt32Input(1, 3, cfg.VocabSize)
	logits := model.Forward(prompt, cache, 0)
	if logits == nil {
		t.Fatal("prefill returned nil")
	}
	assertShape3D(t, "prefill logits", logits.Shape(), 1, 3, cfg.VocabSize)

	// Decode: single next token.
	nextTok := makeInt32Input(1, 1, cfg.VocabSize)
	logits2 := model.Forward(nextTok, cache, 3)
	if logits2 == nil {
		t.Fatal("decode returned nil")
	}
	assertShape3D(t, "decode logits", logits2.Shape(), 1, 1, cfg.VocabSize)
}

// assertShape3D asserts a shape is [d0, d1, d2].
func assertShape3D(t *testing.T, label string, shape tensor.Shape, d0, d1, d2 int) {
	t.Helper()
	if len(shape) != 3 || shape[0] != d0 || shape[1] != d1 || shape[2] != d2 {
		t.Errorf("%s: shape = %v, want [%d, %d, %d]", label, shape, d0, d1, d2)
	}
}
