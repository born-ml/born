// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package llama

import (
	"testing"

	"github.com/born-ml/born/internal/gguf"
)

// TestDefaultConfig verifies that DefaultConfig returns sensible TinyLlama-1.1B defaults.
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	checks := []struct {
		name string
		got  int
		want int
	}{
		{"VocabSize", cfg.VocabSize, 32000},
		{"HiddenSize", cfg.HiddenSize, 2048},
		{"NumLayers", cfg.NumLayers, 22},
		{"NumHeads", cfg.NumHeads, 32},
		{"NumKVHeads", cfg.NumKVHeads, 4},
		{"HeadDim", cfg.HeadDim, 64},
		{"FFNSize", cfg.FFNSize, 5632},
		{"MaxSeqLen", cfg.MaxSeqLen, 2048},
	}

	for _, tt := range checks {
		if tt.got != tt.want {
			t.Errorf("DefaultConfig().%s = %d, want %d", tt.name, tt.got, tt.want)
		}
	}

	if cfg.RopeTheta != 10000.0 {
		t.Errorf("DefaultConfig().RopeTheta = %f, want 10000.0", cfg.RopeTheta)
	}
	if cfg.NormEpsilon != 1e-5 {
		t.Errorf("DefaultConfig().NormEpsilon = %v, want 1e-5", cfg.NormEpsilon)
	}
}

// TestConfigFromGGUF_MockMetadata tests ConfigFromGGUF with a manually-constructed
// gguf.File whose metadata matches a known configuration.
func TestConfigFromGGUF_MockMetadata(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]interface{}
		tokens   []string
		wantCfg  Config
	}{
		{
			name: "llama2_7b_style",
			metadata: map[string]interface{}{
				"general.architecture":                   "llama",
				"llama.embedding_length":                 uint32(4096),
				"llama.block_count":                      uint32(32),
				"llama.attention.head_count":             uint32(32),
				"llama.attention.head_count_kv":          uint32(32),
				"llama.feed_forward_length":              uint32(11008),
				"llama.context_length":                   uint32(4096),
				"llama.rope.freq_base":                   float32(10000.0),
				"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
			},
			tokens: makeTokens(32000),
			wantCfg: Config{
				VocabSize:   32000,
				HiddenSize:  4096,
				NumLayers:   32,
				NumHeads:    32,
				NumKVHeads:  32,
				HeadDim:     128,
				FFNSize:     11008,
				MaxSeqLen:   4096,
				RopeTheta:   10000.0,
				NormEpsilon: 1e-5,
			},
		},
		{
			name: "llama3_8b_gqa",
			metadata: map[string]interface{}{
				"general.architecture":                   "llama",
				"llama.embedding_length":                 uint32(4096),
				"llama.block_count":                      uint32(32),
				"llama.attention.head_count":             uint32(32),
				"llama.attention.head_count_kv":          uint32(8),
				"llama.feed_forward_length":              uint32(14336),
				"llama.context_length":                   uint32(8192),
				"llama.rope.freq_base":                   float32(500000.0),
				"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
			},
			tokens: makeTokens(128256),
			wantCfg: Config{
				VocabSize:   128256,
				HiddenSize:  4096,
				NumLayers:   32,
				NumHeads:    32,
				NumKVHeads:  8,
				HeadDim:     128,
				FFNSize:     14336,
				MaxSeqLen:   8192,
				RopeTheta:   500000.0,
				NormEpsilon: 1e-5,
			},
		},
		{
			name: "missing_rope_defaults_to_10000",
			metadata: map[string]interface{}{
				"general.architecture":          "llama",
				"llama.embedding_length":        uint32(2048),
				"llama.block_count":             uint32(22),
				"llama.attention.head_count":    uint32(32),
				"llama.attention.head_count_kv": uint32(4),
				"llama.feed_forward_length":     uint32(5632),
				"llama.context_length":          uint32(2048),
			},
			tokens: makeTokens(32000),
			wantCfg: Config{
				VocabSize:   32000,
				HiddenSize:  2048,
				NumLayers:   22,
				NumHeads:    32,
				NumKVHeads:  4,
				HeadDim:     64,
				FFNSize:     5632,
				MaxSeqLen:   2048,
				RopeTheta:   10000.0, // default
				NormEpsilon: 1e-5,    // default
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file := &gguf.File{
				Metadata: tt.metadata,
			}
			// Inject tokens to simulate VocabSize.
			if len(tt.tokens) > 0 {
				file.Metadata["tokenizer.ggml.tokens"] = tt.tokens
			}

			got := ConfigFromGGUF(file)
			assertConfigEqual(t, got, tt.wantCfg)
		})
	}
}

// assertConfigEqual compares two Config values and reports differences.
func assertConfigEqual(t *testing.T, got, want Config) {
	t.Helper()

	intFields := []struct {
		name      string
		got, want int
	}{
		{"VocabSize", got.VocabSize, want.VocabSize},
		{"HiddenSize", got.HiddenSize, want.HiddenSize},
		{"NumLayers", got.NumLayers, want.NumLayers},
		{"NumHeads", got.NumHeads, want.NumHeads},
		{"NumKVHeads", got.NumKVHeads, want.NumKVHeads},
		{"HeadDim", got.HeadDim, want.HeadDim},
		{"FFNSize", got.FFNSize, want.FFNSize},
		{"MaxSeqLen", got.MaxSeqLen, want.MaxSeqLen},
	}

	for _, f := range intFields {
		if f.got != f.want {
			t.Errorf("%s = %d, want %d", f.name, f.got, f.want)
		}
	}

	if got.RopeTheta != want.RopeTheta {
		t.Errorf("RopeTheta = %f, want %f", got.RopeTheta, want.RopeTheta)
	}
	if got.NormEpsilon != want.NormEpsilon {
		t.Errorf("NormEpsilon = %v, want %v", got.NormEpsilon, want.NormEpsilon)
	}
}

// makeTokens creates a dummy token slice of the given size to simulate VocabSize.
func makeTokens(n int) []string {
	tokens := make([]string, n)
	for i := range tokens {
		tokens[i] = "<tok>"
	}
	return tokens
}
