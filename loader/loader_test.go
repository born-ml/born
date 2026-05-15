// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package loader

import (
	"testing"
)

// TestOpenModel_UnsupportedExtension verifies that OpenModel returns an error
// for unrecognized file extensions.
func TestOpenModel_UnsupportedExtension(t *testing.T) {
	tests := []struct {
		name string
		path string
	}{
		{"pt_extension", "model.pt"},
		{"bin_extension", "model.bin"},
		{"no_extension", "model"},
		{"empty", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := OpenModel(tt.path)
			if err == nil {
				t.Errorf("OpenModel(%q): expected error for unsupported extension, got nil", tt.path)
			}
		})
	}
}

// TestOpenModel_NonExistentGGUF verifies that opening a nonexistent .gguf file
// returns an error rather than panicking.
func TestOpenModel_NonExistentGGUF(t *testing.T) {
	_, err := OpenModel("/nonexistent/path/model.gguf")
	if err == nil {
		t.Error("OpenModel: expected error for nonexistent .gguf file, got nil")
	}
}

// TestOpenModel_NonExistentSafeTensors verifies that opening a nonexistent
// .safetensors file returns an error rather than panicking.
func TestOpenModel_NonExistentSafeTensors(t *testing.T) {
	_, err := OpenModel("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Error("OpenModel: expected error for nonexistent .safetensors file, got nil")
	}
}

// TestFormatString verifies that ModelFormat.String() returns human-readable names.
func TestFormatString(t *testing.T) {
	tests := []struct {
		format ModelFormat
		want   string
	}{
		{FormatUnknown, "Unknown"},
		{FormatSafeTensors, "SafeTensors"},
		{FormatGGUF, "GGUF"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := tt.format.String()
			if got != tt.want {
				t.Errorf("ModelFormat(%d).String() = %q, want %q", tt.format, got, tt.want)
			}
		})
	}
}

// TestDetectArchitecture verifies architecture detection from weight name lists.
func TestDetectArchitecture(t *testing.T) {
	tests := []struct {
		name     string
		names    []string
		wantArch string
	}{
		{
			name:     "empty_defaults_to_llama",
			names:    []string{},
			wantArch: "llama",
		},
		{
			name:     "llama_weights",
			names:    []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"},
			wantArch: "llama",
		},
		{
			name:     "deepseek_kv_a_proj",
			names:    []string{"model.layers.0.self_attn.kv_a_proj.weight"},
			wantArch: "deepseek",
		},
		{
			name:     "mixtral_moe",
			names:    []string{"model.layers.0.block_sparse_moe.experts.0.w1.weight"},
			wantArch: "mistral",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectArchitecture(tt.names)
			if got != tt.wantArch {
				t.Errorf("DetectArchitecture() = %q, want %q", got, tt.wantArch)
			}
		})
	}
}

// TestLLaMAMapper verifies that the LLaMA mapper correctly translates weight names.
func TestLLaMAMapper(t *testing.T) {
	mapper := NewLLaMAMapper()

	if got := mapper.Architecture(); got != "llama" {
		t.Errorf("Architecture() = %q, want %q", got, "llama")
	}

	tests := []struct {
		input string
		want  string
	}{
		{"model.embed_tokens.weight", "embedding.weight"},
		{"model.norm.weight", "norm.weight"},
		{"lm_head.weight", "lm_head.weight"},
		{"model.layers.0.self_attn.q_proj.weight", "layers.0.attn.q.weight"},
		{"model.layers.3.mlp.gate_proj.weight", "layers.3.ffn.gate.weight"},
		{"model.layers.1.input_layernorm.weight", "layers.1.norm1.weight"},
		{"model.layers.1.post_attention_layernorm.weight", "layers.1.norm2.weight"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := mapper.MapName(tt.input)
			if err != nil {
				t.Fatalf("MapName(%q) error: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("MapName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
