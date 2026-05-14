// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

// Package llama provides a LLaMA model implementation for the Born ML framework.
//
// It supports LLaMA 2 and LLaMA 3 architectures loaded from GGUF files.
// The model implements generate.LLMModel and can be used directly with
// generate.TextGenerator for text generation.
//
// Example — loading and running inference:
//
//	backend := cpu.New()
//	model, err := llama.LoadGGUF("llama-3-8b-Q4_0.gguf", backend)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	tok, _ := tokenizer.LoadHF("tokenizer.json")
//	gen := generate.NewTextGenerator(model, tok, generate.DefaultSamplingConfig())
//	text, _ := gen.Generate("Hello, world", generate.DefaultGenerateConfig())
//	fmt.Println(text)
package llama

import (
	"github.com/born-ml/born/internal/gguf"
)

// Config holds the hyperparameters for a LLaMA model.
//
// These values are read from GGUF metadata and determine the model architecture.
// All fields have sensible defaults matching TinyLlama-1.1B.
type Config struct {
	// VocabSize is the vocabulary size (number of token embeddings).
	VocabSize int

	// HiddenSize is the embedding/hidden dimension (d_model).
	// Also called embedding_length in GGUF metadata.
	HiddenSize int

	// NumLayers is the number of transformer blocks.
	// Also called block_count in GGUF metadata.
	NumLayers int

	// NumHeads is the number of query attention heads.
	// Also called attention.head_count in GGUF metadata.
	NumHeads int

	// NumKVHeads is the number of key/value heads for GQA.
	// Also called attention.head_count_kv in GGUF metadata.
	// When NumKVHeads < NumHeads, the model uses Grouped Query Attention (GQA).
	NumKVHeads int

	// HeadDim is the dimension per attention head (HiddenSize / NumHeads).
	HeadDim int

	// FFNSize is the feed-forward network intermediate dimension.
	// Also called feed_forward_length in GGUF metadata.
	FFNSize int

	// MaxSeqLen is the maximum context length.
	// Also called context_length in GGUF metadata.
	MaxSeqLen int

	// RopeTheta is the base frequency for Rotary Position Embeddings.
	// Defaults to 10000.0. LLaMA 3 uses 500000.0.
	RopeTheta float64

	// NormEpsilon is the epsilon for RMSNorm numerical stability.
	// Defaults to 1e-5.
	NormEpsilon float32
}

// DefaultConfig returns a Config matching TinyLlama-1.1B defaults.
//
// Useful as a starting point for custom configurations or unit tests.
func DefaultConfig() Config {
	return Config{
		VocabSize:   32000,
		HiddenSize:  2048,
		NumLayers:   22,
		NumHeads:    32,
		NumKVHeads:  4,
		HeadDim:     64,
		FFNSize:     5632,
		MaxSeqLen:   2048,
		RopeTheta:   10000.0,
		NormEpsilon: 1e-5,
	}
}

// ConfigFromGGUF extracts model configuration from a parsed GGUF file.
//
// It reads architecture-specific metadata keys following the GGUF specification.
// All fields absent from the metadata fall back to safe defaults.
func ConfigFromGGUF(file *gguf.File) Config {
	hiddenSize := file.EmbeddingLength()
	numHeads := file.HeadCount()

	headDim := 0
	if numHeads > 0 && hiddenSize > 0 {
		headDim = hiddenSize / numHeads
	}

	rpc := ropeFreqBase(file)

	return Config{
		VocabSize:   file.VocabSize(),
		HiddenSize:  hiddenSize,
		NumLayers:   file.BlockCount(),
		NumHeads:    numHeads,
		NumKVHeads:  file.HeadCountKV(),
		HeadDim:     headDim,
		FFNSize:     file.FeedForwardLength(),
		MaxSeqLen:   contextLength(file),
		RopeTheta:   rpc,
		NormEpsilon: normEpsilon(file),
	}
}

// contextLength reads context_length from GGUF metadata with a safe default.
func contextLength(file *gguf.File) int {
	if v := file.ContextLength(); v > 0 {
		return v
	}
	return 2048
}

// ropeFreqBase reads rope_freq_base from GGUF metadata.
// LLaMA 2 default is 10000; LLaMA 3 is 500000.
func ropeFreqBase(file *gguf.File) float64 {
	key := file.Architecture() + ".rope.freq_base"
	if v, ok := file.Metadata[key].(float32); ok {
		return float64(v)
	}
	return 10000.0
}

// normEpsilon reads layer_norm_epsilon from GGUF metadata with a safe default.
func normEpsilon(file *gguf.File) float32 {
	key := file.Architecture() + ".attention.layer_norm_rms_epsilon"
	if v, ok := file.Metadata[key].(float32); ok {
		return v
	}
	return 1e-5
}
