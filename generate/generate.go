// Package generate provides text generation utilities for LLMs in Born ML.
//
// This package wraps the internal generate implementations and provides
// a clean public API for text generation tasks.
//
// Components:
//   - Sampler: Sampling strategies (greedy, top-k, top-p, temperature, etc.)
//   - TextGenerator: High-level text generation interface
//
// Example usage:
//
//	import (
//	    "github.com/born-ml/born/generate"
//	    "github.com/born-ml/born/tokenizer"
//	)
//
//	// Create sampler
//	config := generate.SamplingConfig{
//	    Temperature: 0.7,
//	    TopP:        0.9,
//	    TopK:        40,
//	    Seed:        42,
//	}
//	sampler := generate.NewSampler(config)
//
//	// Sample from logits
//	token := sampler.Sample(logits, previousTokens)
package generate

import (
	"github.com/born-ml/born/internal/generate"
	"github.com/born-ml/born/internal/tokenizer"
)

// Sampling Configuration

// SamplingConfig configures the sampling strategy for text generation.
//
// Parameters:
//   - Temperature: Controls randomness (0 = greedy, 1 = normal, >1 = more random)
//   - TopK: Limits sampling to top K tokens (0 = disabled)
//   - TopP: Nucleus sampling, limits to tokens with cumulative prob < P (1.0 = disabled)
//   - MinP: Filters tokens with prob < max_prob * MinP (0 = disabled)
//   - RepeatPenalty: Penalty for repeated tokens (1.0 = no penalty)
//   - FrequencyPenalty: Penalty based on token frequency (0 = disabled)
//   - PresencePenalty: Penalty for token presence (0 = disabled)
//   - RepeatWindow: Number of tokens to consider for penalties (0 = all)
//   - Seed: Random seed for reproducibility (-1 = random)
type SamplingConfig = generate.SamplingConfig

// DefaultSamplingConfig returns sensible defaults for text generation.
//
// Defaults:
//   - Temperature: 1.0
//   - TopK: 0 (disabled)
//   - TopP: 1.0 (disabled)
//   - MinP: 0.0 (disabled)
//   - RepeatPenalty: 1.0 (no penalty)
//   - Seed: -1 (random)
func DefaultSamplingConfig() SamplingConfig {
	return generate.DefaultSamplingConfig()
}

// Sampler

// Sampler samples tokens from logits using configurable strategies.
type Sampler = generate.Sampler

// NewSampler creates a new sampler with the given configuration.
//
// Example:
//
//	config := generate.SamplingConfig{
//	    Temperature: 0.7,
//	    TopK:        50,
//	    Seed:        42,
//	}
//	sampler := generate.NewSampler(config)
//	token := sampler.Sample(logits, nil)
func NewSampler(config SamplingConfig) *Sampler {
	return generate.NewSampler(config)
}

// Generation Configuration

// GenerateConfig configures text generation.
//
// Parameters:
//   - MaxTokens: Maximum number of tokens to generate
//   - MinTokens: Minimum number of tokens before stopping
//   - StopStrings: Strings that trigger stopping
//   - StopTokens: Token IDs that trigger stopping
//   - Stream: Enable streaming generation
//   - EchoPrompt: Include prompt in output
//   - Sampling: Sampling configuration
//
//nolint:revive // GenerateConfig is clearer than Config
type GenerateConfig = generate.GenerateConfig

// DefaultGenerateConfig returns sensible defaults for generation.
//
// Defaults:
//   - MaxTokens: 256
//   - MinTokens: 0
//   - Stream: false
//   - EchoPrompt: false
func DefaultGenerateConfig() GenerateConfig {
	return generate.DefaultGenerateConfig()
}

// GenerateResult is a single result from streaming generation.
//
//nolint:revive // GenerateResult is clearer than Result
type GenerateResult = generate.GenerateResult

// TextGenerator

// KVCache is an interface for key-value caches used in generation.
type KVCache = generate.KVCache

// LLMModel is the interface for language models used in generation.
type LLMModel = generate.LLMModel

// TextGenerator generates text using an LLM.
type TextGenerator = generate.TextGenerator

// GeneratorOption configures a TextGenerator.
type GeneratorOption = generate.GeneratorOption

// WithMaxSeqLen sets the maximum sequence length.
//
// Example:
//
//	gen := generate.NewTextGenerator(model, tok, config, generate.WithMaxSeqLen(4096))
func WithMaxSeqLen(n int) GeneratorOption {
	return generate.WithMaxSeqLen(n)
}

// NewTextGenerator creates a new text generator.
//
// Example:
//
//	tok, _ := tokenizer.NewTikToken("cl100k_base")
//	config := generate.DefaultSamplingConfig()
//	config.Temperature = 0.7
//
//	gen := generate.NewTextGenerator(model, tok, config)
//	result, err := gen.Generate("Hello, world!", generate.DefaultGenerateConfig())
func NewTextGenerator(
	model LLMModel,
	tok tokenizer.Tokenizer,
	samplingConfig SamplingConfig,
	opts ...GeneratorOption,
) *TextGenerator {
	return generate.NewTextGenerator(model, tok, samplingConfig, opts...)
}

// ChatMessage is re-exported from tokenizer for convenience.
type ChatMessage = tokenizer.ChatMessage

// ChatTemplate is re-exported from tokenizer for convenience.
type ChatTemplate = tokenizer.ChatTemplate
