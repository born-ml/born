// Package tokenizer provides text tokenization for LLM inference in Born ML.
//
// This package wraps the internal tokenizer implementations and provides
// a clean public API for tokenization tasks.
//
// Supported tokenizers:
//   - TikToken: OpenAI BPE tokenizers (GPT-3, GPT-4)
//   - BPE: Byte-Pair Encoding from HuggingFace
//   - Chat Templates: Format conversational messages
//
// Example usage:
//
//	import "github.com/born-ml/born/tokenizer"
//
//	// Load tiktoken
//	tok, err := tokenizer.NewTikToken("cl100k_base")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Encode text
//	tokens, err := tok.Encode("Hello, world!")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Decode tokens
//	text, err := tok.Decode(tokens)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Apply chat template
//	messages := []tokenizer.ChatMessage{
//	    {Role: "system", Content: "You are helpful."},
//	    {Role: "user", Content: "Hi!"},
//	}
//	template := tokenizer.NewChatMLTemplate()
//	prompt := template.Apply(messages)
package tokenizer

import (
	"github.com/born-ml/born/internal/tokenizer"
)

// Tokenizer is the core interface for text tokenization.
//
// All tokenizer implementations must implement this interface.
type Tokenizer = tokenizer.Tokenizer

// ChatMessage represents a single message in a conversation.
type ChatMessage = tokenizer.ChatMessage

// ChatTemplate formats messages for conversational models.
type ChatTemplate = tokenizer.ChatTemplate

// NewTikToken creates a new TikToken tokenizer with the specified encoding.
//
// Supported encodings: "cl100k_base" (GPT-4), "p50k_base" (GPT-3).
func NewTikToken(encodingName string) (Tokenizer, error) {
	return tokenizer.NewTikToken(encodingName)
}

// NewTikTokenForModel creates a TikToken tokenizer for a specific model.
//
// Example models: "gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002".
func NewTikTokenForModel(modelName string) (Tokenizer, error) {
	return tokenizer.NewTikTokenForModel(modelName)
}

// LoadFromHuggingFace loads a tokenizer from a HuggingFace model directory.
//
// The directory should contain tokenizer.json.
func LoadFromHuggingFace(modelPath string) (Tokenizer, error) {
	return tokenizer.LoadFromHuggingFace(modelPath)
}

// AutoLoad attempts to automatically load the correct tokenizer.
//
// It tries multiple strategies:
//  1. Load from HuggingFace model directory (tokenizer.json)
//  2. Load tiktoken by model name
//  3. Load tiktoken by encoding name
func AutoLoad(pathOrName string) (Tokenizer, error) {
	return tokenizer.AutoLoadTokenizer(pathOrName)
}

// NewChatMLTemplate creates a ChatML template (OpenAI, DeepSeek format).
//
// Format: <|im_start|>role\ncontent<|im_end|>.
func NewChatMLTemplate() ChatTemplate {
	return tokenizer.NewChatMLTemplate()
}

// NewLLaMATemplate creates a LLaMA chat template.
//
// Format: [INST] user message [/INST] assistant response.
func NewLLaMATemplate() ChatTemplate {
	return tokenizer.NewLLaMATemplate()
}

// NewMistralTemplate creates a Mistral chat template.
func NewMistralTemplate() ChatTemplate {
	return tokenizer.NewMistralTemplate()
}

// GetChatTemplate returns a chat template by name.
//
// Supported names: "chatml", "llama", "mistral".
func GetChatTemplate(name string) (ChatTemplate, error) {
	return tokenizer.GetChatTemplate(name)
}

// ExampleBPE creates a minimal BPE tokenizer for testing and examples.
func ExampleBPE() Tokenizer {
	return tokenizer.ExampleBPEVocab()
}
