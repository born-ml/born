// Package loader provides model weight loading functionality for Born ML framework.
//
// This package wraps internal loader implementations and exports a clean public API
// for loading model weights from various formats (SafeTensors, GGUF).
//
// Example usage:
//
//	import (
//	    "github.com/born-ml/born/loader"
//	    "github.com/born-ml/born/backend/cpu"
//	)
//
//	// Open model with auto-detection
//	model, err := loader.OpenModel("path/to/model.safetensors")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	// Get model information
//	fmt.Printf("Format: %s\n", model.Format())
//	fmt.Printf("Architecture: %s\n", model.Architecture())
//
//	// Load a specific tensor
//	backend := cpu.New()
//	tensor, err := model.LoadTensor("model.layers.0.attn.q_proj.weight", backend)
//	if err != nil {
//	    log.Fatal(err)
//	}
package loader

import (
	"github.com/born-ml/born/internal/loader"
)

// ModelFormat represents the model weight format.
type ModelFormat = loader.ModelFormat

// Supported model formats.
const (
	FormatUnknown     ModelFormat = loader.FormatUnknown
	FormatSafeTensors ModelFormat = loader.FormatSafeTensors
	FormatGGUF        ModelFormat = loader.FormatGGUF
)

// ModelReader provides a unified interface for loading model weights.
// It abstracts away the underlying file format and provides consistent access
// to model tensors.
//
// Note: This is a type alias because the LoadTensor method signature references
// internal tensor types that cannot be abstracted without a wrapper layer.
type ModelReader = loader.ModelReader

// OpenModel opens a model file and auto-detects the format.
//
// Supported formats:
//   - .safetensors (Hugging Face standard)
//   - .gguf (llama.cpp ecosystem)
//
// The function automatically detects the model architecture (LLaMA, Mistral, DeepSeek)
// based on weight names and metadata.
//
// Example:
//
//	model, err := loader.OpenModel("path/to/llama-7b.safetensors")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	fmt.Printf("Format: %s\n", model.Format())        // "SafeTensors"
//	fmt.Printf("Architecture: %s\n", model.Architecture()) // "llama"
//
//	// List all tensors
//	for _, name := range model.TensorNames() {
//	    fmt.Println(name)
//	}
//
//	// Load specific tensor
//	backend := cpu.New()
//	weight, err := model.LoadTensor("model.layers.0.attn.q_proj.weight", backend)
//	if err != nil {
//	    log.Fatal(err)
//	}
func OpenModel(path string) (ModelReader, error) {
	return loader.OpenModel(path)
}

// WeightMapper maps model-specific weight names to standard Born names.
// Different model architectures use different naming conventions.
// This interface provides a way to normalize weight names.
type WeightMapper interface {
	// MapName converts a model-specific weight name to Born standard name.
	MapName(name string) (string, error)

	// Architecture returns the architecture name (e.g., "llama", "mistral").
	Architecture() string
}

// NewLLaMAMapper creates a weight mapper for LLaMA models.
// Supports LLaMA, LLaMA 2, and LLaMA 3.
func NewLLaMAMapper() WeightMapper {
	return loader.NewLLaMAMapper()
}

// NewMistralMapper creates a weight mapper for Mistral models.
// Supports Mistral 7B and Mixtral 8x7B MoE.
func NewMistralMapper() WeightMapper {
	return loader.NewMistralMapper()
}

// NewDeepSeekMapper creates a weight mapper for DeepSeek models.
// Supports DeepSeek-V2 and DeepSeek-Coder.
func NewDeepSeekMapper() WeightMapper {
	return loader.NewDeepSeekMapper()
}

// DetectArchitecture attempts to detect model architecture from weight names.
// Returns "llama", "mistral", or "deepseek".
func DetectArchitecture(names []string) string {
	return loader.DetectArchitecture(names)
}
