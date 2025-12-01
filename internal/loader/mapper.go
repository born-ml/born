package loader

import (
	"fmt"
	"strings"
)

// Architecture names.
const (
	ArchitectureLLaMA    = "llama"
	ArchitectureMistral  = "mistral"
	ArchitectureDeepSeek = "deepseek"
)

// WeightMapper maps model-specific weight names to standard Born names.
type WeightMapper interface {
	// MapName converts a model-specific weight name to Born standard name.
	MapName(name string) (string, error)

	// Architecture returns the architecture name (e.g., "llama", "mistral").
	Architecture() string
}

// LLaMAMapper maps LLaMA weight names to Born standard names.
// Supports LLaMA, LLaMA 2, and LLaMA 3.
type LLaMAMapper struct{}

// NewLLaMAMapper creates a new LLaMA weight mapper.
func NewLLaMAMapper() *LLaMAMapper {
	return &LLaMAMapper{}
}

// MapName converts LLaMA weight names to Born standard names.
// LLaMA format:
//   - model.embed_tokens.weight -> embedding.weight
//   - model.layers.{i}.self_attn.q_proj.weight -> layers.{i}.attn.q.weight
//   - model.layers.{i}.mlp.gate_proj.weight -> layers.{i}.ffn.gate.weight
//   - model.norm.weight -> norm.weight
func (m *LLaMAMapper) MapName(name string) (string, error) {
	// Embedding
	if strings.HasPrefix(name, "model.embed_tokens.weight") {
		return "embedding.weight", nil
	}

	// Final norm
	if strings.HasPrefix(name, "model.norm.weight") {
		return "norm.weight", nil
	}

	// Output/LM head
	if strings.HasPrefix(name, "lm_head.weight") {
		return "lm_head.weight", nil
	}

	// Layer-specific weights
	if strings.HasPrefix(name, "model.layers.") {
		return m.mapLayerWeight(name)
	}

	return name, nil // Return original if no mapping found
}

// mapLayerWeight maps LLaMA layer-specific weights.
func (m *LLaMAMapper) mapLayerWeight(name string) (string, error) {
	parts := strings.Split(name, ".")

	if len(parts) < 4 {
		return name, nil
	}

	// Extract layer index
	layerIdx := parts[2]

	// Attention weights
	if parts[3] == "self_attn" {
		switch {
		case strings.Contains(name, "q_proj.weight"):
			return fmt.Sprintf("layers.%s.attn.q.weight", layerIdx), nil
		case strings.Contains(name, "k_proj.weight"):
			return fmt.Sprintf("layers.%s.attn.k.weight", layerIdx), nil
		case strings.Contains(name, "v_proj.weight"):
			return fmt.Sprintf("layers.%s.attn.v.weight", layerIdx), nil
		case strings.Contains(name, "o_proj.weight"):
			return fmt.Sprintf("layers.%s.attn.o.weight", layerIdx), nil
		}
	}

	// MLP/FFN weights
	if parts[3] == "mlp" {
		switch {
		case strings.Contains(name, "gate_proj.weight"):
			return fmt.Sprintf("layers.%s.ffn.gate.weight", layerIdx), nil
		case strings.Contains(name, "up_proj.weight"):
			return fmt.Sprintf("layers.%s.ffn.up.weight", layerIdx), nil
		case strings.Contains(name, "down_proj.weight"):
			return fmt.Sprintf("layers.%s.ffn.down.weight", layerIdx), nil
		}
	}

	// Normalization weights
	switch {
	case strings.Contains(name, "input_layernorm.weight"):
		return fmt.Sprintf("layers.%s.norm1.weight", layerIdx), nil
	case strings.Contains(name, "post_attention_layernorm.weight"):
		return fmt.Sprintf("layers.%s.norm2.weight", layerIdx), nil
	}

	return name, nil
}

// Architecture returns "llama".
func (m *LLaMAMapper) Architecture() string {
	return ArchitectureLLaMA
}

// MistralMapper maps Mistral weight names to Born standard names.
// Supports Mistral 7B and Mixtral 8x7B.
type MistralMapper struct{}

// NewMistralMapper creates a new Mistral weight mapper.
func NewMistralMapper() *MistralMapper {
	return &MistralMapper{}
}

// MapName converts Mistral weight names to Born standard names.
// Mistral uses similar naming to LLaMA but with some differences.
func (m *MistralMapper) MapName(name string) (string, error) {
	// Mistral uses the same structure as LLaMA for most weights
	llamaMapper := NewLLaMAMapper()
	mapped, err := llamaMapper.MapName(name)
	if err != nil {
		return "", err
	}

	// Handle MoE-specific weights for Mixtral
	if strings.Contains(name, "block_sparse_moe") {
		return m.mapMoEWeight(name)
	}

	return mapped, nil
}

// mapMoEWeight maps Mixtral MoE (Mixture of Experts) weights.
func (m *MistralMapper) mapMoEWeight(name string) (string, error) {
	parts := strings.Split(name, ".")

	if len(parts) < 5 {
		return name, nil
	}

	layerIdx := parts[2]

	// MoE experts: model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight
	if parts[4] == "experts" && len(parts) >= 7 {
		expertIdx := parts[5]
		switch parts[6] {
		case "w1.weight":
			return fmt.Sprintf("layers.%s.moe.experts.%s.w1.weight", layerIdx, expertIdx), nil
		case "w2.weight":
			return fmt.Sprintf("layers.%s.moe.experts.%s.w2.weight", layerIdx, expertIdx), nil
		case "w3.weight":
			return fmt.Sprintf("layers.%s.moe.experts.%s.w3.weight", layerIdx, expertIdx), nil
		}
	}

	// MoE gate: model.layers.{i}.block_sparse_moe.gate.weight
	if strings.Contains(name, "gate.weight") {
		return fmt.Sprintf("layers.%s.moe.gate.weight", layerIdx), nil
	}

	return name, nil
}

// Architecture returns "mistral".
func (m *MistralMapper) Architecture() string {
	return ArchitectureMistral
}

// DeepSeekMapper maps DeepSeek weight names to Born standard names.
// Supports DeepSeek-V2 and DeepSeek-Coder.
type DeepSeekMapper struct{}

// NewDeepSeekMapper creates a new DeepSeek weight mapper.
func NewDeepSeekMapper() *DeepSeekMapper {
	return &DeepSeekMapper{}
}

// MapName converts DeepSeek weight names to Born standard names.
// DeepSeek uses similar naming to LLaMA but may have architecture-specific differences.
func (m *DeepSeekMapper) MapName(name string) (string, error) {
	// DeepSeek-V2 uses Multi-head Latent Attention (MLA)
	if strings.Contains(name, "kv_a_proj") || strings.Contains(name, "kv_b_proj") {
		return m.mapMLAWeight(name)
	}

	// For other weights, use LLaMA mapping as base
	llamaMapper := NewLLaMAMapper()
	return llamaMapper.MapName(name)
}

// mapMLAWeight maps DeepSeek-V2 Multi-head Latent Attention weights.
func (m *DeepSeekMapper) mapMLAWeight(name string) (string, error) {
	parts := strings.Split(name, ".")

	if len(parts) < 5 {
		return name, nil
	}

	layerIdx := parts[2]

	switch {
	case strings.Contains(name, "kv_a_proj.weight"):
		return fmt.Sprintf("layers.%s.attn.kv_a.weight", layerIdx), nil
	case strings.Contains(name, "kv_b_proj.weight"):
		return fmt.Sprintf("layers.%s.attn.kv_b.weight", layerIdx), nil
	}

	return name, nil
}

// Architecture returns "deepseek".
func (m *DeepSeekMapper) Architecture() string {
	return ArchitectureDeepSeek
}

// DetectArchitecture attempts to detect model architecture from weight names.
func DetectArchitecture(names []string) string {
	// Check for DeepSeek-specific weights
	for _, name := range names {
		if strings.Contains(name, "kv_a_proj") || strings.Contains(name, "kv_b_proj") {
			return ArchitectureDeepSeek
		}
	}

	// Check for Mixtral MoE weights
	for _, name := range names {
		if strings.Contains(name, "block_sparse_moe") {
			return ArchitectureMistral
		}
	}

	// Default to LLaMA (most common)
	return ArchitectureLLaMA
}

// GetMapper returns the appropriate weight mapper for an architecture.
func GetMapper(architecture string) WeightMapper {
	switch architecture {
	case ArchitectureLLaMA:
		return NewLLaMAMapper()
	case ArchitectureMistral:
		return NewMistralMapper()
	case ArchitectureDeepSeek:
		return NewDeepSeekMapper()
	default:
		return NewLLaMAMapper() // Default to LLaMA
	}
}
