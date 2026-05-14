// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package llama

import (
	"fmt"
	"strings"

	"github.com/born-ml/born/internal/gguf"
	internalLoader "github.com/born-ml/born/internal/loader"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// bornNameEmbeddingWeight is the canonical Born name for the token embedding weight.
// Defined as a constant to satisfy goconst (used in setParameter and tests).
const bornNameEmbeddingWeight = "embedding.weight"

// LoadGGUF loads a LLaMA model from a GGUF file.
//
// It reads architecture metadata, constructs a Model with the correct shape,
// and loads all weights — dequantizing quantized tensors (Q4_0, Q4_K, Q8_0, …)
// transparently.
//
// Supported GGUF architectures: llama, mistral (uses same weight layout).
//
// Example:
//
//	backend := cpu.New()
//	model, err := llama.LoadGGUF("llama-3-8b-Q4_0.gguf", backend)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close() // Release file handle (not needed for CPU backend, but good practice)
func LoadGGUF[B tensor.Backend](path string, backend B, opts ...Option[B]) (*Model[B], error) {
	// 1. Parse GGUF file for metadata and tensor index.
	ggufFile, err := gguf.ParseFile(path)
	if err != nil {
		return nil, fmt.Errorf("llama: parse gguf %s: %w", path, err)
	}

	// 2. Extract model configuration from metadata.
	cfg := ConfigFromGGUF(ggufFile)
	if cfg.VocabSize == 0 {
		return nil, fmt.Errorf("llama: VocabSize is 0 — GGUF file may be missing tokenizer metadata")
	}
	if cfg.HiddenSize == 0 {
		return nil, fmt.Errorf("llama: HiddenSize is 0 — GGUF file may be missing architecture metadata")
	}
	if cfg.NumLayers == 0 {
		return nil, fmt.Errorf("llama: NumLayers is 0 — GGUF file may be missing architecture metadata")
	}

	// 3. Construct model with random weights (will be overwritten by weight loading).
	model := NewModel(cfg, backend, opts...)

	if err := model.validate(); err != nil {
		return nil, err
	}

	// 4. Create TensorConverter for dequantization.
	converter, err := gguf.NewTensorConverter(ggufFile)
	if err != nil {
		return nil, fmt.Errorf("llama: create tensor converter: %w", err)
	}
	defer func() { _ = converter.Close() }()

	// 5. Build the weight mapping: GGUF name → Born name.
	// Auto-detect naming convention (GGML "blk.0..." vs HuggingFace "model.layers.0...").
	var tensorNames []string
	for i := range ggufFile.TensorInfo {
		tensorNames = append(tensorNames, ggufFile.TensorInfo[i].Name)
	}
	mapper := internalLoader.GetMapperForNaming(tensorNames)

	// 6. Load all tensors.
	loader := &weightLoader[B]{
		converter: converter,
		mapper:    mapper,
		model:     model,
		backend:   backend,
	}

	if err := loader.loadAll(ggufFile); err != nil {
		return nil, fmt.Errorf("llama: load weights from %s: %w", path, err)
	}

	// Handle tied embeddings: most LLaMA models omit lm_head.weight from GGUF
	// because it shares weights with embed_tokens (tie_word_embeddings=true).
	if !loader.lmHeadLoaded {
		embedData := model.Embed.Weight.Tensor().Raw().AsFloat32()
		headData := model.Head.Weight().Tensor().Raw().AsFloat32()
		copy(headData, embedData)
	}

	return model, nil
}

// weightLoader centralizes the weight loading logic.
type weightLoader[B tensor.Backend] struct {
	converter    *gguf.TensorConverter
	mapper       internalLoader.WeightMapper
	model        *Model[B]
	backend      B
	lmHeadLoaded bool // tracks whether lm_head.weight was found in GGUF
}

// loadAll iterates every tensor in the GGUF file, maps it to a Born parameter,
// and copies the float32 data into the appropriate model parameter.
func (wl *weightLoader[B]) loadAll(file *gguf.File) error {
	for i := range file.TensorInfo {
		info := &file.TensorInfo[i]
		ggufName := info.Name

		bornName, err := wl.mapper.MapName(ggufName)
		if err != nil {
			// Mapping errors are non-fatal: log and skip unknown tensors.
			continue
		}

		if err := wl.loadTensor(ggufName, bornName); err != nil {
			return fmt.Errorf("tensor %s → %s: %w", ggufName, bornName, err)
		}
	}

	return nil
}

// loadTensor dequantizes a single GGUF tensor and copies it into the matching
// Born model parameter identified by bornName.
func (wl *weightLoader[B]) loadTensor(ggufName, bornName string) error {
	float32Data, shape, err := wl.converter.Convert(ggufName)
	if err != nil {
		return fmt.Errorf("convert: %w", err)
	}

	tShape := make(tensor.Shape, len(shape))
	copy(tShape, shape)

	// Create a temporary RawTensor to hold the dequantized data.
	raw, err := tensor.NewRaw(tShape, tensor.Float32, wl.backend.Device())
	if err != nil {
		return fmt.Errorf("create raw: %w", err)
	}
	copy(raw.AsFloat32(), float32Data)

	// Route the tensor to the correct model parameter based on its Born name.
	return wl.setParameter(bornName, raw)
}

// setParameter routes a raw tensor to the correct model parameter field.
// Born weight names follow the convention established in internal/loader/mapper.go.
func (wl *weightLoader[B]) setParameter(bornName string, raw *tensor.RawTensor) error {
	m := wl.model

	switch {
	case bornName == bornNameEmbeddingWeight:
		return copyToParam(m.Embed.Weight, raw)

	case bornName == "norm.weight":
		return copyToParam(m.Norm.Gamma, raw)

	case bornName == "lm_head.weight":
		wl.lmHeadLoaded = true
		return copyToLinearWeight(m.Head, raw)

	case strings.HasPrefix(bornName, "layers."):
		return wl.setLayerParameter(bornName, raw)

	default:
		// Unknown name: silently skip to tolerate extra tensors in GGUF files.
		return nil
	}
}

// setLayerParameter routes a weight to the correct field within a specific layer.
// Layer names follow the pattern: "layers.{i}.{component}.{field}".
func (wl *weightLoader[B]) setLayerParameter(bornName string, raw *tensor.RawTensor) error {
	// bornName: "layers.{i}.attn.q.weight", "layers.{i}.norm1.weight", etc.
	parts := strings.Split(bornName, ".")
	if len(parts) < 4 {
		return nil // Too short, skip.
	}

	var layerIdx int
	if n, _ := fmt.Sscanf(parts[1], "%d", &layerIdx); n != 1 {
		return nil // Not a numeric layer index, skip.
	}

	if layerIdx < 0 || layerIdx >= len(wl.model.Layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layerIdx, len(wl.model.Layers))
	}

	layer := wl.model.Layers[layerIdx]
	component := parts[2] // "attn", "ffn", "norm1", "norm2"

	switch component {
	case "norm1":
		// Pre-attention RMSNorm gamma.
		return copyToParam(layer.AttnNorm.Gamma, raw)

	case "norm2":
		// Pre-FFN RMSNorm gamma.
		return copyToParam(layer.FFNNorm.Gamma, raw)

	case "attn":
		if len(parts) < 5 {
			return nil
		}
		// "layers.{i}.attn.{q|k|v|o}.weight"
		return wl.setAttnWeight(layer, parts[3], raw)

	case "ffn":
		if len(parts) < 5 {
			return nil
		}
		// "layers.{i}.ffn.{gate|up|down}.weight"
		return wl.setFFNWeight(layer, parts[3], raw)

	default:
		return nil // Unknown component, skip.
	}
}

// setAttnWeight copies raw data to the correct attention projection parameter.
func (wl *weightLoader[B]) setAttnWeight(layer *Layer[B], proj string, raw *tensor.RawTensor) error {
	switch proj {
	case "q":
		return copyToLinearWeight(layer.QProj, raw)
	case "k":
		return copyToLinearWeight(layer.KProj, raw)
	case "v":
		return copyToLinearWeight(layer.VProj, raw)
	case "o":
		return copyToLinearWeight(layer.OProj, raw)
	default:
		return nil
	}
}

// setFFNWeight copies raw data to the correct FFN projection parameter.
func (wl *weightLoader[B]) setFFNWeight(layer *Layer[B], proj string, raw *tensor.RawTensor) error {
	switch proj {
	case "gate":
		return copyToLinearWeight(layer.FFN.GateProj(), raw)
	case "up":
		return copyToLinearWeight(layer.FFN.UpProj(), raw)
	case "down":
		return copyToLinearWeight(layer.FFN.DownProj(), raw)
	default:
		return nil
	}
}

// copyToParam copies float32 data from raw into a Parameter tensor.
// It validates that shapes match before copying.
func copyToParam[B tensor.Backend](param *nn.Parameter[B], raw *tensor.RawTensor) error {
	paramRaw := param.Tensor().Raw()
	if !paramRaw.Shape().Equal(raw.Shape()) {
		return fmt.Errorf("shape mismatch: model expects %v, got %v",
			paramRaw.Shape(), raw.Shape())
	}
	copy(paramRaw.AsFloat32(), raw.AsFloat32())
	return nil
}

// copyToLinearWeight copies float32 data into a Linear layer's weight parameter.
func copyToLinearWeight[B tensor.Backend](linear *nn.Linear[B], raw *tensor.RawTensor) error {
	return copyToParam(linear.Weight(), raw)
}
