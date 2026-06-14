// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package llama

import (
	"fmt"
	"math"

	"github.com/born-ml/born/internal/generate"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// AttentionFunc is a function that computes scaled-dot-product attention.
//
// It replaces the default ScaledDotProductAttention inside each transformer layer.
// Useful for research experiments (Flash Attention, sliding-window attention, etc.).
//
// Parameters:
//   - q: Query tensor [batch, n_heads, seq_q, head_dim]
//   - k: Key tensor [batch, n_heads, seq_k, head_dim]
//   - v: Value tensor [batch, n_heads, seq_k, head_dim]
//   - mask: Optional causal mask (nil for no mask)
//   - scale: Pre-computed 1/sqrt(head_dim)
//
// Returns:
//   - output: Attended values [batch, n_heads, seq_q, head_dim]
//   - weights: Attention weights (may be nil if not needed by the implementation)
type AttentionFunc[B tensor.Backend] func(
	q, k, v *tensor.Tensor[float32, B],
	mask *tensor.Tensor[float32, B],
	scale float32,
) (*tensor.Tensor[float32, B], *tensor.Tensor[float32, B])

// Option is a functional option for configuring a LLaMA Model.
type Option[B tensor.Backend] func(*modelOptions[B])

// modelOptions holds optional configuration collected before model creation.
type modelOptions[B tensor.Backend] struct {
	attnFn AttentionFunc[B]
}

// WithAttentionFunc replaces the default scaled-dot-product attention with a
// custom implementation. Use for research (e.g., Flash Attention, local windows).
func WithAttentionFunc[B tensor.Backend](fn AttentionFunc[B]) Option[B] {
	return func(o *modelOptions[B]) {
		o.attnFn = fn
	}
}

// Layer represents a single LLaMA transformer block.
//
// Each layer applies:
//  1. Pre-attention RMSNorm
//  2. Self-attention (GQA) with RoPE
//  3. Residual connection
//  4. Pre-FFN RMSNorm
//  5. SwiGLU FFN
//  6. Residual connection
type Layer[B tensor.Backend] struct {
	AttnNorm *nn.RMSNorm[B] // Pre-attention layer norm.
	QProj    *nn.Linear[B]  // Query projection [hidden, n_q_heads * head_dim].
	KProj    *nn.Linear[B]  // Key projection [hidden, n_kv_heads * head_dim].
	VProj    *nn.Linear[B]  // Value projection [hidden, n_kv_heads * head_dim].
	OProj    *nn.Linear[B]  // Output projection [n_q_heads * head_dim, hidden].
	Rope     *nn.RotaryEncoding[B]
	FFNNorm  *nn.RMSNorm[B]   // Pre-FFN layer norm.
	FFN      *nn.SwiGLUFFN[B] // SwiGLU feed-forward network.

	config  Config
	attnFn  AttentionFunc[B]
	backend B
}

// newLayer creates a new transformer Layer.
func newLayer[B tensor.Backend](cfg Config, opts *modelOptions[B], backend B) *Layer[B] {
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	noBias := nn.WithBias(false)

	rope := nn.NewRotaryEncoding[B](nn.RotaryEncodingConfig{
		DModel:    cfg.HeadDim,
		MaxSeqLen: cfg.MaxSeqLen,
		Theta:     cfg.RopeTheta,
	}, backend)

	return &Layer[B]{
		AttnNorm: nn.NewRMSNorm[B](cfg.HiddenSize, cfg.NormEpsilon, backend),
		QProj:    nn.NewLinear[B](cfg.HiddenSize, qDim, backend, noBias),
		KProj:    nn.NewLinear[B](cfg.HiddenSize, kvDim, backend, noBias),
		VProj:    nn.NewLinear[B](cfg.HiddenSize, kvDim, backend, noBias),
		OProj:    nn.NewLinear[B](qDim, cfg.HiddenSize, backend, noBias),
		Rope:     rope,
		FFNNorm:  nn.NewRMSNorm[B](cfg.HiddenSize, cfg.NormEpsilon, backend),
		FFN: nn.NewSwiGLUFFN[B](nn.SwiGLUFFNConfig{
			EmbedDim: cfg.HiddenSize,
			FFNDim:   cfg.FFNSize,
		}, backend),
		config:  cfg,
		attnFn:  opts.attnFn,
		backend: backend,
	}
}

// DebugForward is like Forward but returns intermediate attention and FFN contributions
// for diagnostic inspection.
//
// Input x: [batch, seq_len, hidden_size].
// Returns out (final output), attnContrib, and ffnContrib tensors.
func (l *Layer[B]) DebugForward(
	x *tensor.Tensor[float32, B],
	cache *nn.KVCache[B],
	startPos int,
) (out, attnContrib, ffnContrib *tensor.Tensor[float32, B]) {
	normed := l.AttnNorm.Forward(x)
	attnContrib = l.selfAttention(normed, cache, startPos)
	x = x.Add(attnContrib)

	normed2 := l.FFNNorm.Forward(x)
	ffnContrib = l.FFN.Forward(normed2)
	out = x.Add(ffnContrib)
	return
}

// Forward computes the output of a single transformer layer.
//
// Input x: [batch, seq_len, hidden_size].
// Returns output with the same shape.
//
// cache and startPos are used for incremental KV-cache decoding during inference.
// Pass cache=nil for training or non-cached inference.
func (l *Layer[B]) Forward(
	x *tensor.Tensor[float32, B],
	cache *nn.KVCache[B],
	startPos int,
) *tensor.Tensor[float32, B] {
	out, _, _ := l.DebugForward(x, cache, startPos)
	return out
}

// selfAttention performs grouped-query self-attention with RoPE.
//
// Input x is already normalized (AttnNorm applied by the caller).
// Output shape equals input shape.
func (l *Layer[B]) selfAttention(
	x *tensor.Tensor[float32, B],
	cache *nn.KVCache[B],
	startPos int,
) *tensor.Tensor[float32, B] {
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]

	cfg := l.config
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Project Q, K, V (reshape for 3D input → 2D linear → reshape back).
	q := projectLinear(x, l.QProj, batch, seqLen) // [batch, seq, n_q_heads * head_dim]
	k := projectLinear(x, l.KProj, batch, seqLen) // [batch, seq, n_kv_heads * head_dim]
	v := projectLinear(x, l.VProj, batch, seqLen) // [batch, seq, n_kv_heads * head_dim]

	// Reshape to [batch, seq, n_heads, head_dim], then transpose to [batch, n_heads, seq, head_dim].
	q = q.Reshape(batch, seqLen, cfg.NumHeads, cfg.HeadDim).Transpose(0, 2, 1, 3)
	k = k.Reshape(batch, seqLen, cfg.NumKVHeads, cfg.HeadDim).Transpose(0, 2, 1, 3)
	v = v.Reshape(batch, seqLen, cfg.NumKVHeads, cfg.HeadDim).Transpose(0, 2, 1, 3)

	// Apply Rotary Position Embeddings.
	q = l.Rope.ForwardWithOffset(q, startPos)
	k = l.Rope.ForwardWithOffset(k, startPos)

	// Update KV cache if provided.
	if cache != nil {
		cache.Update(k, v)
		k, v = cache.Get()
	}

	// Broadcast KV heads to match Q heads for GQA.
	nRep := cfg.NumHeads / cfg.NumKVHeads
	k = nn.RepeatKV(k, nRep)
	v = nn.RepeatKV(v, nRep)

	// Build causal mask when multiple query tokens are present.
	var mask *tensor.Tensor[float32, B]
	if seqLen > 1 {
		seqK := seqLen
		if cache != nil {
			seqK = k.Shape()[2]
		}
		mask = nn.CausalMask(seqK, l.backend)
	}

	// Compute attention — use injected function if provided, else default SDPA.
	var attnOut, attnWeights *tensor.Tensor[float32, B]
	if l.attnFn != nil {
		attnOut, attnWeights = l.attnFn(q, k, v, mask, scale)
	} else {
		attnOut, attnWeights = nn.ScaledDotProductAttention(q, k, v, mask, scale)
	}
	_ = attnWeights // Available for debug inspection via DebugAttnWeights.

	// Transpose back to [batch, seq, n_q_heads, head_dim] then merge heads.
	attnOut = attnOut.Transpose(0, 2, 1, 3)
	attnOut = attnOut.Reshape(batch, seqLen, cfg.NumHeads*cfg.HeadDim)

	// Output projection.
	return projectLinear(attnOut, l.OProj, batch, seqLen)
}

// projectLinear applies a 2D linear layer to a 3D input tensor by flattening
// the batch and sequence dimensions, running the linear, then restoring shape.
func projectLinear[B tensor.Backend](
	x *tensor.Tensor[float32, B],
	linear *nn.Linear[B],
	batch, seq int,
) *tensor.Tensor[float32, B] {
	inDim := x.Shape()[2]
	out2D := linear.Forward(x.Reshape(batch*seq, inDim))
	outDim := out2D.Shape()[1]
	return out2D.Reshape(batch, seq, outDim)
}

// Parameters returns all trainable parameters in this layer.
func (l *Layer[B]) Parameters() []*nn.Parameter[B] {
	params := make([]*nn.Parameter[B], 0, 14) //nolint:mnd // 14 is the exact count: 2 norms + 4 attn projs + 8 FFN params (3 weights × 2 norms via SwiGLU)
	params = append(params, l.AttnNorm.Parameters()...)
	params = append(params, l.QProj.Parameters()...)
	params = append(params, l.KProj.Parameters()...)
	params = append(params, l.VProj.Parameters()...)
	params = append(params, l.OProj.Parameters()...)
	params = append(params, l.FFNNorm.Parameters()...)
	params = append(params, l.FFN.Parameters()...)
	return params
}

// ModelCache holds per-layer KV caches for a LLaMA model.
//
// Each transformer layer requires its own independent KV cache; using a single
// cache for all layers would mix keys and values from different layers.
//
// Pass a ModelCache to Model.Forward to enable incremental decoding.
// Create one with NewModelCache or let the model create one via NewCache.
type ModelCache[B tensor.Backend] struct {
	layers []*nn.KVCache[B]
}

// NewModelCache creates a per-layer KV cache for the given model configuration.
//
// Parameters:
//   - cfg: Model configuration (used to determine cache shape per layer)
//   - maxSeqLen: Maximum sequence length (may differ from cfg.MaxSeqLen for memory control)
//   - backend: Computation backend
func NewModelCache[B tensor.Backend](cfg Config, maxSeqLen int, backend B) *ModelCache[B] {
	caches := make([]*nn.KVCache[B], cfg.NumLayers)
	for i := range caches {
		caches[i] = nn.NewKVCache[B](1, cfg.NumKVHeads, maxSeqLen, cfg.HeadDim, backend)
	}
	return &ModelCache[B]{layers: caches}
}

// Clear resets all per-layer caches.
// Satisfies the generate.KVCache interface so ModelCache can be passed to TextGenerator.
func (c *ModelCache[B]) Clear() {
	for _, lc := range c.layers {
		lc.Clear()
	}
}

// Model is a LLaMA causal language model.
//
// It implements generate.LLMModel and can be used with generate.TextGenerator
// for text generation. Weights can be loaded from GGUF via LoadGGUF.
//
// Type parameter B is the computation backend (e.g., cpu.Backend, webgpu.Backend).
//
// Example:
//
//	backend := cpu.New()
//	model, err := llama.LoadGGUF("model.gguf", backend)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	// model implements generate.LLMModel
//	cache := llama.NewModelCache(model.Config, model.Config.MaxSeqLen, backend)
//	gen := generate.NewTextGenerator(model, tok, generate.DefaultSamplingConfig())
//	gen.SetCache(cache)
type Model[B tensor.Backend] struct {
	// Embed is the token embedding table [vocab_size, hidden_size].
	Embed *nn.Embedding[B]
	// Layers are the transformer decoder blocks.
	Layers []*Layer[B]
	// Norm is the final RMSNorm before the output projection.
	Norm *nn.RMSNorm[B]
	// Head is the language-model head that projects to vocabulary logits.
	// LLaMA may share weights between Embed and Head (tie_word_embeddings).
	Head *nn.Linear[B]
	// Config holds the model hyperparameters.
	Config Config

	backend B
}

// NewModel creates an uninitialised LLaMA Model with random weights.
//
// Weights must be loaded via LoadStateDict or by calling LoadGGUF.
// opts allow injecting a custom attention function (see WithAttentionFunc).
func NewModel[B tensor.Backend](cfg Config, backend B, opts ...Option[B]) *Model[B] {
	options := &modelOptions[B]{}
	for _, o := range opts {
		o(options)
	}

	layers := make([]*Layer[B], cfg.NumLayers)
	for i := range layers {
		layers[i] = newLayer(cfg, options, backend)
	}

	return &Model[B]{
		Embed:   nn.NewEmbedding[B](cfg.VocabSize, cfg.HiddenSize, backend),
		Layers:  layers,
		Norm:    nn.NewRMSNorm[B](cfg.HiddenSize, cfg.NormEpsilon, backend),
		Head:    nn.NewLinear[B](cfg.HiddenSize, cfg.VocabSize, backend, nn.WithBias(false)),
		Config:  cfg,
		backend: backend,
	}
}

// Release frees all GPU buffers held by the model's parameters.
// Call when the model is no longer needed to reclaim GPU memory immediately
// instead of waiting for GC. Safe to call multiple times.
func (m *Model[B]) Release() {
	release := func(params []*nn.Parameter[B]) {
		for _, p := range params {
			p.Tensor().Raw().ReleaseGPU()
		}
	}
	release(m.Embed.Parameters())
	for _, layer := range m.Layers {
		release(layer.Parameters())
	}
	release(m.Norm.Parameters())
	release(m.Head.Parameters())
}

// Forward performs a forward pass and returns logits.
//
// This method satisfies the generate.LLMModel interface.
//
// Parameters:
//   - input: Token ids [batch, seq_len] as a *tensor.RawTensor (int32).
//   - cache: Optional *ModelCache[B] for incremental KV-cache decoding.
//     Pass nil for full-sequence forward (training or non-cached inference).
//     The cache must have been created with NewModelCache for this model.
//   - startPos: Position offset for RoPE (0 for full-sequence; equals number of
//     tokens already in the cache for incremental decoding).
//
// Returns logits [batch, seq_len, vocab_size] as a *tensor.RawTensor.
func (m *Model[B]) Forward(
	input *tensor.RawTensor,
	cache generate.KVCache,
	startPos int,
) *tensor.RawTensor {
	// Convert raw int32 input to typed tensor for embedding lookup.
	inputTyped := tensor.New[int32, B](input, m.backend)

	// Token embeddings: [batch, seq, hidden_size].
	hidden := m.Embed.Forward(inputTyped)

	// Cast cache to the model-specific per-layer cache type.
	var modelCache *ModelCache[B]
	if cache != nil {
		if typed, ok := cache.(*ModelCache[B]); ok {
			modelCache = typed
		}
	}

	// Pass through all transformer layers, providing each with its own KV cache slice.
	for i, layer := range m.Layers {
		var layerCache *nn.KVCache[B]
		if modelCache != nil && i < len(modelCache.layers) {
			layerCache = modelCache.layers[i]
		}
		hidden = layer.Forward(hidden, layerCache, startPos)
	}

	// Final norm.
	hidden = m.Norm.Forward(hidden)

	// LM head: project to vocab logits.
	shape := hidden.Shape()
	batch, seqLen := shape[0], shape[1]
	hidden2D := hidden.Reshape(batch*seqLen, m.Config.HiddenSize)
	logits2D := m.Head.Forward(hidden2D)
	logits := logits2D.Reshape(batch, seqLen, m.Config.VocabSize)

	return logits.Raw()
}

// VocabSize returns the vocabulary size.
// Satisfies the generate.LLMModel interface.
func (m *Model[B]) VocabSize() int {
	return m.Config.VocabSize
}

// SetAttentionFunc replaces the attention scoring function on all layers.
// Pass nil to restore default ScaledDotProductAttention.
func (m *Model[B]) SetAttentionFunc(fn AttentionFunc[B]) {
	for _, l := range m.Layers {
		l.attnFn = fn
	}
}

// Parameters returns all trainable parameters of the model.
func (m *Model[B]) Parameters() []*nn.Parameter[B] {
	var params []*nn.Parameter[B]
	params = append(params, m.Embed.Parameters()...)
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	params = append(params, m.Norm.Parameters()...)
	params = append(params, m.Head.Parameters()...)
	return params
}

// validate checks that the Config is internally consistent.
func (m *Model[B]) validate() error {
	cfg := m.Config

	if cfg.VocabSize <= 0 {
		return fmt.Errorf("llama: VocabSize must be positive, got %d", cfg.VocabSize)
	}
	if cfg.HiddenSize <= 0 {
		return fmt.Errorf("llama: HiddenSize must be positive, got %d", cfg.HiddenSize)
	}
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("llama: NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.NumHeads <= 0 {
		return fmt.Errorf("llama: NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.NumKVHeads <= 0 {
		return fmt.Errorf("llama: NumKVHeads must be positive, got %d", cfg.NumKVHeads)
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("llama: NumHeads (%d) must be divisible by NumKVHeads (%d)",
			cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim <= 0 {
		return fmt.Errorf("llama: HeadDim must be positive, got %d", cfg.HeadDim)
	}
	if cfg.FFNSize <= 0 {
		return fmt.Errorf("llama: FFNSize must be positive, got %d", cfg.FFNSize)
	}

	return nil
}
