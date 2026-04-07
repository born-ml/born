//go:build !wasm

package operators

import (
	"fmt"

	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

func (r *Registry) registerNormalizationOps() {
	r.Register("LayerNormalization", handleLayerNormalization)
}

func handleLayerNormalization(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("layerNormalization requires 3 inputs, got %d", len(inputs))
	}
	X := inputs[0]
	Scale := inputs[1]
	B := inputs[2]

	epsilon := float32(GetAttrFloat(node, "epsilon", 1e-5))
	hiddenSize := X.Shape()[len(X.Shape())-1]

	xTyped := tensor.New[float32](X, ctx.Backend)
	gammaTyped := tensor.New[float32](Scale, ctx.Backend)
	betaTyped := tensor.New[float32](B, ctx.Backend)

	ln := nn.NewLayerNorm(hiddenSize, epsilon, ctx.Backend)
	ln.Gamma = nn.NewParameter("gamma", gammaTyped)
	ln.Beta = nn.NewParameter("beta", betaTyped)

	output := ln.Forward(xTyped)
	return []*tensor.RawTensor{output.Raw()}, nil
}
