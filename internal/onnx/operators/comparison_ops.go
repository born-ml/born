//go:build !wasm

package operators

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// registerComparisonOps adds comparison operators to the registry.
func (r *Registry) registerComparisonOps() {
	r.Register("Equal", handleEqual)
}

func handleEqual(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("equal requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.Equal(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}
