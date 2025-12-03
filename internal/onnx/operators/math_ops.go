//go:build !wasm

package operators

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// registerMathOps adds math operators to the registry.
func (r *Registry) registerMathOps() {
	r.Register("Add", handleAdd)
	r.Register("Sub", handleSub)
	r.Register("Mul", handleMul)
	r.Register("Div", handleDiv)
	r.Register("MatMul", handleMatMul)
	r.Register("Gemm", handleGemm)
	r.Register("Sqrt", handleSqrt)
	r.Register("Exp", handleExp)
	r.Register("Log", handleLog)
	r.Register("Sum", handleSum)
}

func handleAdd(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("add requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.Add(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleSub(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("sub requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.Sub(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleMul(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("mul requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.Mul(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleDiv(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("div requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.Div(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleMatMul(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("matMul requires 2 inputs, got %d", len(inputs))
	}
	result := ctx.Backend.MatMul(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

// handleGemm implements General Matrix Multiplication: Y = alpha*A*B + beta*C.
func handleGemm(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("gemm requires at least 2 inputs, got %d", len(inputs))
	}

	// Get attributes
	alpha := GetAttrFloat(node, "alpha", 1.0)
	beta := GetAttrFloat(node, "beta", 1.0)
	transA := GetAttrInt(node, "transA", 0) != 0
	transB := GetAttrInt(node, "transB", 0) != 0

	a := inputs[0]
	b := inputs[1]

	// Transpose if needed
	if transA {
		a = ctx.Backend.Transpose(a)
	}
	if transB {
		b = ctx.Backend.Transpose(b)
	}

	// Compute A @ B
	result := ctx.Backend.MatMul(a, b)

	// Scale by alpha
	if alpha != 1.0 {
		result = ctx.Backend.MulScalar(result, alpha)
	}

	// Add bias (C) scaled by beta
	if len(inputs) >= 3 && beta != 0 {
		c := inputs[2]
		if beta != 1.0 {
			c = ctx.Backend.MulScalar(c, beta)
		}
		result = ctx.Backend.Add(result, c)
	}

	return []*tensor.RawTensor{result}, nil
}

func handleSqrt(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("sqrt requires 1 input, got %d", len(inputs))
	}
	result := ctx.Backend.Sqrt(inputs[0])
	return []*tensor.RawTensor{result}, nil
}

func handleExp(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("exp requires 1 input, got %d", len(inputs))
	}
	result := ctx.Backend.Exp(inputs[0])
	return []*tensor.RawTensor{result}, nil
}

func handleLog(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("log requires 1 input, got %d", len(inputs))
	}
	result := ctx.Backend.Log(inputs[0])
	return []*tensor.RawTensor{result}, nil
}

func handleSum(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("sum requires at least 1 input")
	}
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = ctx.Backend.Add(result, inputs[i])
	}
	return []*tensor.RawTensor{result}, nil
}
