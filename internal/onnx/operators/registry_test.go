//go:build !wasm

package operators

import (
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

func TestNewRegistry(t *testing.T) {
	r := NewRegistry()

	// Check that essential operators are registered
	essentialOps := []string{
		"Add", "Sub", "Mul", "Div", "MatMul",
		"Relu", "Sigmoid", "Tanh", "Softmax",
		"Reshape", "Transpose",
		"Identity", "Dropout",
	}

	for _, op := range essentialOps {
		if _, ok := r.Get(op); !ok {
			t.Errorf("Expected operator %s to be registered", op)
		}
	}
}

func TestRegistryGetUnknown(t *testing.T) {
	r := NewRegistry()

	if _, ok := r.Get("UnknownOp"); ok {
		t.Error("Expected unknown operator to not be found")
	}
}

func TestSupportedOps(t *testing.T) {
	r := NewRegistry()
	ops := r.SupportedOps()

	if len(ops) < 20 {
		t.Errorf("Expected at least 20 supported ops, got %d", len(ops))
	}
}

func TestRegisterCustomOp(t *testing.T) {
	r := NewRegistry()

	// Register custom operator
	r.Register("MyCustomOp", func(_ *Context, _ *Node, _ []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
		return nil, nil
	})

	if _, ok := r.Get("MyCustomOp"); !ok {
		t.Error("Expected custom operator to be registered")
	}
}
