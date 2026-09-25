package ops

import "github.com/born-ml/born/internal/tensor"

// CastOp represents a dtype-conversion operation: output = Cast(x, targetDtype).
//
// Forward:
//
//	y = Cast(x, targetDtype)   — converts every element to targetDtype
//
// Backward:
//
//	grad_x = Cast(grad_y, inputDtype)
//
// Cast is element-wise and value-preserving (up to precision), so the
// gradient passes through by casting it back to the original dtype.
// Shape is unchanged.
type CastOp struct {
	input      *tensor.RawTensor  // original input (used for Inputs())
	output     *tensor.RawTensor  // cast output
	inputDType tensor.DataType    // original dtype; backward casts grad back to this
}

// NewCastOp creates a CastOp that records the forward-pass input and output.
func NewCastOp(input, output *tensor.RawTensor) *CastOp {
	return &CastOp{
		input:      input,
		output:     output,
		inputDType: input.DType(),
	}
}

// Inputs returns the input tensors [x].
func (op *CastOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the cast output tensor.
func (op *CastOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward casts the upstream gradient back to the original input dtype.
//
// Since Cast is a pointwise value-preserving op, ∂Cast(x)/∂x = 1 (in the
// floating-point sense), so the gradient simply passes through after being
// re-cast to the input dtype.  This keeps the gradient tensor consistent
// with the dtype of the parameter it will be applied to.
func (op *CastOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	// If gradient already has the right dtype, pass a clone to avoid aliasing.
	if outputGrad.DType() == op.inputDType {
		return []*tensor.RawTensor{outputGrad.Clone()}
	}
	gradX := backend.Cast(outputGrad, op.inputDType)
	return []*tensor.RawTensor{gradX}
}
