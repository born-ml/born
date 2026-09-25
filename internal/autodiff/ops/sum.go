package ops

import "github.com/born-ml/born/internal/tensor"

// SumOp represents a full-reduction sum: output = Sum(x) → scalar.
//
// Forward:
//
//	y = sum(x)        — reduces all elements to a scalar (Shape{})
//
// Backward:
//
//	grad_x = Expand(grad_y, x.shape)
//
// The output gradient is a scalar; we broadcast it back to the input shape
// so that each element of x receives the upstream scalar gradient.
type SumOp struct {
	input      *tensor.RawTensor // original input tensor
	output     *tensor.RawTensor // scalar result
	inputShape tensor.Shape      // shape captured at op creation time
}

// NewSumOp creates a SumOp that records the forward-pass input and output.
func NewSumOp(input, output *tensor.RawTensor) *SumOp {
	return &SumOp{
		input:      input,
		output:     output,
		inputShape: input.Shape().Clone(), // capture shape; input may be mutated
	}
}

// Inputs returns the input tensors [x].
func (op *SumOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the scalar output tensor.
func (op *SumOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward broadcasts the scalar upstream gradient back to the input shape.
//
// d(sum(x))/dx_i = 1 for every element i, so the gradient at each input
// position equals outputGrad (the upstream scalar).
//
// backend.Expand does not support rank-0 → rank-N promotion.  We therefore
// go through reduceBroadcast which has an explicit scalar-upstream path that
// fills a new tensor of the target shape with the scalar value via FullRaw
// (CPU path) or an equivalent device-local fill, matching the pattern used
// throughout the ops package for scalar gradient flow.
func (op *SumOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	gradX := reduceBroadcast(outputGrad, op.inputShape, backend)
	return []*tensor.RawTensor{gradX}
}
