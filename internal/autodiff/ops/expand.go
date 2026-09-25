package ops

import "github.com/born-ml/born/internal/tensor"

// ExpandOp represents a broadcast-expand operation: output = Expand(x, targetShape).
//
// Forward:
//
//	y = Expand(x, targetShape)   — broadcasts x to a larger shape
//
// Backward:
//
//	grad_x = reduceBroadcast(grad_y, x.shape)
//
// The output gradient has targetShape; we reduce it back to the original
// input shape by summing along every dimension that was broadcast.
// This is the same logic used in AddOp / MulOp for broadcast gradients,
// expressed through the shared reduceBroadcast helper.
type ExpandOp struct {
	input      *tensor.RawTensor // original (smaller) input
	output     *tensor.RawTensor // expanded output
	inputShape tensor.Shape      // shape captured at op creation time
}

// NewExpandOp creates an ExpandOp that records the forward-pass input and output.
func NewExpandOp(input, output *tensor.RawTensor) *ExpandOp {
	return &ExpandOp{
		input:      input,
		output:     output,
		inputShape: input.Shape().Clone(),
	}
}

// Inputs returns the input tensors [x].
func (op *ExpandOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the expanded output tensor.
func (op *ExpandOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward reduces the upstream gradient to the original input shape by
// summing along all broadcast dimensions.
//
// Uses the shared reduceBroadcast helper (same one used by AddOp) which
// handles leading-dimension padding and size-1 reductions correctly, and
// stays on GPU via backend.SumDim — no CPU readback.
func (op *ExpandOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	gradX := reduceBroadcast(outputGrad, op.inputShape, backend)
	return []*tensor.RawTensor{gradX}
}
