package autodiff

import (
	"github.com/born-ml/born/internal/autodiff/ops"
	"github.com/born-ml/born/internal/tensor"
)

// GradientTape records operations during the forward pass and computes
// gradients during the backward pass using reverse-mode automatic differentiation.
//
// Usage:
//
//	tape := NewGradientTape()
//	tape.StartRecording()
//	// ... perform operations ...
//	gradients := tape.Backward(outputGrad, backend)
type GradientTape struct {
	operations []ops.Operation // Recorded operations (in execution order)
	recording  bool            // Whether tape is currently recording
}

// NewGradientTape creates a new gradient tape.
func NewGradientTape() *GradientTape {
	return &GradientTape{
		operations: make([]ops.Operation, 0, 64), // Pre-allocate for common case
		recording:  false,
	}
}

// StartRecording enables operation recording.
func (t *GradientTape) StartRecording() {
	t.recording = true
}

// StopRecording disables operation recording.
func (t *GradientTape) StopRecording() {
	t.recording = false
}

// IsRecording returns true if the tape is currently recording operations.
func (t *GradientTape) IsRecording() bool {
	return t.recording
}

// Record adds an operation to the tape.
// Only records if the tape is currently recording.
func (t *GradientTape) Record(op ops.Operation) {
	if t.recording {
		t.operations = append(t.operations, op)
	}
}

// Clear resets the tape, removing all recorded operations.
// Recording state is preserved.
//
// GPU buffer release for intermediate outputs is handled via the BackendReleaser
// path in callers (parameter.go, adam.go, sgd.go) and by Backend.ReclaimMemory()
// which releases all non-persistent live GPU tensors (ADR-019 Phase 4).
func (t *GradientTape) Clear() {
	// Nil out slice elements so GC can collect Operation objects and their
	// referenced RawTensors. Without this, the backing array retains pointers
	// to all operations from the previous step — each holding input/output
	// RawTensors with CPU buffers. This was the #1 memory leak: 6.7 GB / 97%
	// of heap was unreachable tensorBuffers held alive by stale slice entries.
	for i := range t.operations {
		t.operations[i] = nil
	}
	t.operations = t.operations[:0]
}

// Backward computes gradients for all inputs by walking the tape in reverse.
//
// Algorithm:
//  1. Start with the output gradient (typically ones for scalar loss)
//  2. Walk operations in reverse order
//  3. For each operation, compute input gradients using chain rule
//  4. Accumulate gradients when the same tensor is used multiple times
//
// Returns a map from RawTensor to its accumulated gradient.
//
// outputTensor identifies the loss tensor on the tape. The backward pass
// starts from the operation that produced outputTensor, ignoring any
// operations recorded after it (e.g., metrics, logging). If outputTensor
// is nil or not found on the tape, falls back to the last recorded op.
func (t *GradientTape) Backward(outputTensor *tensor.RawTensor, outputGrad *tensor.RawTensor, backend tensor.Backend) map[*tensor.RawTensor]*tensor.RawTensor {
	if len(t.operations) == 0 {
		return make(map[*tensor.RawTensor]*tensor.RawTensor)
	}

	// Temporarily stop recording so backward's own ops (gradient computation)
	// are not captured on tape. Restore recording state after backward completes
	// so callers that rely on persistent recording (e.g., HRM trainer) continue
	// to record forward ops in subsequent steps.
	//
	// Optimizer.Step() ops that run AFTER Backward but BEFORE ClearTape will be
	// recorded. ClearTape() stops recording and releases their GPU buffers safely
	// because moments are replaced (not the same tensor objects as tape outputs).
	wasRecording := t.recording
	t.recording = false
	defer func() { t.recording = wasRecording }()

	// Map to accumulate gradients for each tensor
	grads := make(map[*tensor.RawTensor]*tensor.RawTensor)

	// Find the operation that produced the loss tensor.
	rootIdx := len(t.operations) - 1
	if outputTensor != nil {
		for i := len(t.operations) - 1; i >= 0; i-- {
			if t.operations[i].Output() == outputTensor {
				rootIdx = i
				break
			}
		}
	}
	grads[t.operations[rootIdx].Output()] = outputGrad

	// Walk tape backwards from rootIdx, skipping ops recorded after loss.
	for i := rootIdx; i >= 0; i-- {
		op := t.operations[i]
		inputGrads := t.computeInputGrads(op, grads, backend)
		if inputGrads == nil {
			continue
		}
		t.accumulateGrads(op, inputGrads, grads, backend)

		// NOTE: eager release removed (was commit 11a7999). op.Output() may be
		// shared as input to another op (e.g. Transpose output used by MatMul).
		// Releasing here causes use-after-free when that other op's backward
		// tries to read the buffer. Buffers are freed in ClearTape() instead.
	}

	return grads
}

// computeInputGrads computes gradients for an operation's inputs.
// Returns nil if no gradient flows to this operation.
func (t *GradientTape) computeInputGrads(
	op ops.Operation,
	grads map[*tensor.RawTensor]*tensor.RawTensor,
	backend tensor.Backend,
) []*tensor.RawTensor {
	// Check if this is a multi-output operation (e.g., Chunk)
	multiOp, isMulti := op.(ops.MultiOutputOperation)
	if isMulti {
		return t.computeMultiOutputGrads(multiOp, grads, backend)
	}
	return t.computeSingleOutputGrads(op, grads, backend)
}

// computeMultiOutputGrads handles backward pass for multi-output operations.
func (t *GradientTape) computeMultiOutputGrads(
	multiOp ops.MultiOutputOperation,
	grads map[*tensor.RawTensor]*tensor.RawTensor,
	backend tensor.Backend,
) []*tensor.RawTensor {
	outputs := multiOp.Outputs()
	outputGrads, hasAnyGrad := t.collectOutputGrads(outputs, grads)
	if !hasAnyGrad {
		return nil
	}
	t.fillMissingGradsWithZeros(outputs, outputGrads, backend)
	return multiOp.BackwardMulti(outputGrads, backend)
}

// computeSingleOutputGrads handles backward pass for single-output operations.
func (t *GradientTape) computeSingleOutputGrads(
	op ops.Operation,
	grads map[*tensor.RawTensor]*tensor.RawTensor,
	backend tensor.Backend,
) []*tensor.RawTensor {
	opOutput := op.Output()
	opOutputGrad, hasGrad := grads[opOutput]
	if !hasGrad {
		return nil
	}
	return op.Backward(opOutputGrad, backend)
}

// collectOutputGrads collects gradients for all outputs of a multi-output operation.
func (t *GradientTape) collectOutputGrads(
	outputs []*tensor.RawTensor,
	grads map[*tensor.RawTensor]*tensor.RawTensor,
) ([]*tensor.RawTensor, bool) {
	outputGrads := make([]*tensor.RawTensor, len(outputs))
	hasAnyGrad := false
	for j, out := range outputs {
		if grad, exists := grads[out]; exists {
			outputGrads[j] = grad
			hasAnyGrad = true
		}
	}
	return outputGrads, hasAnyGrad
}

// fillMissingGradsWithZeros fills nil gradients with zero tensors.
func (t *GradientTape) fillMissingGradsWithZeros(
	outputs []*tensor.RawTensor,
	outputGrads []*tensor.RawTensor,
	backend tensor.Backend,
) {
	for j, out := range outputs {
		if outputGrads[j] != nil {
			continue
		}
		zeroGrad, err := tensor.NewRaw(out.Shape(), out.DType(), backend.Device())
		if err != nil {
			continue // Skip if can't create zero grad
		}
		outputGrads[j] = zeroGrad
	}
}

// accumulateGrads accumulates gradients for each input tensor.
// When a gradient already exists for an input (fan-out in the graph), the two
// partial gradients are summed and the old buffer is released immediately to
// avoid accumulating unreferenced GPU buffers across the backward pass.
func (t *GradientTape) accumulateGrads(
	op ops.Operation,
	inputGrads []*tensor.RawTensor,
	grads map[*tensor.RawTensor]*tensor.RawTensor,
	backend tensor.Backend,
) {
	inputs := op.Inputs()
	for j, input := range inputs {
		if j >= len(inputGrads) {
			break
		}
		inputGrad := inputGrads[j]
		if inputGrad == nil {
			continue
		}
		if existing, ok := grads[input]; ok {
			newGrad := backend.Add(existing, inputGrad)
			// Release the old accumulated gradient buffer now that it has been
			// consumed by Add. Uses BackendReleaser when available (ADR-019 Phase 4).
			if br, ok := any(backend).(tensor.BackendReleaser); ok {
				br.ReleaseBackendData(existing.BackendData())
				existing.SetBackendData(nil)
			}
			grads[input] = newGrad
		} else {
			grads[input] = inputGrad
		}
	}
}

// NumOps returns the number of recorded operations.
func (t *GradientTape) NumOps() int {
	return len(t.operations)
}
