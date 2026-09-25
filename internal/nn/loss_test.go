package nn_test

import (
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// TestMSELoss_GradientFlows verifies that MSELoss records on the autodiff tape
// so that gradients propagate back through predictions.
//
// Before the fix, MSELoss used .AsFloat32() + a Go loop (CPU readback), which
// broke the gradient chain: the tape saw nothing after squared = diff.Mul(diff).
// After the fix, Sum + DivScalar are backend ops and appear on the tape.
func TestMSELoss_GradientFlows(t *testing.T) {
	backend := autodiff.New(cpu.New())

	backend.Tape().StartRecording()
	defer backend.Tape().StopRecording()

	// predictions: [1, 2, 3], targets: [1, 1, 1]
	// diff = [0, 1, 2], squared = [0, 1, 4], mean = 5/3
	predictions, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	targets, _ := tensor.FromSlice([]float32{1, 1, 1}, tensor.Shape{3}, backend)

	mse := nn.NewMSELoss(backend)
	loss := mse.Forward(predictions, targets)

	// Tape must have recorded at least one operation.
	if backend.Tape().NumOps() == 0 {
		t.Fatal("MSELoss forward recorded no ops on the tape — gradient chain is broken")
	}

	// Backward pass must succeed and return a non-nil gradient for predictions.
	grads := autodiff.Backward(loss, backend)

	predGrad := grads[predictions.Raw()]
	if predGrad == nil {
		t.Fatal("MSELoss backward: gradient for predictions is nil — gradients are not flowing")
	}

	// Shape must match predictions.
	if !predGrad.Shape().Equal(predictions.Shape()) {
		t.Errorf("gradient shape = %v, want %v", predGrad.Shape(), predictions.Shape())
	}

	// d(MSE)/dx_i = 2*(x_i - t_i) / N
	// predictions = [1, 2, 3], targets = [1, 1, 1], N = 3
	// expected gradients: [0, 2/3, 4/3]
	wantGrads := []float32{0.0, 2.0 / 3.0, 4.0 / 3.0}
	gotGrads := predGrad.AsFloat32()

	for i, want := range wantGrads {
		got := gotGrads[i]
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-5 {
			t.Errorf("gradient[%d] = %f, want %f (diff %f)", i, got, want, diff)
		}
	}
}

// TestMSELoss_GradientFlows_ValueCorrect verifies the forward value is unchanged
// after the refactor (no regression on correctness).
func TestMSELoss_GradientFlows_ValueCorrect(t *testing.T) {
	backend := autodiff.New(cpu.New())

	backend.Tape().StartRecording()
	defer backend.Tape().StopRecording()

	predictions, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	targets, _ := tensor.FromSlice([]float32{1, 1, 1}, tensor.Shape{3}, backend)

	mse := nn.NewMSELoss(backend)
	loss := mse.Forward(predictions, targets)

	// Expected: mean((0)² + (1)² + (2)²) = 5/3
	const want = float32(5.0 / 3.0)
	got := loss.Raw().AsFloat32()[0]
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-5 {
		t.Errorf("MSE forward value = %f, want %f", got, want)
	}
}
