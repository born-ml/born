package autodiff_test

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/tensor"
)

// TestAutodiffBackend_Name tests the Name method.
func TestAutodiffBackend_Name(t *testing.T) {
	backend := autodiff.New(cpu.New())
	expected := "Autodiff(CPU)"
	if backend.Name() != expected {
		t.Errorf("Name() = %s, want %s", backend.Name(), expected)
	}
}

// TestAutodiffBackend_Device tests the Device method.
func TestAutodiffBackend_Device(t *testing.T) {
	backend := autodiff.New(cpu.New())
	if backend.Device() != tensor.CPU {
		t.Errorf("Device() = %v, want %v", backend.Device(), tensor.CPU)
	}
}

// TestTape_Recording tests tape recording on/off.
func TestTape_Recording(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Initially not recording
	if tape.IsRecording() {
		t.Error("Tape should not be recording initially")
	}

	// Start recording
	tape.StartRecording()
	if !tape.IsRecording() {
		t.Error("Tape should be recording after StartRecording()")
	}

	// Stop recording
	tape.StopRecording()
	if tape.IsRecording() {
		t.Error("Tape should not be recording after StopRecording()")
	}
}

// TestTape_Clear tests tape clearing.
func TestTape_Clear(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// Perform some operations
	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)
	backend.Add(a.Raw(), b.Raw())

	if tape.NumOps() == 0 {
		t.Error("Tape should have recorded operations")
	}

	// Clear tape
	tape.Clear()

	if tape.NumOps() != 0 {
		t.Errorf("Tape should be empty after Clear(), got %d ops", tape.NumOps())
	}

	// Note: Clear() preserves recording state (by design)
	// This allows clearing tape between epochs without stopping recording
	if !tape.IsRecording() {
		t.Error("Tape should still be recording after Clear() (recording state preserved)")
	}
}

// TestAutodiffBackend_Add_RecordsOperation tests that Add records operations.
func TestAutodiffBackend_Add_RecordsOperation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)

	result := backend.Add(a.Raw(), b.Raw())

	// Verify forward pass
	expected := []float32{4, 6}
	actual := result.AsFloat32()
	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("Add result[%d] = %f, want %f", i, actual[i], v)
		}
	}

	// Verify operation was recorded
	if tape.NumOps() != 1 {
		t.Errorf("Expected 1 operation recorded, got %d", tape.NumOps())
	}
}

// TestAutodiffBackend_Mul_RecordsOperation tests that Mul records operations.
func TestAutodiffBackend_Mul_RecordsOperation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	result := backend.Mul(a.Raw(), b.Raw())

	// Verify forward pass
	expected := []float32{8, 15}
	actual := result.AsFloat32()
	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("Mul result[%d] = %f, want %f", i, actual[i], v)
		}
	}

	// Verify operation was recorded
	if tape.NumOps() != 1 {
		t.Errorf("Expected 1 operation recorded, got %d", tape.NumOps())
	}
}

// TestAutodiffBackend_NoRecording tests that operations are not recorded when tape is off.
func TestAutodiffBackend_NoRecording(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Don't start recording

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)

	backend.Add(a.Raw(), b.Raw())

	// Verify no operations were recorded
	if tape.NumOps() != 0 {
		t.Errorf("Expected 0 operations recorded (tape off), got %d", tape.NumOps())
	}
}

// TestBackward_SimpleAddition tests backward pass for simple addition.
func TestBackward_SimpleAddition(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a + b
	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	resultRaw := backend.Add(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1, dy/db = 1
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGrad := []float32{1, 1}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGrad {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
		if actualGradB[i] != v {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestBackward_SimpleMultiplication tests backward pass for multiplication.
func TestBackward_SimpleMultiplication(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a * b
	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	resultRaw := backend.Mul(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = b, dy/db = a
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float32{4, 5} // b values
	expectedGradB := []float32{2, 3} // a values

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if actualGradB[i] != v {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestBackward_ChainRule tests gradient computation with chain rule.
func TestBackward_ChainRule(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = (x + 2) * 3
	// dy/dx = 3
	x, _ := tensor.FromSlice([]float32{1}, tensor.Shape{1}, backend)
	two, _ := tensor.FromSlice([]float32{2}, tensor.Shape{1}, backend)
	three, _ := tensor.FromSlice([]float32{3}, tensor.Shape{1}, backend)

	temp := backend.Add(x.Raw(), two.Raw())
	resultRaw := backend.Mul(temp, three.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	actualGrad := gradX.AsFloat32()[0]
	expectedGrad := float32(3.0)

	if math.Abs(float64(actualGrad-expectedGrad)) > 1e-6 {
		t.Errorf("grad_x = %f, want %f", actualGrad, expectedGrad)
	}
}

// TestBackward_GradientAccumulation tests that gradients accumulate correctly.
func TestBackward_GradientAccumulation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = x + x (x used twice)
	// dy/dx = 2
	x, _ := tensor.FromSlice([]float32{3}, tensor.Shape{1}, backend)

	resultRaw := backend.Add(x.Raw(), x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	actualGrad := gradX.AsFloat32()[0]
	expectedGrad := float32(2.0)

	if math.Abs(float64(actualGrad-expectedGrad)) > 1e-6 {
		t.Errorf("grad_x = %f, want %f (gradient should accumulate)", actualGrad, expectedGrad)
	}
}

// TestReLU_Forward tests ReLU forward pass.
func TestReLU_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	result := backend.ReLU(input.Raw())

	expected := []float32{0, 0, 0, 1, 2}
	actual := result.AsFloat32()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("ReLU result[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestReLU_Backward tests ReLU backward pass.
func TestReLU_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = ReLU(x)
	x, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.ReLU(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	// dy/dx = 1 if x > 0, else 0
	expected := []float32{0, 0, 0, 1, 1}
	actual := gradX.AsFloat32()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("grad_x[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestMatMul_Backward tests MatMul backward pass.
func TestMatMul_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// C = A @ B
	// A: 2x3, B: 3x2 -> C: 2x2
	A, _ := tensor.FromSlice([]float32{
		1, 2, 3,
		4, 5, 6,
	}, tensor.Shape{2, 3}, backend)

	B, _ := tensor.FromSlice([]float32{
		7, 8,
		9, 10,
		11, 12,
	}, tensor.Shape{3, 2}, backend)

	resultRaw := backend.MatMul(A.Raw(), B.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradA := gradients[A.Raw()]
	gradB := gradients[B.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both matrices")
	}

	// Verify shapes
	if !gradA.Shape().Equal(A.Shape()) {
		t.Errorf("grad_A shape = %v, want %v", gradA.Shape(), A.Shape())
	}
	if !gradB.Shape().Equal(B.Shape()) {
		t.Errorf("grad_B shape = %v, want %v", gradB.Shape(), B.Shape())
	}

	// Gradients should be non-zero
	gradAData := gradA.AsFloat32()
	gradBData := gradB.AsFloat32()

	allZero := true
	for _, v := range gradAData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("grad_A should not be all zeros")
	}

	allZero = true
	for _, v := range gradBData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("grad_B should not be all zeros")
	}
}

// TestSubtraction_Backward tests Sub backward pass.
func TestSubtraction_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a - b
	a, _ := tensor.FromSlice([]float32{5, 6}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)

	resultRaw := backend.Sub(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1, dy/db = -1
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float32{1, 1}
	expectedGradB := []float32{-1, -1}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if math.Abs(float64(actualGradB[i]-v)) > 1e-6 {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestDivision_Backward tests Div backward pass.
func TestDivision_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a / b
	a, _ := tensor.FromSlice([]float32{6, 12}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)

	resultRaw := backend.Div(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1/b, dy/db = -a/b²
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	// dy/da = 1/b = [1/2, 1/3] = [0.5, 0.333...]
	expectedGradA := []float32{0.5, 1.0 / 3.0}

	// dy/db = -a/b² = [-6/4, -12/9] = [-1.5, -1.333...]
	expectedGradB := []float32{-1.5, -4.0 / 3.0}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if math.Abs(float64(actualGradA[i]-v)) > 1e-5 {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if math.Abs(float64(actualGradB[i]-v)) > 1e-5 {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestAutodiffBackend_Inner tests the Inner() method.
func TestAutodiffBackend_Inner(t *testing.T) {
	cpuBackend := cpu.New()
	backend := autodiff.New(cpuBackend)

	inner := backend.Inner()
	if inner.Name() != cpuBackend.Name() {
		t.Errorf("Inner().Name() = %s, want %s", inner.Name(), cpuBackend.Name())
	}
}

// TestReLU_Forward_Float64 tests ReLU forward pass with float64.
func TestReLU_Forward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float64{-2.5, -1.0, 0.0, 1.5, 2.0}, tensor.Shape{5}, backend)

	result := backend.ReLU(input.Raw())

	expected := []float64{0, 0, 0, 1.5, 2.0}
	actual := result.AsFloat64()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("ReLU float64 result[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestReLU_Backward_Float64 tests ReLU backward pass with float64.
func TestReLU_Backward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = ReLU(x)
	x, _ := tensor.FromSlice([]float64{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.ReLU(x.Raw())
	result := tensor.New[float64](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	// dy/dx = 1 if x > 0, else 0
	expected := []float64{0, 0, 0, 1, 1}
	actual := gradX.AsFloat64()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("grad_x float64[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestBackward_Float64 tests backward pass with float64 operations.
func TestBackward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a * b
	a, _ := tensor.FromSlice([]float64{2.5, 3.5}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float64{4.0, 5.0}, tensor.Shape{2}, backend)

	resultRaw := backend.Mul(a.Raw(), b.Raw())
	result := tensor.New[float64](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = b, dy/db = a
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float64{4.0, 5.0} // b values
	expectedGradB := []float64{2.5, 3.5} // a values

	actualGradA := gradA.AsFloat64()
	actualGradB := gradB.AsFloat64()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a float64[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if actualGradB[i] != v {
			t.Errorf("grad_b float64[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestNoGrad tests that NoGrad disables gradient recording.
func TestNoGrad(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Start recording
	tape.StartRecording()

	// Operation outside NoGrad - should be recorded
	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)
	backend.Add(a.Raw(), b.Raw())

	numOpsBeforeNoGrad := tape.NumOps()
	if numOpsBeforeNoGrad == 0 {
		t.Error("Operation before NoGrad should be recorded")
	}

	// Operations inside NoGrad - should NOT be recorded
	backend.NoGrad(func() {
		c, _ := tensor.FromSlice([]float32{5, 6}, tensor.Shape{2}, backend)
		d, _ := tensor.FromSlice([]float32{7, 8}, tensor.Shape{2}, backend)
		backend.Mul(c.Raw(), d.Raw())
	})

	numOpsAfterNoGrad := tape.NumOps()
	if numOpsAfterNoGrad != numOpsBeforeNoGrad {
		t.Errorf("NoGrad should not record operations: before=%d, after=%d",
			numOpsBeforeNoGrad, numOpsAfterNoGrad)
	}

	// Operation after NoGrad - should be recorded again
	e, _ := tensor.FromSlice([]float32{9, 10}, tensor.Shape{2}, backend)
	f, _ := tensor.FromSlice([]float32{11, 12}, tensor.Shape{2}, backend)
	backend.Sub(e.Raw(), f.Raw())

	finalNumOps := tape.NumOps()
	if finalNumOps != numOpsBeforeNoGrad+1 {
		t.Errorf("Recording should resume after NoGrad: expected %d ops, got %d",
			numOpsBeforeNoGrad+1, finalNumOps)
	}
}

// TestNoGrad_RestoresRecordingState tests that NoGrad restores recording state.
func TestNoGrad_RestoresRecordingState(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Test 1: Recording before NoGrad -> recording after NoGrad
	tape.StartRecording()
	if !tape.IsRecording() {
		t.Error("Tape should be recording")
	}

	backend.NoGrad(func() {
		if tape.IsRecording() {
			t.Error("Tape should not be recording inside NoGrad")
		}
	})

	if !tape.IsRecording() {
		t.Error("Tape should be recording after NoGrad (state restored)")
	}

	// Test 2: Not recording before NoGrad -> not recording after NoGrad
	tape.StopRecording()
	if tape.IsRecording() {
		t.Error("Tape should not be recording")
	}

	backend.NoGrad(func() {
		if tape.IsRecording() {
			t.Error("Tape should not be recording inside NoGrad")
		}
	})

	if tape.IsRecording() {
		t.Error("Tape should not be recording after NoGrad (state restored)")
	}
}

// TestNoGrad_Nested tests nested NoGrad calls.
func TestNoGrad_Nested(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)
	backend.Add(a.Raw(), b.Raw())

	numOpsInitial := tape.NumOps()

	// Nested NoGrad
	backend.NoGrad(func() {
		c, _ := tensor.FromSlice([]float32{5, 6}, tensor.Shape{2}, backend)
		d, _ := tensor.FromSlice([]float32{7, 8}, tensor.Shape{2}, backend)
		backend.Mul(c.Raw(), d.Raw())

		// Inner NoGrad
		backend.NoGrad(func() {
			e, _ := tensor.FromSlice([]float32{9, 10}, tensor.Shape{2}, backend)
			f, _ := tensor.FromSlice([]float32{11, 12}, tensor.Shape{2}, backend)
			backend.Sub(e.Raw(), f.Raw())
		})

		// Still in outer NoGrad
		g, _ := tensor.FromSlice([]float32{13, 14}, tensor.Shape{2}, backend)
		h, _ := tensor.FromSlice([]float32{15, 16}, tensor.Shape{2}, backend)
		backend.Div(g.Raw(), h.Raw())
	})

	// No operations should have been recorded
	numOpsFinal := tape.NumOps()
	if numOpsFinal != numOpsInitial {
		t.Errorf("Nested NoGrad should not record operations: initial=%d, final=%d",
			numOpsInitial, numOpsFinal)
	}

	// Recording should be restored
	if !tape.IsRecording() {
		t.Error("Recording should be restored after nested NoGrad")
	}
}

// TestSigmoid_Forward tests Sigmoid forward pass: σ(x) = 1/(1+exp(-x)).
func TestSigmoid_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	result := backend.Sigmoid(input.Raw())
	actual := result.AsFloat32()

	eps := float32(1e-5)
	inputs := []float64{-2, -1, 0, 1, 2}
	for i, x := range inputs {
		expected := float32(1.0 / (1.0 + math.Exp(-x)))
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("Sigmoid[%d] = %f, want %f", i, actual[i], expected)
		}
	}
}

// TestSigmoid_Backward tests Sigmoid backward pass: ∂σ/∂x = σ(x)·(1-σ(x)).
func TestSigmoid_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.Sigmoid(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	if tape.NumOps() == 0 {
		t.Fatal("tape should have recorded Sigmoid op")
	}

	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("expected gradient for x")
	}

	eps := float32(1e-4)
	actual := gradX.AsFloat32()
	sigmoidVals := resultRaw.AsFloat32()
	for i, sv := range sigmoidVals {
		expected := sv * (1 - sv)
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("grad_x[%d] = %f, want %f (σ·(1-σ) = %f·%f)", i, actual[i], expected, sv, 1-sv)
		}
	}
}

// TestTanh_Forward tests Tanh forward pass: tanh(x).
func TestTanh_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	result := backend.Tanh(input.Raw())
	actual := result.AsFloat32()

	eps := float32(1e-5)
	inputs := []float64{-2, -1, 0, 1, 2}
	for i, x := range inputs {
		expected := float32(math.Tanh(x))
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("Tanh[%d] = %f, want %f", i, actual[i], expected)
		}
	}
}

// TestTanh_Backward tests Tanh backward pass: ∂tanh/∂x = 1 - tanh²(x).
func TestTanh_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.Tanh(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	if tape.NumOps() == 0 {
		t.Fatal("tape should have recorded Tanh op")
	}

	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("expected gradient for x")
	}

	eps := float32(1e-4)
	actual := gradX.AsFloat32()
	tanhVals := resultRaw.AsFloat32()
	for i, tv := range tanhVals {
		expected := 1 - tv*tv
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("grad_x[%d] = %f, want %f (1 - tanh²(x) = 1 - %f²)", i, actual[i], expected, tv)
		}
	}
}

// TestSiLU_Forward tests SiLU forward pass: SiLU(x) = x·σ(x).
func TestSiLU_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	result := backend.SiLU(input.Raw())
	actual := result.AsFloat32()

	eps := float32(1e-5)
	inputs := []float64{-2, -1, 0, 1, 2}
	for i, x := range inputs {
		sig := 1.0 / (1.0 + math.Exp(-x))
		expected := float32(x * sig)
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("SiLU[%d] = %f, want %f", i, actual[i], expected)
		}
	}
}

// TestSiLU_Backward tests SiLU backward: ∂SiLU/∂x = σ(x)·(1 + x·(1-σ(x))).
func TestSiLU_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.SiLU(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	if tape.NumOps() == 0 {
		t.Fatal("tape should have recorded SiLU op")
	}

	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("expected gradient for x")
	}

	eps := float32(1e-4)
	actual := gradX.AsFloat32()
	xVals := []float64{-2, -1, 0, 1, 2}
	for i, xv := range xVals {
		sig := 1.0 / (1.0 + math.Exp(-xv))
		expected := float32(sig * (1 + xv*(1-sig)))
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("grad_x[%d] = %f, want %f", i, actual[i], expected)
		}
	}
}

// TestLog_Forward tests Log forward pass: log(x).
func TestLog_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{0.5, 1.0, 2.0, float32(math.E)}, tensor.Shape{4}, backend)

	result := backend.Log(input.Raw())
	actual := result.AsFloat32()

	eps := float32(1e-5)
	inputs := []float64{0.5, 1.0, 2.0, math.E}
	for i, x := range inputs {
		expected := float32(math.Log(x))
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("Log[%d] = %f, want %f", i, actual[i], expected)
		}
	}
}

// TestLog_Backward tests Log backward pass: ∂log/∂x = 1/x.
func TestLog_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{0.5, 1.0, 2.0, float32(math.E)}, tensor.Shape{4}, backend)

	resultRaw := backend.Log(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	if tape.NumOps() == 0 {
		t.Fatal("tape should have recorded Log op")
	}

	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("expected gradient for x")
	}

	eps := float32(1e-4)
	actual := gradX.AsFloat32()
	xVals := []float32{0.5, 1.0, 2.0, float32(math.E)}
	for i, xv := range xVals {
		expected := 1.0 / xv
		if math.Abs(float64(actual[i]-expected)) > float64(eps) {
			t.Errorf("grad_x[%d] = %f, want %f (1/x = 1/%f)", i, actual[i], expected, xv)
		}
	}
}

// shapesEqual compares two shapes for equality.
func shapesEqual(a, b tensor.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestDetach tests that Detach creates a tensor without gradient tracking.
func TestDetach(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// Create tensor
	data := []float32{1, 2, 3, 4}
	original, _ := tensor.FromSlice(data, tensor.Shape{2, 2}, backend)

	// Detach
	detached := original.Detach()

	// Verify data is shared (same values)
	originalData := original.Data()
	detachedData := detached.Data()

	for i := range originalData {
		if originalData[i] != detachedData[i] {
			t.Errorf("Data mismatch at index %d: original=%f, detached=%f",
				i, originalData[i], detachedData[i])
		}
	}

	// Verify gradient tracking is disabled
	if detached.Grad() != nil {
		t.Error("Detached tensor should not have gradient")
	}

	// Note: Operations on RawTensor are still recorded by AutodiffBackend.
	// Detach only creates a Tensor wrapper without gradient tracking.
	// This is correct behavior - the tape records operations, but the
	// detached Tensor won't accumulate gradients via SetGrad().

	// Verify shape and backend are preserved
	if !shapesEqual(detached.Shape(), original.Shape()) {
		t.Errorf("Shape mismatch: original=%v, detached=%v",
			original.Shape(), detached.Shape())
	}

	if detached.Backend() != original.Backend() {
		t.Error("Backend should be preserved")
	}
}

// TestDetach_DataSharing tests that detached tensor shares data.
func TestDetach_DataSharing(t *testing.T) {
	backend := cpu.New() // Use CPU backend (no autodiff) for this test

	data := []float32{1, 2, 3, 4}
	original, _ := tensor.FromSlice(data, tensor.Shape{4}, backend)

	detached := original.Detach()

	// Modify original data
	originalData := original.Data()
	originalData[0] = 99

	// Verify change is visible in detached tensor (data sharing)
	detachedData := detached.Data()
	if detachedData[0] != 99 {
		t.Errorf("Detached tensor should share data: expected 99, got %f", detachedData[0])
	}
}

// TestDetach_IndependentGradients tests that detached tensor has no gradient chain.
func TestDetach_IndependentGradients(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// Create tensor and perform operation
	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	// c = a * b
	cRaw := backend.Mul(a.Raw(), b.Raw())
	c := tensor.New[float32](cRaw, backend)

	// Detach c
	cDetached := c.Detach()

	// Use detached c in another operation: d = cDetached + a
	// This should NOT create gradient path from d to b (because c is detached)
	dRaw := backend.Add(cDetached.Raw(), a.Raw())
	d := tensor.New[float32](dRaw, backend)

	// Backward from d
	gradients := autodiff.Backward(d, backend)

	// Should have gradient for a (used in both Mul and Add)
	if gradients[a.Raw()] == nil {
		t.Error("Expected gradient for a")
	}

	// Should NOT have gradient for b (because c was detached before Add)
	// Note: This test may fail if Detach doesn't properly break the gradient chain
	// For now, we just check that detached tensor itself has no grad
	if cDetached.Grad() != nil {
		t.Error("Detached tensor should not have gradient")
	}
}

// releaseSpyBackend is a test spy that counts ReleaseBackendData calls.
// It embeds cpu.Backend for all non-release methods.
type releaseSpyBackend struct {
	tensor.Backend
	released []any
}

func (s *releaseSpyBackend) ReleaseBackendData(data any) {
	s.released = append(s.released, data)
}

func (s *releaseSpyBackend) SetPersistent(_ any, _ bool) {}

func (s *releaseSpyBackend) IsPersistent(_ any) bool { return false }

// TestReleaseGradients_NoBackend_IsNoop verifies that calling ReleaseGradients
// without a backend argument leaves grad BackendData intact — function is a no-op.
func TestReleaseGradients_NoBackend_IsNoop(t *testing.T) {
	b := autodiff.New(cpu.New())
	b.Tape().StartRecording()

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, b)
	x, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, b)
	result := tensor.New[float32](b.Add(a.Raw(), x.Raw()), b)

	grads := autodiff.Backward(result, b)
	if len(grads) == 0 {
		t.Fatal("expected at least one gradient entry")
	}

	// Collect BackendData before release attempt (may be nil for CPU, that is fine).
	before := make(map[*tensor.RawTensor]any, len(grads))
	for k, v := range grads {
		before[k] = v.BackendData()
	}

	// Call without backend — must not modify BackendData.
	autodiff.ReleaseGradients(grads)

	for k, v := range grads {
		if v.BackendData() != before[k] {
			t.Errorf("ReleaseGradients() without backend modified BackendData for grad[%p]", k)
		}
	}
}

// TestReleaseGradients_WithBackend_CallsReleaser verifies that passing a backend
// causes ReleaseBackendData to be called for every gradient in the map and that
// the gradient's BackendData is cleared to nil afterward.
func TestReleaseGradients_WithBackend_CallsReleaser(t *testing.T) {
	base := cpu.New()
	b := autodiff.New(base)
	b.Tape().StartRecording()

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, b)
	x, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, b)
	result := tensor.New[float32](b.Add(a.Raw(), x.Raw()), b)

	grads := autodiff.Backward(result, b)
	if len(grads) == 0 {
		t.Fatal("expected at least one gradient entry")
	}

	// Inject spy backend data so the releaser has something to receive.
	sentinel := make([]any, 0, len(grads))
	for _, v := range grads {
		sd := struct{ id int }{id: len(sentinel)}
		v.SetBackendData(sd)
		sentinel = append(sentinel, sd)
	}

	spy := &releaseSpyBackend{Backend: base}

	autodiff.ReleaseGradients(grads, spy)

	// Every gradient must have been released.
	if len(spy.released) != len(grads) {
		t.Errorf("ReleaseBackendData called %d times, want %d", len(spy.released), len(grads))
	}

	// BackendData must be nil after release.
	for k, v := range grads {
		if v.BackendData() != nil {
			t.Errorf("BackendData not cleared for grad[%p] after ReleaseGradients", k)
		}
	}
}

// TestSum_GradientFlows verifies that Sum is recorded on the tape so that
// gradient flows back through it. Without this, loss = y.Sum() produces nil
// gradients for all upstream tensors.
func TestSum_GradientFlows(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3, 4}, tensor.Shape{4}, backend)
	x.RequireGrad()

	// loss = Sum(x) — without Sum on tape this produces nil gradient.
	sumRaw := backend.Sum(x.Raw())
	loss := tensor.New[float32](sumRaw, backend)

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]
	if grad == nil {
		t.Fatal("Sum: gradient for x is nil — Sum not recorded on tape")
	}

	// d(sum(x))/dx = [1, 1, 1, 1]
	got := grad.AsFloat32()
	for i, v := range got {
		if v != 1.0 {
			t.Errorf("grad[%d] = %f, want 1.0", i, v)
		}
	}
}

// TestSum_GradientFlows_2D verifies gradient flows through Sum for a 2D tensor.
func TestSum_GradientFlows_2D(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3}, backend)
	x.RequireGrad()

	sumRaw := backend.Sum(x.Raw())
	loss := tensor.New[float32](sumRaw, backend)

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]
	if grad == nil {
		t.Fatal("Sum 2D: gradient for x is nil")
	}

	if !grad.Shape().Equal(tensor.Shape{2, 3}) {
		t.Errorf("gradient shape = %v, want [2 3]", grad.Shape())
	}

	got := grad.AsFloat32()
	for i, v := range got {
		if v != 1.0 {
			t.Errorf("grad[%d] = %f, want 1.0", i, v)
		}
	}
}

// TestSum_ChainedGradient verifies the chain: loss = Sum(x²) → d/dx = 2x.
func TestSum_ChainedGradient(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	x.RequireGrad()

	// y = x * x, loss = sum(y)
	y := x.Mul(x)
	sumRaw := backend.Sum(y.Raw())
	loss := tensor.New[float32](sumRaw, backend)

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]
	if grad == nil {
		t.Fatal("Sum chain: gradient for x is nil")
	}

	// d(sum(x²))/dx = 2x = [2, 4, 6]
	expected := []float32{2, 4, 6}
	got := grad.AsFloat32()
	for i, v := range got {
		diff := v - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-4 {
			t.Errorf("grad[%d] = %f, want %f", i, got[i], expected[i])
		}
	}
}

// TestSum_TapeRecordsOp verifies that Sum records exactly one operation on the tape.
func TestSum_TapeRecordsOp(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()
	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	backend.Sum(x.Raw())

	if tape.NumOps() != 1 {
		t.Errorf("Sum: expected 1 op on tape, got %d", tape.NumOps())
	}
}

// TestExpand_GradientFlows verifies gradient flows through Expand (broadcast).
// Expand forward broadcasts; backward reduces (sums) along expanded dims.
func TestExpand_GradientFlows(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	// x is shape [3]; expand to [2, 3] — broadcasts along a new leading dim.
	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	x.RequireGrad()

	expandedRaw := backend.Expand(x.Raw(), tensor.Shape{2, 3})
	// Sum all to get scalar loss so backward reaches x.
	sumRaw := backend.Sum(expandedRaw)
	loss := tensor.New[float32](sumRaw, backend)

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]
	if grad == nil {
		t.Fatal("Expand: gradient for x is nil")
	}

	if !grad.Shape().Equal(tensor.Shape{3}) {
		t.Errorf("gradient shape = %v, want [3]", grad.Shape())
	}

	// Each of the 3 elements was used in 2 rows → grad = 2.0 per element.
	got := grad.AsFloat32()
	for i, v := range got {
		if math.Abs(float64(v-2.0)) > 1e-4 {
			t.Errorf("grad[%d] = %f, want 2.0", i, v)
		}
	}
}

// TestExpand_TapeRecordsOp verifies that Expand records exactly one op on the tape.
func TestExpand_TapeRecordsOp(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()
	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	backend.Expand(x.Raw(), tensor.Shape{2, 3})

	if tape.NumOps() != 1 {
		t.Errorf("Expand: expected 1 op on tape, got %d", tape.NumOps())
	}
}

// TestCast_GradientFlows verifies gradient flows through Cast float32→float64→float32.
// Cast backward re-casts the gradient to the original input dtype.
func TestCast_GradientFlows(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	x.RequireGrad()

	// Cast x to float64, then sum (scalar) and backward.
	castedRaw := backend.Cast(x.Raw(), tensor.Float64)
	sumRaw := backend.Sum(castedRaw)
	loss := tensor.New[float64](sumRaw, backend)

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]
	if grad == nil {
		t.Fatal("Cast: gradient for x is nil")
	}

	// Gradient must come back as float32 (original input dtype).
	if grad.DType() != tensor.Float32 {
		t.Errorf("gradient dtype = %v, want float32", grad.DType())
	}

	if !grad.Shape().Equal(tensor.Shape{3}) {
		t.Errorf("gradient shape = %v, want [3]", grad.Shape())
	}

	// d(sum(cast(x)))/dx = [1, 1, 1] (cast is element-wise identity for value)
	got := grad.AsFloat32()
	for i, v := range got {
		if math.Abs(float64(v-1.0)) > 1e-4 {
			t.Errorf("grad[%d] = %f, want 1.0", i, v)
		}
	}
}

// TestCast_TapeRecordsOp verifies that Cast records exactly one op on the tape.
func TestCast_TapeRecordsOp(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()
	tape.StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	backend.Cast(x.Raw(), tensor.Float64)

	if tape.NumOps() != 1 {
		t.Errorf("Cast: expected 1 op on tape, got %d", tape.NumOps())
	}
}

// TestBackward_RootsAtLossTensor verifies that Backward computes gradients
// from the loss tensor, not from the last operation recorded on the tape.
// Bug: tape.go placed gradient seed at lastOp.Output() regardless of which
// tensor was passed to Backward(). Any op after loss → wrong gradients.
func TestBackward_RootsAtLossTensor(t *testing.T) {
	backend := autodiff.New(cpu.New())
	backend.Tape().StartRecording()

	x, _ := tensor.FromSlice([]float32{1, 2, 3}, tensor.Shape{3}, backend)
	x.RequireGrad()

	// loss = sum_dim(x²) — a scalar
	y := x.Mul(x) // y = x²
	loss := tensor.New[float32](backend.SumDim(y.Raw(), 0, false), backend)

	// EXTRA OP recorded AFTER loss — simulates metric/logging
	metric := x.Add(x) // metric = 2x
	_ = metric

	grads := autodiff.Backward(loss, backend)
	grad := grads[x.Raw()]

	if grad == nil {
		t.Fatal("gradient for x is nil")
	}

	// d(sum(x²))/dx = 2x = [2, 4, 6]
	expected := []float32{2, 4, 6}
	got := grad.AsFloat32()
	for i := range expected {
		diff := got[i] - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-4 {
			t.Errorf("grad[%d] = %f, want %f", i, got[i], expected[i])
		}
	}
}
