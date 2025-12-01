package cpu

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

// TestBatchMatMul_3D_Basic tests basic 3D batch matmul.
func TestBatchMatMul_3D_Basic(t *testing.T) {
	backend := New()

	// Create input tensors: [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
	a, err := tensor.NewRaw(tensor.Shape{2, 3, 4}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.NewRaw(tensor.Shape{2, 4, 5}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}

	// Fill with simple data
	aData := a.AsFloat32()
	bData := b.AsFloat32()
	for i := range aData {
		aData[i] = float32(i + 1)
	}
	for i := range bData {
		bData[i] = float32(i + 1)
	}

	// Execute
	result := backend.BatchMatMul(a, b)

	// Verify shape
	expected := tensor.Shape{2, 3, 5}
	if !shapeEqual(result.Shape(), expected) {
		t.Errorf("Expected shape %v, got %v", expected, result.Shape())
	}

	// Verify dtype
	if result.DType() != tensor.Float32 {
		t.Errorf("Expected dtype Float32, got %v", result.DType())
	}

	// Verify result is not zero (computation happened)
	resultData := result.AsFloat32()
	if resultData[0] == 0 {
		t.Error("Expected non-zero result")
	}
}

// TestBatchMatMul_4D_MultiHead tests 4D batch matmul (multi-head attention scenario).
func TestBatchMatMul_4D_MultiHead(t *testing.T) {
	backend := New()

	// Simulates multi-head attention: [B=2, H=4, S=8, D=16] @ [B=2, H=4, D=16, S'=8]
	// Result: [B=2, H=4, S=8, S'=8]
	a, err := tensor.NewRaw(tensor.Shape{2, 4, 8, 16}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.NewRaw(tensor.Shape{2, 4, 16, 8}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}

	// Fill with simple data
	aData := a.AsFloat32()
	bData := b.AsFloat32()
	for i := range aData {
		aData[i] = 1.0
	}
	for i := range bData {
		bData[i] = 1.0
	}

	// Execute
	result := backend.BatchMatMul(a, b)

	// Verify shape
	expected := tensor.Shape{2, 4, 8, 8}
	if !shapeEqual(result.Shape(), expected) {
		t.Errorf("Expected shape %v, got %v", expected, result.Shape())
	}

	// For all 1s input, each output element should be sum of K elements = 16
	resultData := result.AsFloat32()
	expectedValue := float32(16.0)
	for i, val := range resultData {
		if math.Abs(float64(val-expectedValue)) > 1e-5 {
			t.Errorf("Element %d: expected %f, got %f", i, expectedValue, val)
		}
	}
}

// TestBatchMatMul_ShapeMismatch tests dimension validation.
func TestBatchMatMul_ShapeMismatch(t *testing.T) {
	backend := New()

	// 2D input should panic
	a, _ := tensor.NewRaw(tensor.Shape{3, 4}, tensor.Float32, tensor.CPU)
	b, _ := tensor.NewRaw(tensor.Shape{4, 5}, tensor.Float32, tensor.CPU)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for 2D input")
		}
	}()

	backend.BatchMatMul(a, b)
}

// TestBatchMatMul_InnerDimMismatch tests matrix dimension validation.
func TestBatchMatMul_InnerDimMismatch(t *testing.T) {
	backend := New()

	// [2, 3, 4] @ [2, 5, 6] should panic (K mismatch)
	a, _ := tensor.NewRaw(tensor.Shape{2, 3, 4}, tensor.Float32, tensor.CPU)
	b, _ := tensor.NewRaw(tensor.Shape{2, 5, 6}, tensor.Float32, tensor.CPU)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for inner dimension mismatch")
		}
	}()

	backend.BatchMatMul(a, b)
}

// TestBatchMatMul_BatchDimMismatch tests batch dimension validation.
func TestBatchMatMul_BatchDimMismatch(t *testing.T) {
	backend := New()

	// [2, 3, 4] @ [3, 4, 5] should panic (batch dim mismatch)
	a, _ := tensor.NewRaw(tensor.Shape{2, 3, 4}, tensor.Float32, tensor.CPU)
	b, _ := tensor.NewRaw(tensor.Shape{3, 4, 5}, tensor.Float32, tensor.CPU)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for batch dimension mismatch")
		}
	}()

	backend.BatchMatMul(a, b)
}

// TestBatchMatMul_vs_LoopWorkaround tests numerical equivalence with manual loop.
func TestBatchMatMul_vs_LoopWorkaround(t *testing.T) {
	backend := New()

	// Create 3D tensors
	batchSize := 2
	m, k, n := 3, 4, 5

	a, _ := tensor.NewRaw(tensor.Shape{batchSize, m, k}, tensor.Float32, tensor.CPU)
	b, _ := tensor.NewRaw(tensor.Shape{batchSize, k, n}, tensor.Float32, tensor.CPU)

	// Fill with incrementing values
	aData := a.AsFloat32()
	bData := b.AsFloat32()
	for i := range aData {
		aData[i] = float32(i + 1)
	}
	for i := range bData {
		bData[i] = float32(i + 1)
	}

	// BatchMatMul
	result := backend.BatchMatMul(a, b)
	resultData := result.AsFloat32()

	// Manual loop workaround (reference implementation)
	expected, _ := tensor.NewRaw(tensor.Shape{batchSize, m, n}, tensor.Float32, tensor.CPU)
	expectedData := expected.AsFloat32()

	for batch := 0; batch < batchSize; batch++ {
		aOffset := batch * m * k
		bOffset := batch * k * n
		cOffset := batch * m * n

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for kIdx := 0; kIdx < k; kIdx++ {
					sum += aData[aOffset+i*k+kIdx] * bData[bOffset+kIdx*n+j]
				}
				expectedData[cOffset+i*n+j] = sum
			}
		}
	}

	// Compare results
	for i := range resultData {
		if math.Abs(float64(resultData[i]-expectedData[i])) > 1e-5 {
			t.Errorf("Element %d: expected %f, got %f", i, expectedData[i], resultData[i])
		}
	}
}

// TestBatchMatMul_Float64 tests float64 support.
func TestBatchMatMul_Float64(t *testing.T) {
	backend := New()

	a, _ := tensor.NewRaw(tensor.Shape{2, 3, 4}, tensor.Float64, tensor.CPU)
	b, _ := tensor.NewRaw(tensor.Shape{2, 4, 5}, tensor.Float64, tensor.CPU)

	aData := a.AsFloat64()
	bData := b.AsFloat64()
	for i := range aData {
		aData[i] = float64(i + 1)
	}
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	result := backend.BatchMatMul(a, b)

	// Verify shape
	expected := tensor.Shape{2, 3, 5}
	if !shapeEqual(result.Shape(), expected) {
		t.Errorf("Expected shape %v, got %v", expected, result.Shape())
	}

	// Verify dtype
	if result.DType() != tensor.Float64 {
		t.Errorf("Expected dtype Float64, got %v", result.DType())
	}

	// Verify result is not zero
	resultData := result.AsFloat64()
	if resultData[0] == 0 {
		t.Error("Expected non-zero result")
	}
}

// Helper function to compare shapes.
func shapeEqual(a, b tensor.Shape) bool {
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
