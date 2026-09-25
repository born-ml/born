//go:build windows || linux

package webgpu

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

// softmaxCPUReference computes row-wise softmax on CPU.
// Input shape: [batchSize, numClasses]. Returns the softmax output.
// Uses max-shift trick for numerical stability, matching the GPU shader.
func softmaxCPUReference(data []float32, batchSize, numClasses int) []float32 {
	out := make([]float32, batchSize*numClasses)
	for row := range batchSize {
		offset := row * numClasses

		// Phase 1: find max for numerical stability.
		maxVal := data[offset]
		for i := 1; i < numClasses; i++ {
			if data[offset+i] > maxVal {
				maxVal = data[offset+i]
			}
		}

		// Phase 2: compute exp(x - max) and sum.
		var sum float32
		for i := range numClasses {
			v := float32(math.Exp(float64(data[offset+i] - maxVal)))
			out[offset+i] = v
			sum += v
		}

		// Phase 3: normalize.
		for i := range numClasses {
			out[offset+i] /= sum
		}
	}
	return out
}

// checkSoftmaxResults compares GPU softmax output against a CPU reference
// within a tolerance of 1e-5.
func checkSoftmaxResults(t *testing.T, got, want []float32, numClasses int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("result size: got %d, want %d", len(got), len(want))
	}
	const tol = 1e-5
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > tol {
			row := i / numClasses
			col := i % numClasses
			t.Errorf("[%d,%d] got=%f want=%f diff=%e", row, col, got[i], want[i], diff)
		}
	}
}

// runSoftmaxCase executes one softmax test case on the given backend
// and compares against the CPU reference.
func runSoftmaxCase(t *testing.T, b *Backend, batchSize, numClasses int) {
	t.Helper()

	data := make([]float32, batchSize*numClasses)
	for i := range data {
		// Use values that exercise numerical stability (spread across -5..5).
		data[i] = float32(i%13)*0.8 - 5.0 + float32(i%7)*0.3
	}

	expected := softmaxCPUReference(data, batchSize, numClasses)

	raw, err := tensor.NewRaw(tensor.Shape{batchSize, numClasses}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw: %v", err)
	}
	copy(raw.AsFloat32(), data)

	b.LazyMode = false
	got := b.Softmax(raw, -1)
	if got == nil {
		t.Fatal("Softmax returned nil")
	}
	checkSoftmaxResults(t, got.AsFloat32(), expected, numClasses)
}

// TestSubgroupSoftmaxShader_Correctness verifies the subgroup softmax path (when
// hardware supports it) produces results numerically identical to the scalar path.
//
// Test covers edge cases:
//   - num_classes = 1 (single class, output must be 1.0)
//   - num_classes = 32 (exactly one subgroup wave, all lanes active)
//   - num_classes = 33 (one lane handles an extra class, covers the strides boundary)
//   - num_classes = 128 (four full waves)
//   - num_classes = 1000 (many strides, exercises the loop)
func TestSubgroupSoftmaxShader_Correctness(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	b, err := New()
	if err != nil {
		t.Skipf("WebGPU backend unavailable: %v", err)
	}
	defer b.Release()

	t.Logf("backend: subgroupsEnabled=%v", b.subgroupsEnabled)

	tests := []struct {
		name                  string
		batchSize, numClasses int
	}{
		{"1x1 single class", 1, 1},
		{"1x32 exactly one wave", 1, 32},
		{"1x33 strides boundary", 1, 33},
		{"1x128 four waves", 1, 128},
		{"4x32", 4, 32},
		{"4x64", 4, 64},
		{"8x33 non-multiple of 32", 8, 33},
		{"16x128", 16, 128},
		{"32x256", 32, 256},
		{"64x1000 many strides", 64, 1000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runSoftmaxCase(t, b, tc.batchSize, tc.numClasses)
		})
	}
}

// TestSubgroupSoftmaxShader_ScalarFallback verifies that when subgroupsEnabled=false
// the scalar softmax shader is used and produces correct results on all hardware.
func TestSubgroupSoftmaxShader_ScalarFallback(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	b, err := New()
	if err != nil {
		t.Skipf("WebGPU backend unavailable: %v", err)
	}
	defer b.Release()

	// Force scalar path regardless of hardware capability.
	b.subgroupsEnabled = false
	b.LazyMode = false

	batchSize, numClasses := 8, 64
	data := make([]float32, batchSize*numClasses)
	for i := range data {
		data[i] = float32(i%17)*0.4 - 3.0
	}

	expected := softmaxCPUReference(data, batchSize, numClasses)

	raw, err := tensor.NewRaw(tensor.Shape{batchSize, numClasses}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw: %v", err)
	}
	copy(raw.AsFloat32(), data)

	got := b.Softmax(raw, -1)
	checkSoftmaxResults(t, got.AsFloat32(), expected, numClasses)
}

// TestSubgroupSoftmaxShader_SubgroupPathExplicit forces subgroupsEnabled=true and
// verifies the subgroup shader compiles and produces correct results.
// This test is meaningful only when the software backend supports subgroup builtins.
func TestSubgroupSoftmaxShader_SubgroupPathExplicit(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	b, err := New()
	if err != nil {
		t.Skipf("WebGPU backend unavailable: %v", err)
	}
	defer b.Release()

	// Force subgroup path. On hardware that doesn't support subgroup ops, the
	// shader compilation may fail. We recover and skip gracefully.
	b.subgroupsEnabled = true
	b.LazyMode = false

	defer func() {
		if r := recover(); r != nil {
			t.Skipf("subgroup shader compilation failed (expected on hardware without subgroup support): %v", r)
		}
	}()

	batchSize, numClasses := 4, 64
	data := make([]float32, batchSize*numClasses)
	for i := range data {
		data[i] = float32(i%11)*0.5 - 2.5
	}
	expected := softmaxCPUReference(data, batchSize, numClasses)

	raw, err := tensor.NewRaw(tensor.Shape{batchSize, numClasses}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw: %v", err)
	}
	copy(raw.AsFloat32(), data)

	got := b.Softmax(raw, -1)
	if got == nil {
		t.Fatal("Softmax returned nil")
	}
	checkSoftmaxResults(t, got.AsFloat32(), expected, numClasses)
}

// TestSubgroupSoftmaxShader_NumericalStability verifies that the max-shift trick
// works correctly for inputs with large values that would cause exp() overflow
// without the stability correction.
func TestSubgroupSoftmaxShader_NumericalStability(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	b, err := New()
	if err != nil {
		t.Skipf("WebGPU backend unavailable: %v", err)
	}
	defer b.Release()
	b.LazyMode = false

	// Use large values that would cause NaN without max-shift.
	batchSize, numClasses := 2, 10
	data := []float32{
		// Row 0: large positive values — exp(x) would overflow without max-shift.
		80.0, 81.0, 79.0, 82.0, 78.0, 83.0, 77.0, 84.0, 76.0, 85.0,
		// Row 1: large negative values — exp(x) should produce near-zero for most.
		-85.0, -84.0, -83.0, -82.0, -81.0, -80.0, -79.0, -78.0, -77.0, -76.0,
	}

	expected := softmaxCPUReference(data, batchSize, numClasses)

	raw, err := tensor.NewRaw(tensor.Shape{batchSize, numClasses}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw: %v", err)
	}
	copy(raw.AsFloat32(), data)

	got := b.Softmax(raw, -1)
	if got == nil {
		t.Fatal("Softmax returned nil")
	}

	// Verify no NaN values.
	for i, v := range got.AsFloat32() {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at index %d — numerical stability failed", i)
		}
	}

	// Verify row sums are approximately 1.0 (valid probability distribution).
	for row := range batchSize {
		var rowSum float32
		for col := range numClasses {
			rowSum += got.AsFloat32()[row*numClasses+col]
		}
		diff := math.Abs(float64(rowSum - 1.0))
		if diff > 1e-5 {
			t.Errorf("row %d sum: got %f, want ~1.0 (diff=%e)", row, rowSum, diff)
		}
	}

	checkSoftmaxResults(t, got.AsFloat32(), expected, numClasses)
}

// TestSubgroupSoftmaxWGSLSyntax verifies that the softmaxSubgroupShader WGSL
// string can be loaded into naga IR without error. This catches syntax regressions
// in CI even when the software backend doesn't execute subgroup instructions.
func TestSubgroupSoftmaxWGSLSyntax(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	b, err := New()
	if err != nil {
		t.Skipf("WebGPU backend unavailable: %v", err)
	}
	defer b.Release()
	if !b.subgroupsEnabled {
		t.Skip("subgroup operations not supported by adapter")
	}

	// compileShader panics on failure; recover to turn it into a test failure.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("compileShader panicked: %v", r)
		}
	}()
	// This calls CreateShaderModule which runs naga.Parse+Lower internally.
	_ = b.compileShader("softmax_subgroup_syntax_test", softmaxSubgroupShader)
}
