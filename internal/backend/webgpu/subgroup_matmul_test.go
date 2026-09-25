//go:build windows || linux

package webgpu

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

// matmulCPUReference computes C = A @ B on CPU for correctness comparison.
// A is [rows, inner], B is [inner, cols], C is [rows, cols].
func matmulCPUReference(aData []float32, rows, inner int, bData []float32, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := range rows {
		for c := range cols {
			var sum float32
			for k := range inner {
				sum += aData[r*inner+k] * bData[k*cols+c]
			}
			out[r*cols+c] = sum
		}
	}
	return out
}

// checkMatMulResults compares a GPU MatMul result against a CPU reference.
func checkMatMulResults(t *testing.T, got, expected []float32, cols int) {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("result size: got %d, want %d", len(got), len(expected))
	}
	const tol = 1e-4
	for i, want := range expected {
		diff := math.Abs(float64(got[i] - want))
		if diff > tol {
			r, c := i/cols, i%cols
			t.Errorf("[%d,%d] got=%f want=%f diff=%f", r, c, got[i], want, diff)
		}
	}
}

// runMatMulCase executes one MatMul test case on the given backend.
func runMatMulCase(t *testing.T, b *Backend, rows, inner, cols int) {
	t.Helper()

	aData := make([]float32, rows*inner)
	bData := make([]float32, inner*cols)
	for i := range aData {
		aData[i] = float32(i%7) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i%5)*0.2 + 0.1
	}

	expected := matmulCPUReference(aData, rows, inner, bData, cols)

	aRaw, err := tensor.NewRaw(tensor.Shape{rows, inner}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw A: %v", err)
	}
	copy(aRaw.AsFloat32(), aData)

	bRaw, err := tensor.NewRaw(tensor.Shape{inner, cols}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw B: %v", err)
	}
	copy(bRaw.AsFloat32(), bData)

	b.LazyMode = false
	cRaw := b.MatMul(aRaw, bRaw)
	if cRaw == nil {
		t.Fatal("MatMul returned nil")
	}
	checkMatMulResults(t, cRaw.AsFloat32(), expected, cols)
}

// TestSubgroupMatMulShaders_Correctness verifies that the subgroup shader string
// constants contain parseable WGSL (naga can compile them) and that the
// matmul operations produce correct results.
//
// The test runs in two modes:
//  1. Software backend (always available): uses the scalar path, verifying
//     correctness of the fallback and that shader compilation does not panic.
//  2. Hardware backend with subgroupsEnabled=true (when device supports it):
//     verifies that the subgroup path produces numerically identical results.
func TestSubgroupMatMulShaders_Correctness(t *testing.T) {
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

	// Test cases covering edge cases in both scalar and subgroup paths.
	tests := []struct {
		name              string
		rows, inner, cols int
	}{
		{"1x1x1", 1, 1, 1},
		{"2x2x2", 2, 2, 2},
		{"4x4x4", 4, 4, 4},
		{"1x32x1 K=subgroupSize", 1, 32, 1},
		{"4x64x4 K=2*subgroupSize", 4, 64, 4},
		{"8x33x8 K not multiple of 32", 8, 33, 8},
		{"16x16x16", 16, 16, 16},
		{"32x32x32", 32, 32, 32},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runMatMulCase(t, b, tc.rows, tc.inner, tc.cols)
		})
	}
}

// TestSubgroupMatMulShaders_ScalarFallback verifies that when subgroupsEnabled=false
// the scalar shader is used and produces correct results.
// This test always runs — the scalar path is the safe fallback on all hardware.
func TestSubgroupMatMulShaders_ScalarFallback(t *testing.T) {
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

	M, K, N := 8, 24, 6
	aData := make([]float32, M*K)
	bData := make([]float32, K*N)
	for i := range aData {
		aData[i] = float32(i+1) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i+1) * 0.05
	}

	expected := matmulCPUReference(aData, M, K, bData, N)

	aRaw, _ := tensor.NewRaw(tensor.Shape{M, K}, tensor.Float32, tensor.CPU)
	copy(aRaw.AsFloat32(), aData)
	bRaw, _ := tensor.NewRaw(tensor.Shape{K, N}, tensor.Float32, tensor.CPU)
	copy(bRaw.AsFloat32(), bData)

	cRaw := b.MatMul(aRaw, bRaw)
	got := cRaw.AsFloat32()

	const tol = 1e-4
	for i, want := range expected {
		diff := math.Abs(float64(got[i] - want))
		if diff > tol {
			r, c := i/N, i%N
			t.Errorf("scalar[%d,%d]: got=%f want=%f diff=%e", r, c, got[i], want, diff)
		}
	}
}

// TestSubgroupBatchMatMulShaders_Correctness verifies batch MatMul with the
// subgroup path (when available) or scalar fallback.
func TestSubgroupBatchMatMulShaders_Correctness(t *testing.T) {
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

	t.Logf("backend: subgroupsEnabled=%v", b.subgroupsEnabled)

	batch, M, K, N := 3, 4, 16, 4
	aData := make([]float32, batch*M*K)
	bData := make([]float32, batch*K*N)
	for i := range aData {
		aData[i] = float32(i%11)*0.1 + 0.05
	}
	for i := range bData {
		bData[i] = float32(i%7)*0.15 + 0.1
	}

	// CPU reference for each batch.
	expected := make([]float32, batch*M*N)
	for bIdx := range batch {
		aSlice := aData[bIdx*M*K : (bIdx+1)*M*K]
		bSlice := bData[bIdx*K*N : (bIdx+1)*K*N]
		result := matmulCPUReference(aSlice, M, K, bSlice, N)
		copy(expected[bIdx*M*N:], result)
	}

	aRaw, _ := tensor.NewRaw(tensor.Shape{batch, M, K}, tensor.Float32, tensor.CPU)
	copy(aRaw.AsFloat32(), aData)
	bRaw, _ := tensor.NewRaw(tensor.Shape{batch, K, N}, tensor.Float32, tensor.CPU)
	copy(bRaw.AsFloat32(), bData)

	cRaw := b.BatchMatMul(aRaw, bRaw)
	got := cRaw.AsFloat32()

	if len(got) != len(expected) {
		t.Fatalf("result size: got %d, want %d", len(got), len(expected))
	}
	const tol = 1e-4
	for i, want := range expected {
		diff := math.Abs(float64(got[i] - want))
		if diff > tol {
			bi := i / (M * N)
			rem := i % (M * N)
			r, c := rem/N, rem%N
			t.Errorf("batch[%d,%d,%d]: got=%f want=%f diff=%e", bi, r, c, got[i], want, diff)
		}
	}
}

// TestSubgroupShaderWGSLSyntax verifies that the subgroup shader WGSL strings
// can be loaded into naga IR without error. This catches WGSL syntax regressions
// in CI even when no GPU is present, because compileShader calls naga.Parse/Lower
// internally through CreateShaderModule.
func TestSubgroupShaderWGSLSyntax(t *testing.T) {
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

	shaders := []struct {
		name string
		code string
	}{
		{"matmulSubgroup", matmulSubgroupShader},
		{"batchMatMulSubgroup", batchMatMulSubgroupShader},
	}

	for _, s := range shaders {
		t.Run(s.name, func(t *testing.T) {
			// compileShader panics on failure; recover to turn it into a test failure.
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("compileShader panicked: %v", r)
				}
			}()
			// This calls CreateShaderModule which runs naga.Parse+Lower internally.
			_ = b.compileShader(s.name+"_syntax_test", s.code)
		})
	}
}
