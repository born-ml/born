package cpu

import (
	"math/rand"
	"testing"
)

// createRandomFloat64Slices returns two 1024-element slices filled with
// random float64 values in [-1, 1), suitable for benchmarking element-wise ops.
func createRandomFloat64Slices() ([]float64, []float64) {
	aSlice := make([]float64, 1024)
	bSlice := make([]float64, 1024)
	rng := rand.New(rand.NewSource(0))
	for i := range aSlice {
		aSlice[i] = rng.Float64()*2 - 1
	}
	for i := range bSlice {
		bSlice[i] = rng.Float64()*2 - 1
	}
	return aSlice, bSlice
}

// BenchmarkAddInplaceF64_Scalar benchmarks a[i] += b[i] using the scalar fallback.
func BenchmarkAddInplaceF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()

	saved := simdAddInplaceFloat64
	simdAddInplaceFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		addInplaceFloat64(aSlice, bSlice)
	}
	simdAddInplaceFloat64 = saved
}

// BenchmarkAddInplaceF64_SIMD benchmarks a[i] += b[i] using the SIMD implementation.
func BenchmarkAddInplaceF64_SIMD(b *testing.B) {
	if simdAddInplaceFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		addInplaceFloat64(aSlice, bSlice)
	}
}

// BenchmarkSubInplaceF64_Scalar benchmarks a[i] -= b[i] using the scalar fallback.
func BenchmarkSubInplaceF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()

	saved := simdSubInplaceFloat64
	simdSubInplaceFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		subInplaceFloat64(aSlice, bSlice)
	}
	simdSubInplaceFloat64 = saved
}

// BenchmarkSubInplaceF64_SIMD benchmarks a[i] -= b[i] using the SIMD implementation.
func BenchmarkSubInplaceF64_SIMD(b *testing.B) {
	if simdSubInplaceFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		subInplaceFloat64(aSlice, bSlice)
	}
}

// BenchmarkMulInplaceF64_Scalar benchmarks a[i] *= b[i] using the scalar fallback.
func BenchmarkMulInplaceF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()

	saved := simdMulInplaceFloat64
	simdMulInplaceFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		mulInplaceFloat64(aSlice, bSlice)
	}
	simdMulInplaceFloat64 = saved
}

// BenchmarkMulInplaceF64_SIMD benchmarks a[i] *= b[i] using the SIMD implementation.
func BenchmarkMulInplaceF64_SIMD(b *testing.B) {
	if simdMulInplaceFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		mulInplaceFloat64(aSlice, bSlice)
	}
}

// BenchmarkDivInplaceF64_Scalar benchmarks a[i] /= b[i] using the scalar fallback.
func BenchmarkDivInplaceF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()

	saved := simdDivInplaceFloat64
	simdDivInplaceFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		divInplaceFloat64(aSlice, bSlice)
	}
	simdDivInplaceFloat64 = saved
}

// BenchmarkDivInplaceF64_SIMD benchmarks a[i] /= b[i] using the SIMD implementation.
func BenchmarkDivInplaceF64_SIMD(b *testing.B) {
	if simdDivInplaceFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		divInplaceFloat64(aSlice, bSlice)
	}
}

// BenchmarkAddVectorizedF64_Scalar benchmarks dst[i] = a[i] + b[i] using the scalar fallback.
func BenchmarkAddVectorizedF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	saved := simdAddVectorizedFloat64
	simdAddVectorizedFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		addVectorizedFloat64(dst, aSlice, bSlice)
	}
	simdAddVectorizedFloat64 = saved
}

// BenchmarkAddVectorizedF64_SIMD benchmarks dst[i] = a[i] + b[i] using the SIMD implementation.
func BenchmarkAddVectorizedF64_SIMD(b *testing.B) {
	if simdAddVectorizedFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedFloat64(dst, aSlice, bSlice)
	}
}

// BenchmarkSubVectorizedF64_Scalar benchmarks dst[i] = a[i] - b[i] using the scalar fallback.
func BenchmarkSubVectorizedF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	saved := simdSubVectorizedFloat64
	simdSubVectorizedFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		subVectorizedFloat64(dst, aSlice, bSlice)
	}
	simdSubVectorizedFloat64 = saved
}

// BenchmarkSubVectorizedF64_SIMD benchmarks dst[i] = a[i] - b[i] using the SIMD implementation.
func BenchmarkSubVectorizedF64_SIMD(b *testing.B) {
	if simdSubVectorizedFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedFloat64(dst, aSlice, bSlice)
	}
}

// BenchmarkMulVectorizedF64_Scalar benchmarks dst[i] = a[i] * b[i] using the scalar fallback.
func BenchmarkMulVectorizedF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	saved := simdMulVectorizedFloat64
	simdMulVectorizedFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		mulVectorizedFloat64(dst, aSlice, bSlice)
	}
	simdMulVectorizedFloat64 = saved
}

// BenchmarkMulVectorizedF64_SIMD benchmarks dst[i] = a[i] * b[i] using the SIMD implementation.
func BenchmarkMulVectorizedF64_SIMD(b *testing.B) {
	if simdMulVectorizedFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedFloat64(dst, aSlice, bSlice)
	}
}

// BenchmarkDivVectorizedF64_Scalar benchmarks dst[i] = a[i] / b[i] using the scalar fallback.
func BenchmarkDivVectorizedF64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	saved := simdDivVectorizedFloat64
	simdDivVectorizedFloat64 = nil
	b.ResetTimer()
	for b.Loop() {
		divVectorizedFloat64(dst, aSlice, bSlice)
	}
	simdDivVectorizedFloat64 = saved
}

// BenchmarkDivVectorizedF64_SIMD benchmarks dst[i] = a[i] / b[i] using the SIMD implementation.
func BenchmarkDivVectorizedF64_SIMD(b *testing.B) {
	if simdDivVectorizedFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		divVectorizedFloat64(dst, aSlice, bSlice)
	}
}
