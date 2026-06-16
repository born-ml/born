package cpu

import (
	"math/rand"
	"testing"
)

// createRandomInt32Slices returns two 1024-element slices filled with
// random int32 values, suitable for benchmarking element-wise ops.
func createRandomInt32Slices() ([]int32, []int32) {
	aSlice := make([]int32, 1024)
	bSlice := make([]int32, 1024)

	rng := rand.New(rand.NewSource(0))
	for i := range aSlice {
		aSlice[i] = int32(rng.Int())
	}
	for i := range bSlice {
		bSlice[i] = int32(rng.Int())
	}
	return aSlice, bSlice
}

// BenchmarkAddInplaceI32_Scalar benchmarks a[i] += b[i] using the scalar fallback.
func BenchmarkAddInplaceI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()

	saved := simdAddInplaceInt32
	simdAddInplaceInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		addInplaceInt32(aSlice, bSlice)
	}
	simdAddInplaceInt32 = saved
}

// BenchmarkAddInplaceI32_SIMD benchmarks a[i] += b[i] using the SIMD implementation.
func BenchmarkAddInplaceI32_SIMD(b *testing.B) {
	if simdAddInplaceInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		addInplaceInt32(aSlice, bSlice)
	}
}

// BenchmarkSubInplaceI32_Scalar benchmarks a[i] -= b[i] using the scalar fallback.
func BenchmarkSubInplaceI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()

	saved := simdSubInplaceInt32
	simdSubInplaceInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		subInplaceInt32(aSlice, bSlice)
	}
	simdSubInplaceInt32 = saved
}

// BenchmarkSubInplaceI32_SIMD benchmarks a[i] -= b[i] using the SIMD implementation.
func BenchmarkSubInplaceI32_SIMD(b *testing.B) {
	if simdSubInplaceInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		subInplaceInt32(aSlice, bSlice)
	}
}

// BenchmarkMulInplaceI32_Scalar benchmarks a[i] *= b[i] using the scalar fallback.
func BenchmarkMulInplaceI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()

	saved := simdMulInplaceInt32
	simdMulInplaceInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		mulInplaceInt32(aSlice, bSlice)
	}
	simdMulInplaceInt32 = saved
}

// BenchmarkMulInplaceI32_SIMD benchmarks a[i] *= b[i] using the SIMD implementation.
func BenchmarkMulInplaceI32_SIMD(b *testing.B) {
	if simdMulInplaceInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		mulInplaceInt32(aSlice, bSlice)
	}
}

// BenchmarkAddVectorizedI32_Scalar benchmarks dst[i] = a[i] + b[i] using the scalar fallback.
func BenchmarkAddVectorizedI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	saved := simdAddVectorizedInt32
	simdAddVectorizedInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		addVectorizedInt32(dst, aSlice, bSlice)
	}
	simdAddVectorizedInt32 = saved
}

// BenchmarkAddVectorizedI32_SIMD benchmarks dst[i] = a[i] + b[i] using the SIMD implementation.
func BenchmarkAddVectorizedI32_SIMD(b *testing.B) {
	if simdAddVectorizedInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedInt32(dst, aSlice, bSlice)
	}
}

// BenchmarkSubVectorizedI32_Scalar benchmarks dst[i] = a[i] - b[i] using the scalar fallback.
func BenchmarkSubVectorizedI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	saved := simdSubVectorizedInt32
	simdSubVectorizedInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		subVectorizedInt32(dst, aSlice, bSlice)
	}
	simdSubVectorizedInt32 = saved
}

// BenchmarkSubVectorizedI32_SIMD benchmarks dst[i] = a[i] - b[i] using the SIMD implementation.
func BenchmarkSubVectorizedI32_SIMD(b *testing.B) {
	if simdSubVectorizedInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedInt32(dst, aSlice, bSlice)
	}
}

// BenchmarkMulVectorizedI32_Scalar benchmarks dst[i] = a[i] * b[i] using the scalar fallback.
func BenchmarkMulVectorizedI32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	saved := simdMulVectorizedInt32
	simdMulVectorizedInt32 = nil
	b.ResetTimer()
	for b.Loop() {
		mulVectorizedInt32(dst, aSlice, bSlice)
	}
	simdMulVectorizedInt32 = saved
}

// BenchmarkMulVectorizedI32_SIMD benchmarks dst[i] = a[i] * b[i] using the SIMD implementation.
func BenchmarkMulVectorizedI32_SIMD(b *testing.B) {
	if simdMulVectorizedInt32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedInt32(dst, aSlice, bSlice)
	}
}
