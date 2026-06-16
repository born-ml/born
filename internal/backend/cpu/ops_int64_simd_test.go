package cpu

import (
	"math/rand"
	"testing"
)

func createRandomInt64Slices() ([]int64, []int64) {
	aSlice := make([]int64, 1024)
	bSlice := make([]int64, 1024)

	rng := rand.New(rand.NewSource(0))
	for i := range aSlice {
		aSlice[i] = int64(rng.Int())
	}
	for i := range bSlice {
		bSlice[i] = int64(rng.Int())
	}
	return aSlice, bSlice
}

// BenchmarkAddInplaceI64_Scalar benchmarks a[i] += b[i] using the scalar fallback.
func BenchmarkAddInplaceI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()

	saved := simdAddInplaceInt64
	simdAddInplaceInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		addInplaceInt64(aSlice, bSlice)
	}
	simdAddInplaceInt64 = saved
}

// BenchmarkAddInplaceI64_SIMD benchmarks a[i] += b[i] using the SIMD implementation.
func BenchmarkAddInplaceI64_SIMD(b *testing.B) {
	if simdAddInplaceInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()

	b.ResetTimer()
	for b.Loop() {
		addInplaceInt64(aSlice, bSlice)
	}
}

// BenchmarkSubInplaceI64_Scalar benchmarks a[i] -= b[i] using the scalar fallback.
func BenchmarkSubInplaceI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()

	saved := simdSubInplaceInt64
	simdSubInplaceInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		subInplaceInt64(aSlice, bSlice)
	}
	simdSubInplaceInt64 = saved
}

// BenchmarkSubInplaceI64_SIMD benchmarks a[i] -= b[i] using the SIMD implementation.
func BenchmarkSubInplaceI64_SIMD(b *testing.B) {
	if simdSubInplaceInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()

	b.ResetTimer()
	for b.Loop() {
		subInplaceInt64(aSlice, bSlice)
	}
}

// BenchmarkMulInplaceI64_Scalar benchmarks a[i] *= b[i] using the scalar fallback.
func BenchmarkMulInplaceI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()

	saved := simdMulInplaceInt64
	simdMulInplaceInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		mulInplaceInt64(aSlice, bSlice)
	}
	simdMulInplaceInt64 = saved
}

// BenchmarkMulInplaceI64_SIMD benchmarks a[i] *= b[i] using the SIMD implementation.
func BenchmarkMulInplaceI64_SIMD(b *testing.B) {
	if simdMulInplaceInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()

	b.ResetTimer()
	for b.Loop() {
		mulInplaceInt64(aSlice, bSlice)
	}
}

// BenchmarkAddVectorizedI64_Scalar benchmarks dst[i] = a[i] + b[i] using the scalar fallback.
func BenchmarkAddVectorizedI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	saved := simdAddVectorizedInt64
	simdAddVectorizedInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		addVectorizedInt64(dst, aSlice, bSlice)
	}
	simdAddVectorizedInt64 = saved
}

// BenchmarkAddVectorizedI64_SIMD benchmarks dst[i] = a[i] + b[i] using the SIMD implementation.
func BenchmarkAddVectorizedI64_SIMD(b *testing.B) {
	if simdAddVectorizedInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedInt64(dst, aSlice, bSlice)
	}
}

// BenchmarkSubVectorizedI64_Scalar benchmarks dst[i] = a[i] - b[i] using the scalar fallback.
func BenchmarkSubVectorizedI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	saved := simdSubVectorizedInt64
	simdSubVectorizedInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		subVectorizedInt64(dst, aSlice, bSlice)
	}
	simdSubVectorizedInt64 = saved
}

// BenchmarkSubVectorizedI64_SIMD benchmarks dst[i] = a[i] - b[i] using the SIMD implementation.
func BenchmarkSubVectorizedI64_SIMD(b *testing.B) {
	if simdSubVectorizedInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedInt64(dst, aSlice, bSlice)
	}
}

// BenchmarkMulVectorizedI64_Scalar benchmarks dst[i] = a[i] * b[i] using the scalar fallback.
func BenchmarkMulVectorizedI64_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	saved := simdMulVectorizedInt64
	simdMulVectorizedInt64 = nil
	b.ResetTimer()
	for b.Loop() {
		mulVectorizedInt64(dst, aSlice, bSlice)
	}
	simdMulVectorizedInt64 = saved
}

// BenchmarkMulVectorizedI64_SIMD benchmarks dst[i] = a[i] * b[i] using the SIMD implementation.
func BenchmarkMulVectorizedI64_SIMD(b *testing.B) {
	if simdMulVectorizedInt64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomInt64Slices()
	dst := make([]int64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedInt64(dst, aSlice, bSlice)
	}
}
