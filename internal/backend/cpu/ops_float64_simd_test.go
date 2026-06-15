package cpu

import (
	"math/rand"
	"testing"
)

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

func BenchmarkAddInplaceF64_SIMD(b *testing.B) {
	if simdAddInplaceFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		addInplaceFloat64(aSlice, bSlice)
	}
}

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

func BenchmarkSubInplaceF64_SIMD(b *testing.B) {
	if simdSubInplaceFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		subInplaceFloat64(aSlice, bSlice)
	}
}

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

func BenchmarkMulInplaceF64_SIMD(b *testing.B) {
	if simdMulInplaceFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		mulInplaceFloat64(aSlice, bSlice)
	}
}

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

func BenchmarkDivInplaceF64_SIMD(b *testing.B) {
	if simdDivInplaceFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()

	b.ResetTimer()
	for b.Loop() {
		divInplaceFloat64(aSlice, bSlice)
	}
}

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

func BenchmarkAddVectorizedF64_SIMD(b *testing.B) {
	if simdAddVectorizedFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedFloat64(dst, aSlice, bSlice)
	}
}

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

func BenchmarkSubVectorizedF64_SIMD(b *testing.B) {
	if simdSubVectorizedFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedFloat64(dst, aSlice, bSlice)
	}
}

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

func BenchmarkMulVectorizedF64_SIMD(b *testing.B) {
	if simdMulVectorizedFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedFloat64(dst, aSlice, bSlice)
	}
}

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

func BenchmarkDivVectorizedF64_SIMD(b *testing.B) {
	if simdDivVectorizedFloat64 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomFloat64Slices()
	dst := make([]float64, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		divVectorizedFloat64(dst, aSlice, bSlice)
	}
}
