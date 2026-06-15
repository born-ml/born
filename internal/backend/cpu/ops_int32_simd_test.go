package cpu

import (
	"math/rand"
	"testing"
)

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

func BenchmarkAddInplaceI32_SIMD(b *testing.B) {
	if simdAddInplaceInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		addInplaceInt32(aSlice, bSlice)
	}
}

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

func BenchmarkSubInplaceI32_SIMD(b *testing.B) {
	if simdSubInplaceInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		subInplaceInt32(aSlice, bSlice)
	}
}

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

func BenchmarkMulInplaceI32_SIMD(b *testing.B) {
	if simdMulInplaceInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()

	b.ResetTimer()
	for b.Loop() {
		mulInplaceInt32(aSlice, bSlice)
	}
}

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

func BenchmarkAddVectorizedI32_SIMD(b *testing.B) {
	if simdAddVectorizedInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedInt32(dst, aSlice, bSlice)
	}
}

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

func BenchmarkSubVectorizedI32_SIMD(b *testing.B) {
	if simdSubVectorizedInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedInt32(dst, aSlice, bSlice)
	}
}

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

func BenchmarkMulVectorizedI32_SIMD(b *testing.B) {
	if simdMulVectorizedInt32 == nil {
		b.Skip("SIMD kernel not available")
	}

	aSlice, bSlice := createRandomInt32Slices()
	dst := make([]int32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedInt32(dst, aSlice, bSlice)
	}
}
