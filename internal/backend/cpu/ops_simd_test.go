package cpu

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// TestAddInplaceFloat32_SIMDMatchesScalar verifies that the SIMD add-inplace
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestAddInplaceFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdAddInplaceFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(1))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			aScalar := make([]float32, n)
			copy(aScalar, a)
			saved := simdAddInplaceFloat32
			simdAddInplaceFloat32 = nil
			addInplaceFloat32(aScalar, b)
			simdAddInplaceFloat32 = saved

			addInplaceFloat32(a, b)

			for i := range a {
				diff := math.Abs(float64(a[i] - aScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestSubInplaceFloat32_SIMDMatchesScalar verifies that the SIMD sub-inplace
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestSubInplaceFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdSubInplaceFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(2))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			aScalar := make([]float32, n)
			copy(aScalar, a)
			saved := simdSubInplaceFloat32
			simdSubInplaceFloat32 = nil
			subInplaceFloat32(aScalar, b)
			simdSubInplaceFloat32 = saved

			subInplaceFloat32(a, b)

			for i := range a {
				diff := math.Abs(float64(a[i] - aScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestMulInplaceFloat32_SIMDMatchesScalar verifies that the SIMD mul-inplace
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestMulInplaceFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdMulInplaceFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(3))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			aScalar := make([]float32, n)
			copy(aScalar, a)
			saved := simdMulInplaceFloat32
			simdMulInplaceFloat32 = nil
			mulInplaceFloat32(aScalar, b)
			simdMulInplaceFloat32 = saved

			mulInplaceFloat32(a, b)

			for i := range a {
				diff := math.Abs(float64(a[i] - aScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestDivInplaceFloat32_SIMDMatchesScalar verifies that the SIMD div-inplace
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestDivInplaceFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdDivInplaceFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(4))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 0.5
			}

			aScalar := make([]float32, n)
			copy(aScalar, a)
			saved := simdDivInplaceFloat32
			simdDivInplaceFloat32 = nil
			divInplaceFloat32(aScalar, b)
			simdDivInplaceFloat32 = saved

			divInplaceFloat32(a, b)

			for i := range a {
				diff := math.Abs(float64(a[i] - aScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestAddVectorizedFloat32_SIMDMatchesScalar verifies that the SIMD add-vectorized
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestAddVectorizedFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdAddVectorizedFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(5))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			dstScalar := make([]float32, n)
			saved := simdAddVectorizedFloat32
			simdAddVectorizedFloat32 = nil
			addVectorizedFloat32(dstScalar, a, b)
			simdAddVectorizedFloat32 = saved

			dstSIMD := make([]float32, n)
			addVectorizedFloat32(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(float64(dstSIMD[i] - dstScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestSubVectorizedFloat32_SIMDMatchesScalar verifies that the SIMD sub-vectorized
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestSubVectorizedFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdSubVectorizedFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(6))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			dstScalar := make([]float32, n)
			saved := simdSubVectorizedFloat32
			simdSubVectorizedFloat32 = nil
			subVectorizedFloat32(dstScalar, a, b)
			simdSubVectorizedFloat32 = saved

			dstSIMD := make([]float32, n)
			subVectorizedFloat32(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(float64(dstSIMD[i] - dstScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestMulVectorizedFloat32_SIMDMatchesScalar verifies that the SIMD mul-vectorized
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestMulVectorizedFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdMulVectorizedFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(7))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}

			dstScalar := make([]float32, n)
			saved := simdMulVectorizedFloat32
			simdMulVectorizedFloat32 = nil
			mulVectorizedFloat32(dstScalar, a, b)
			simdMulVectorizedFloat32 = saved

			dstSIMD := make([]float32, n)
			mulVectorizedFloat32(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(float64(dstSIMD[i] - dstScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestDivVectorizedFloat32_SIMDMatchesScalar verifies that the SIMD div-vectorized
// kernel produces results matching the scalar fallback within float32 ULP noise.
func TestDivVectorizedFloat32_SIMDMatchesScalar(t *testing.T) {
	if simdDivVectorizedFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-5

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(8))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 0.5
			}

			dstScalar := make([]float32, n)
			saved := simdDivVectorizedFloat32
			simdDivVectorizedFloat32 = nil
			divVectorizedFloat32(dstScalar, a, b)
			simdDivVectorizedFloat32 = saved

			dstSIMD := make([]float32, n)
			divVectorizedFloat32(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(float64(dstSIMD[i] - dstScalar[i]))
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.8f scalar=%.8f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestAddInplaceFloat64_SIMDMatchesScalar verifies that the SIMD add-inplace
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestAddInplaceFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdAddInplaceFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(10))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			aScalar := make([]float64, n)
			copy(aScalar, a)
			saved := simdAddInplaceFloat64
			simdAddInplaceFloat64 = nil
			addInplaceFloat64(aScalar, b)
			simdAddInplaceFloat64 = saved

			addInplaceFloat64(a, b)

			for i := range a {
				diff := math.Abs(a[i] - aScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestSubInplaceFloat64_SIMDMatchesScalar verifies that the SIMD sub-inplace
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestSubInplaceFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdSubInplaceFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(11))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			aScalar := make([]float64, n)
			copy(aScalar, a)
			saved := simdSubInplaceFloat64
			simdSubInplaceFloat64 = nil
			subInplaceFloat64(aScalar, b)
			simdSubInplaceFloat64 = saved

			subInplaceFloat64(a, b)

			for i := range a {
				diff := math.Abs(a[i] - aScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestMulInplaceFloat64_SIMDMatchesScalar verifies that the SIMD mul-inplace
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestMulInplaceFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdMulInplaceFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(12))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			aScalar := make([]float64, n)
			copy(aScalar, a)
			saved := simdMulInplaceFloat64
			simdMulInplaceFloat64 = nil
			mulInplaceFloat64(aScalar, b)
			simdMulInplaceFloat64 = saved

			mulInplaceFloat64(a, b)

			for i := range a {
				diff := math.Abs(a[i] - aScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestDivInplaceFloat64_SIMDMatchesScalar verifies that the SIMD div-inplace
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestDivInplaceFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdDivInplaceFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(13))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 0.5
			}

			aScalar := make([]float64, n)
			copy(aScalar, a)
			saved := simdDivInplaceFloat64
			simdDivInplaceFloat64 = nil
			divInplaceFloat64(aScalar, b)
			simdDivInplaceFloat64 = saved

			divInplaceFloat64(a, b)

			for i := range a {
				diff := math.Abs(a[i] - aScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, a[i], aScalar[i], diff)
				}
			}
		})
	}
}

// TestAddVectorizedFloat64_SIMDMatchesScalar verifies that the SIMD add-vectorized
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestAddVectorizedFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdAddVectorizedFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(14))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			dstScalar := make([]float64, n)
			saved := simdAddVectorizedFloat64
			simdAddVectorizedFloat64 = nil
			addVectorizedFloat64(dstScalar, a, b)
			simdAddVectorizedFloat64 = saved

			dstSIMD := make([]float64, n)
			addVectorizedFloat64(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(dstSIMD[i] - dstScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestSubVectorizedFloat64_SIMDMatchesScalar verifies that the SIMD sub-vectorized
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestSubVectorizedFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdSubVectorizedFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(15))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			dstScalar := make([]float64, n)
			saved := simdSubVectorizedFloat64
			simdSubVectorizedFloat64 = nil
			subVectorizedFloat64(dstScalar, a, b)
			simdSubVectorizedFloat64 = saved

			dstSIMD := make([]float64, n)
			subVectorizedFloat64(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(dstSIMD[i] - dstScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestMulVectorizedFloat64_SIMDMatchesScalar verifies that the SIMD mul-vectorized
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestMulVectorizedFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdMulVectorizedFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(16))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 1
			}

			dstScalar := make([]float64, n)
			saved := simdMulVectorizedFloat64
			simdMulVectorizedFloat64 = nil
			mulVectorizedFloat64(dstScalar, a, b)
			simdMulVectorizedFloat64 = saved

			dstSIMD := make([]float64, n)
			mulVectorizedFloat64(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(dstSIMD[i] - dstScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestDivVectorizedFloat64_SIMDMatchesScalar verifies that the SIMD div-vectorized
// kernel produces results matching the scalar fallback within float64 ULP noise.
func TestDivVectorizedFloat64_SIMDMatchesScalar(t *testing.T) {
	if simdDivVectorizedFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	const maxDiff = 1e-10

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(17))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]float64, n)
			b := make([]float64, n)
			for i := range a {
				a[i] = rng.Float64()*2 - 1
				b[i] = rng.Float64()*2 - 0.5
			}

			dstScalar := make([]float64, n)
			saved := simdDivVectorizedFloat64
			simdDivVectorizedFloat64 = nil
			divVectorizedFloat64(dstScalar, a, b)
			simdDivVectorizedFloat64 = saved

			dstSIMD := make([]float64, n)
			divVectorizedFloat64(dstSIMD, a, b)

			for i := range dstSIMD {
				diff := math.Abs(dstSIMD[i] - dstScalar[i])
				if diff > maxDiff {
					t.Fatalf("element[%d]: SIMD=%.15f scalar=%.15f diff=%.2e", i, dstSIMD[i], dstScalar[i], diff)
				}
			}
		})
	}
}

// TestAddInplaceInt32_SIMDMatchesScalar verifies that the SIMD add-inplace
// kernel produces bit-exact results matching the scalar fallback.
func TestAddInplaceInt32_SIMDMatchesScalar(t *testing.T) {
	if simdAddInplaceInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(20))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(1000) - 500)
				b[i] = int32(rng.Intn(1000) - 500)
			}

			aScalar := make([]int32, n)
			copy(aScalar, a)
			saved := simdAddInplaceInt32
			simdAddInplaceInt32 = nil
			addInplaceInt32(aScalar, b)
			simdAddInplaceInt32 = saved

			addInplaceInt32(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestSubInplaceInt32_SIMDMatchesScalar verifies that the SIMD sub-inplace
// kernel produces bit-exact results matching the scalar fallback.
func TestSubInplaceInt32_SIMDMatchesScalar(t *testing.T) {
	if simdSubInplaceInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(21))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(1000) - 500)
				b[i] = int32(rng.Intn(1000) - 500)
			}

			aScalar := make([]int32, n)
			copy(aScalar, a)
			saved := simdSubInplaceInt32
			simdSubInplaceInt32 = nil
			subInplaceInt32(aScalar, b)
			simdSubInplaceInt32 = saved

			subInplaceInt32(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestMulInplaceInt32_SIMDMatchesScalar verifies that the SIMD mul-inplace
// kernel produces bit-exact results matching the scalar fallback.
// Uses small values to avoid integer overflow.
func TestMulInplaceInt32_SIMDMatchesScalar(t *testing.T) {
	if simdMulInplaceInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(22))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(50) - 25)
				b[i] = int32(rng.Intn(50) - 25)
			}

			aScalar := make([]int32, n)
			copy(aScalar, a)
			saved := simdMulInplaceInt32
			simdMulInplaceInt32 = nil
			mulInplaceInt32(aScalar, b)
			simdMulInplaceInt32 = saved

			mulInplaceInt32(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestAddVectorizedInt32_SIMDMatchesScalar verifies that the SIMD add-vectorized
// kernel produces bit-exact results matching the scalar fallback.
func TestAddVectorizedInt32_SIMDMatchesScalar(t *testing.T) {
	if simdAddVectorizedInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(23))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(1000) - 500)
				b[i] = int32(rng.Intn(1000) - 500)
			}

			dstScalar := make([]int32, n)
			saved := simdAddVectorizedInt32
			simdAddVectorizedInt32 = nil
			addVectorizedInt32(dstScalar, a, b)
			simdAddVectorizedInt32 = saved

			dstSIMD := make([]int32, n)
			addVectorizedInt32(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}

// TestSubVectorizedInt32_SIMDMatchesScalar verifies that the SIMD sub-vectorized
// kernel produces bit-exact results matching the scalar fallback.
func TestSubVectorizedInt32_SIMDMatchesScalar(t *testing.T) {
	if simdSubVectorizedInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(24))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(1000) - 500)
				b[i] = int32(rng.Intn(1000) - 500)
			}

			dstScalar := make([]int32, n)
			saved := simdSubVectorizedInt32
			simdSubVectorizedInt32 = nil
			subVectorizedInt32(dstScalar, a, b)
			simdSubVectorizedInt32 = saved

			dstSIMD := make([]int32, n)
			subVectorizedInt32(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}

// TestMulVectorizedInt32_SIMDMatchesScalar verifies that the SIMD mul-vectorized
// kernel produces bit-exact results matching the scalar fallback.
// Uses small values to avoid integer overflow.
func TestMulVectorizedInt32_SIMDMatchesScalar(t *testing.T) {
	if simdMulVectorizedInt32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(25))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int32, n)
			b := make([]int32, n)
			for i := range a {
				a[i] = int32(rng.Intn(50) - 25)
				b[i] = int32(rng.Intn(50) - 25)
			}

			dstScalar := make([]int32, n)
			saved := simdMulVectorizedInt32
			simdMulVectorizedInt32 = nil
			mulVectorizedInt32(dstScalar, a, b)
			simdMulVectorizedInt32 = saved

			dstSIMD := make([]int32, n)
			mulVectorizedInt32(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}

// TestAddInplaceInt64_SIMDMatchesScalar verifies that the SIMD add-inplace
// kernel produces bit-exact results matching the scalar fallback.
func TestAddInplaceInt64_SIMDMatchesScalar(t *testing.T) {
	if simdAddInplaceInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(30))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(1000) - 500
				b[i] = rng.Int63n(1000) - 500
			}

			aScalar := make([]int64, n)
			copy(aScalar, a)
			saved := simdAddInplaceInt64
			simdAddInplaceInt64 = nil
			addInplaceInt64(aScalar, b)
			simdAddInplaceInt64 = saved

			addInplaceInt64(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestSubInplaceInt64_SIMDMatchesScalar verifies that the SIMD sub-inplace
// kernel produces bit-exact results matching the scalar fallback.
func TestSubInplaceInt64_SIMDMatchesScalar(t *testing.T) {
	if simdSubInplaceInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(31))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(1000) - 500
				b[i] = rng.Int63n(1000) - 500
			}

			aScalar := make([]int64, n)
			copy(aScalar, a)
			saved := simdSubInplaceInt64
			simdSubInplaceInt64 = nil
			subInplaceInt64(aScalar, b)
			simdSubInplaceInt64 = saved

			subInplaceInt64(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestMulInplaceInt64_SIMDMatchesScalar verifies that the SIMD mul-inplace
// kernel produces bit-exact results matching the scalar fallback.
// Uses small values to avoid integer overflow.
func TestMulInplaceInt64_SIMDMatchesScalar(t *testing.T) {
	if simdMulInplaceInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(32))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(50) - 25
				b[i] = rng.Int63n(50) - 25
			}

			aScalar := make([]int64, n)
			copy(aScalar, a)
			saved := simdMulInplaceInt64
			simdMulInplaceInt64 = nil
			mulInplaceInt64(aScalar, b)
			simdMulInplaceInt64 = saved

			mulInplaceInt64(a, b)

			for i := range a {
				if a[i] != aScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, a[i], aScalar[i])
				}
			}
		})
	}
}

// TestAddVectorizedInt64_SIMDMatchesScalar verifies that the SIMD add-vectorized
// kernel produces bit-exact results matching the scalar fallback.
func TestAddVectorizedInt64_SIMDMatchesScalar(t *testing.T) {
	if simdAddVectorizedInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(33))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(1000) - 500
				b[i] = rng.Int63n(1000) - 500
			}

			dstScalar := make([]int64, n)
			saved := simdAddVectorizedInt64
			simdAddVectorizedInt64 = nil
			addVectorizedInt64(dstScalar, a, b)
			simdAddVectorizedInt64 = saved

			dstSIMD := make([]int64, n)
			addVectorizedInt64(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}

// TestSubVectorizedInt64_SIMDMatchesScalar verifies that the SIMD sub-vectorized
// kernel produces bit-exact results matching the scalar fallback.
func TestSubVectorizedInt64_SIMDMatchesScalar(t *testing.T) {
	if simdSubVectorizedInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(34))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(1000) - 500
				b[i] = rng.Int63n(1000) - 500
			}

			dstScalar := make([]int64, n)
			saved := simdSubVectorizedInt64
			simdSubVectorizedInt64 = nil
			subVectorizedInt64(dstScalar, a, b)
			simdSubVectorizedInt64 = saved

			dstSIMD := make([]int64, n)
			subVectorizedInt64(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}

// TestMulVectorizedInt64_SIMDMatchesScalar verifies that the SIMD mul-vectorized
// kernel produces bit-exact results matching the scalar fallback.
// Uses small values to avoid integer overflow.
func TestMulVectorizedInt64_SIMDMatchesScalar(t *testing.T) {
	if simdMulVectorizedInt64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	sizes := []int{1, 3, 4, 7, 8, 13, 16, 31, 32, 64, 100, 128, 256, 1024}

	rng := rand.New(rand.NewSource(35))
	for _, n := range sizes {
		t.Run("n="+strconv.Itoa(n), func(t *testing.T) {
			a := make([]int64, n)
			b := make([]int64, n)
			for i := range a {
				a[i] = rng.Int63n(50) - 25
				b[i] = rng.Int63n(50) - 25
			}

			dstScalar := make([]int64, n)
			saved := simdMulVectorizedInt64
			simdMulVectorizedInt64 = nil
			mulVectorizedInt64(dstScalar, a, b)
			simdMulVectorizedInt64 = saved

			dstSIMD := make([]int64, n)
			mulVectorizedInt64(dstSIMD, a, b)

			for i := range dstSIMD {
				if dstSIMD[i] != dstScalar[i] {
					t.Fatalf("element[%d]: SIMD=%d scalar=%d", i, dstSIMD[i], dstScalar[i])
				}
			}
		})
	}
}
