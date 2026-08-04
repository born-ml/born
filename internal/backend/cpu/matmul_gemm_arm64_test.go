//go:build arm64 && !goexperiment.simd

package cpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/born-ml/born/internal/tensor"
	"golang.org/x/sys/cpu"
)

// TestGemmNEONF32MatchesScalar verifies the NEON GEMM kernel produces the same
// product as the naive reference across tile-aligned shapes, row/col/k tails,
// and a model-representative large-K shape. Mirrors TestGemmAVX2F32MatchesScalar
// on amd64 with shapes adapted to gemmMr=4, gemmNr=8.
func TestGemmNEONF32MatchesScalar(t *testing.T) {
	if !cpu.ARM64.HasASIMD {
		t.Skip("ASIMD/NEON not available on this CPU")
	}
	r := rand.New(rand.NewSource(0x6e656f6e)) // "neon"

	shapes := []struct{ m, k, n int }{
		{1, 1, 1},
		{4, 0, 8},    // empty inner dimension (k==0): zero matrix
		{1, 64, 16},  // GEMV: 1 row, full column tiles (1x8 kernel)
		{2, 100, 24}, // 2 remainder rows
		{3, 77, 17},  // 3 remainder rows + column tail
		{4, 8, 8},    // exact 4x8 tile
		{4, 8, 9},    // column tail
		{5, 8, 8},    // 1 remainder row
		{5, 8, 9},    // row + column tail
		{3, 7, 11},   // small odd
		{8, 32, 16},  // multi-tile
		{16, 64, 64},
		{33, 65, 65}, // odd primes-ish, both tails
		// 4x8 packing stress: every m%4 residue against full/partial n tiles.
		{4, 8, 8},      // exact 4x8 tile
		{4, 40, 32},    // multi-k, multi-n-tile, exact rows
		{5, 16, 8},     // 1 remainder row over a full block
		{8, 16, 16},    // two full row blocks
		{9, 17, 17},    // two blocks + 1 row + k/n tails
		{4, 5, 8},      // k < default block, exact rows
		{4, 8, 9},      // exact rows + column tail
		{9, 8, 8},      // 2 blocks + 1 remainder row
		{1, 512, 48},   // classifier-like GEMV, full n tiles
		{7, 1024, 257}, // model-representative large K + row/col tails
		{4, 1024, 16},  // large K, exact rows
		// Column-tail stress: n%gemmNr == 4 across row residues.
		{7, 64, 12},  // 1 full tile + 4-col tail, row tail
		{9, 32, 20},  // 2 full tiles + 4-col tail, row tail
		{64, 64, 12}, // model-shaped: many blocks + 4-col tail
		{1, 512, 12}, // GEMV + 4-col tail
		{3, 32, 4},   // pure tail (n < gemmNr), thin rows
		{8, 64, 4},   // pure tail (n < gemmNr), full blocks
	}

	for _, s := range shapes {
		t.Run("", func(t *testing.T) {
			a := randSliceF32(r, s.m*s.k)
			b := randSliceF32(r, s.k*s.n)
			want := naiveMatMulF32(a, b, s.m, s.k, s.n)

			got := make([]float32, s.m*s.n)
			for i := range got {
				got[i] = 12345.0 // poison: kernel must overwrite, not accumulate
			}
			gemmNEONF32(got, a, b, s.m, s.k, s.n)

			var maxDiff float64
			for i := range want {
				d := math.Abs(float64(got[i]-want[i])) / (1 + math.Abs(float64(want[i])))
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-4 {
				t.Errorf("shape %dx%dx%d: max rel diff %.3e exceeds 1e-4", s.m, s.k, s.n, maxDiff)
			}
		})
	}
}

// TestGemmNEONDispatch verifies matmulFloat32 routes large multiplications to
// the NEON kernel (m*k*n >= blockThreshold) and small ones to the scalar path,
// and that both produce correct results.
func TestGemmNEONDispatch(t *testing.T) {
	if !cpu.ARM64.HasASIMD {
		t.Skip("ASIMD/NEON not available on this CPU")
	}
	r := rand.New(rand.NewSource(0x6469737061)) // "dispa"

	var called bool
	prev := gemmF32
	gemmF32 = func(c, a, b []float32, m, k, n int) {
		called = true
		gemmNEONF32(c, a, b, m, k, n)
	}
	t.Cleanup(func() { gemmF32 = prev })

	cases := []struct {
		name          string
		m, k, n       int
		wantGemmTaken bool
	}{
		{"large routes to gemm", 32, 1024, 64, true},           // full 4x8 tiles
		{"thin m<4 routes to gemm (GEMV)", 1, 512, 3072, true}, // 1x8 GEMV path
		{"small stays scalar", 8, 8, 8, false},                 // below blockThreshold
		{"narrow n<8 stays scalar", 32, 512, 4, false},         // n < gemmMinCols
	}

	be := New()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			called = false
			a := randSliceF32(r, tc.m*tc.k)
			b := randSliceF32(r, tc.k*tc.n)
			at := rawFromSlice(t, a, tc.m, tc.k)
			bt := rawFromSlice(t, b, tc.k, tc.n)

			got := be.MatMul(at, bt).AsFloat32()
			want := naiveMatMulF32(a, b, tc.m, tc.k, tc.n)

			if called != tc.wantGemmTaken {
				t.Errorf("gemm path taken = %v, want %v", called, tc.wantGemmTaken)
			}
			var maxDiff float64
			for i := range want {
				d := math.Abs(float64(got[i]-want[i])) / (1 + math.Abs(float64(want[i])))
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-4 {
				t.Errorf("max rel diff %.3e exceeds 1e-4", maxDiff)
			}
		})
	}
}

// TestGemmNEONNoAllocs asserts the NEON GEMM fast path performs no heap allocations
// in steady state (packing buffers are reused from the pool).
func TestGemmNEONNoAllocs(t *testing.T) {
	if testing.Short() {
		t.Skip("AllocsPerRun over the shared sync.Pool is unreliable under -short -race")
	}
	if !cpu.ARM64.HasASIMD {
		t.Skip("ASIMD/NEON not available on this CPU")
	}
	const m, k, n = 32, 256, 64
	r := rand.New(rand.NewSource(1))
	a := randSliceF32(r, m*k)
	b := randSliceF32(r, k*n)
	c := make([]float32, m*n)
	if allocs := testing.AllocsPerRun(20, func() { gemmNEONF32(c, a, b, m, k, n) }); allocs != 0 {
		t.Errorf("gemmNEONF32 allocated %v times, want 0", allocs)
	}
}

// TestGemmNEONWiredIn verifies the always-on dispatch contract: init wires the
// kernel into gemmF32 exactly when the CPU has ASIMD.
func TestGemmNEONWiredIn(t *testing.T) {
	want := cpu.ARM64.HasASIMD
	if got := gemmF32 != nil; got != want {
		t.Errorf("gemmF32 wired in = %v, want %v (ASIMD=%v)", got, want, cpu.ARM64.HasASIMD)
	}
}

// BenchmarkGemmNEON compares the scalar matmul against the NEON kernel at
// model-representative GEMM shapes.
func BenchmarkGemmNEON(b *testing.B) {
	shapes := []struct {
		name    string
		m, k, n int
	}{
		{"attn_64x512x512", 64, 512, 512},
		{"ffn_32x2048x512", 32, 2048, 512},
		{"gemv_1x512x3072", 1, 512, 3072},
		{"square_256", 256, 256, 256},
	}
	r := rand.New(rand.NewSource(7))
	for _, s := range shapes {
		a := randSliceF32(r, s.m*s.k)
		bb := randSliceF32(r, s.k*s.n)
		c := make([]float32, s.m*s.n)
		b.Run(s.name+"/scalar", func(b *testing.B) {
			prev := gemmF32
			gemmF32 = nil
			defer func() { gemmF32 = prev }()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmulFloat32(c, a, bb, s.m, s.k, s.n)
			}
		})
		b.Run(s.name+"/neon", func(b *testing.B) {
			if !cpu.ARM64.HasASIMD {
				b.Skip("ASIMD not available")
			}
			for i := 0; i < b.N; i++ {
				gemmNEONF32(c, a, bb, s.m, s.k, s.n)
			}
		})
	}
}

// rawFromSlice builds a row-major float32 RawTensor with the given shape.
// Declared here because the amd64 version in matmul_gemm_amd64_test.go is
// build-tag-gated and unavailable on arm64.
func rawFromSlice(t *testing.T, data []float32, shape ...int) *tensor.RawTensor {
	t.Helper()
	sh := make(tensor.Shape, len(shape))
	copy(sh, shape)
	rt, err := tensor.NewRaw(sh, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(rt.AsFloat32(), data)
	return rt
}

// naiveMatMulF32 is an independent reference for GEMM correctness checking.
// Declared here because the amd64 version in matmul_gemm_amd64_test.go is
// build-tag-gated and unavailable on arm64.
func naiveMatMulF32(a, b []float32, m, k, n int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// randSliceF32 returns a slice of length n filled with normally-distributed
// random float32 values.
func randSliceF32(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}
