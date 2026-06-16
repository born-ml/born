//go:build amd64

package cpu

//go:generate sh -c "cd _gen/gemm && go run . -out ../../gemm_microkernel_amd64.s -stubs ../../gemm_microkernel_stub_amd64.go -pkg cpu"

import (
	"os"

	"golang.org/x/sys/cpu"
)

// simdGemmEnabled reports whether the experimental vendored GEMM kernel should
// be wired into the dispatch: the opt-in env flag must be exactly "1" (matching
// the repo's BORN_DEBUG_GPU convention) and the CPU must support AVX2+FMA. Kept
// as a pure function so the gate logic is unit-testable without touching the
// process environment or relying on the host CPU.
func simdGemmEnabled(envFlag string, hasAVX2, hasFMA bool) bool {
	return envFlag == "1" && hasAVX2 && hasFMA
}

// init wires the vendored AVX2+FMA GEMM kernel into the matmul dispatch, but
// only when BORN_EXPERIMENTAL_SIMD=1 and the CPU supports AVX2+FMA. Default
// builds (flag unset) keep the scalar path, so this is opt-in.
func init() {
	if simdGemmEnabled(os.Getenv("BORN_EXPERIMENTAL_SIMD"), cpu.X86.HasAVX2, cpu.X86.HasFMA) {
		gemmF32 = gemmAVX2F32
	}
}

// gemmAVX2F32 computes C[m,n] = A[m,k] @ B[k,n] (row-major, overwriting C). Full
// 4x16 tiles use the vendored AVX2+FMA micro-kernel (C held in registers across
// the k-loop); the row/column remainders fall back to a scalar dot product.
func gemmAVX2F32(c, a, b []float32, m, k, n int) {
	if k == 0 {
		// Empty inner dimension: the product is the zero matrix. The matmulFloat32
		// dispatch already excludes k==0 (m*k*n < blockThreshold), but guard here so
		// a direct call cannot reslice b past its zero length in the tile loop.
		clear(c[:m*n])
		return
	}

	const mr, nr = 4, 16
	mFull := m - m%mr
	nFull := n - n%nr

	// Full 4x16 register tiles.
	for i := 0; i < mFull; i += mr {
		for j := 0; j < nFull; j += nr {
			gemmMicroKernel4x16AVX2(c[i*n+j:], a[i*k:], b[j:], k, n)
		}
	}
	// Remainder rows [mFull, m): one row at a time over full 16-col tiles
	// (this is also the GEMV path when m < 4, so thin shapes stay vectorized).
	for i := mFull; i < m; i++ {
		for j := 0; j < nFull; j += nr {
			gemmMicroKernel1x16AVX2(c[i*n+j:], a[i*k:], b[j:], k, n)
		}
	}
	// Column remainder [nFull, n) for all rows: scalar dot product.
	for i := 0; i < m; i++ {
		for j := nFull; j < n; j++ {
			var s float32
			for kk := 0; kk < k; kk++ {
				s += a[i*k+kk] * b[kk*n+j]
			}
			c[i*n+j] = s
		}
	}
}
