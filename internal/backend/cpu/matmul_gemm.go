package cpu

// gemmMinCols is the smallest n for which the SIMD GEMM kernel dispatches.
// The kernel vectorizes over columns in gemmNr-wide tiles; shapes with n < gemmMinCols
// have no full tile and lose to the cache-tiled scalar path, so dispatch skips them.
//
// Initialized to gemmNr by the arch-specific init (matmul_gemm_<arch>.go):
//
//	amd64: 16 (gemmNr=16, AVX2 256-bit)
//	arm64:  8 (gemmNr=8,  NEON 128-bit)
//
// The zero value keeps gemmMinCols disabled (no SIMD dispatch) on arches without
// a vendored kernel.
var gemmMinCols int

// gemmF32 is the optional vendored-SIMD GEMM fast path for float32:
//
//	C[m,n] = A[m,k] @ B[k,n]   (row-major, overwriting C)
//
// It is nil by default and wired in by an arch-specific init when the CPU supports
// the required instructions (AVX2+FMA on amd64, see matmul_gemm_amd64.go). When
// non-nil, matmulFloat32 dispatches large multiplications here instead of the
// scalar blocked path; when nil (other arches, older CPUs, or a GOEXPERIMENT=simd
// build where the archsimd micro-kernel owns dispatch) the scalar path is used
// unchanged.
var gemmF32 func(c, a, b []float32, m, k, n int)

// transposeF32 writes the [rows, cols] -> [cols, rows] transpose of src into dst:
// dst[c*rows+r] = src[r*cols+c]. Used to recast the conv im2col product
// out = kernel @ colBuf^T into the GEMM kernel's A @ B form.
func transposeF32(dst, src []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		base := r * cols
		for c := 0; c < cols; c++ {
			dst[c*rows+r] = src[base+c]
		}
	}
}
