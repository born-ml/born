package cpu

// gemmMinCols is the smallest n the SIMD GEMM kernel handles with at least one
// full 16-wide column tile. The kernel vectorizes over n (4-row tiles plus a
// 1-row remainder/GEMV path), so any m >= 1 is fine, but for n below this every
// column would fall to a naive scalar loop that loses to the cache-tiled scalar
// path, so the dispatch keeps such narrow shapes on scalar.
const gemmMinCols = 16

// gemmF32 is the optional vendored-SIMD GEMM fast path for float32:
//
//	C[m,n] = A[m,k] @ B[k,n]   (row-major, overwriting C)
//
// It is nil by default. An arch-specific build wires in a kernel only when BOTH
// the experimental env flag BORN_EXPERIMENTAL_SIMD is set AND the CPU supports
// the required instructions (see matmul_gemm_amd64.go). When non-nil,
// matmulFloat32 dispatches large multiplications here instead of the scalar
// blocked path; when nil the scalar path is used unchanged.
var gemmF32 func(c, a, b []float32, m, k, n int)
