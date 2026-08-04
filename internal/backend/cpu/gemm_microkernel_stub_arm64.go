//go:build arm64 && !goexperiment.simd

package cpu

// gemmMicroKernel4x8NEON computes a 4×8 tile C[r,col] = sum_kk a[kk*4+r]*b[kk*8+col]
// (overwriting C) from PACKED panels a ([k][4]) and b ([k][8]), holding all 8 NEON
// accumulator registers (4 rows × 2 vectors of 4×float32) across the k-loop.
// cStride is the C row stride in elements.
//
//go:noescape
func gemmMicroKernel4x8NEON(c []float32, a []float32, b []float32, k int, cStride int)

// gemmMicroKernel1x8NEON computes a 1×8 tile C[0,col] = sum_kk a[kk]*b[kk*8+col]
// (overwriting C) from an unpacked source row a ([k]) and the packed B panel b ([k][8]),
// with the two NEON accumulators held in registers across the k-loop.
//
//go:noescape
func gemmMicroKernel1x8NEON(c []float32, a []float32, b []float32, k int)
