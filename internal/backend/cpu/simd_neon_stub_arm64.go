//go:build arm64 && !goexperiment.simd

package cpu

// ARM64 NEON element-wise float32 kernels. Implemented in simd_neon_arm64.s using
// NEON FADD/FSUB/FMUL/FDIV on 128-bit .4S vectors (4 float32 per operation).
// These are the Go declarations required by the Plan 9 assembler linkage.

// neonAddInplaceFloat32 computes a[i] += b[i] for all i using NEON FADD.
//
//go:noescape
func neonAddInplaceFloat32(a, b []float32)

// neonSubInplaceFloat32 computes a[i] -= b[i] for all i using NEON FSUB.
//
//go:noescape
func neonSubInplaceFloat32(a, b []float32)

// neonMulInplaceFloat32 computes a[i] *= b[i] for all i using NEON FMUL.
//
//go:noescape
func neonMulInplaceFloat32(a, b []float32)

// neonDivInplaceFloat32 computes a[i] /= b[i] for all i using NEON FDIV.
//
//go:noescape
func neonDivInplaceFloat32(a, b []float32)

// neonAddVectorizedFloat32 computes dst[i] = a[i] + b[i] for all i using NEON FADD.
//
//go:noescape
func neonAddVectorizedFloat32(dst, a, b []float32)

// neonSubVectorizedFloat32 computes dst[i] = a[i] - b[i] for all i using NEON FSUB.
//
//go:noescape
func neonSubVectorizedFloat32(dst, a, b []float32)

// neonMulVectorizedFloat32 computes dst[i] = a[i] * b[i] for all i using NEON FMUL.
//
//go:noescape
func neonMulVectorizedFloat32(dst, a, b []float32)

// neonDivVectorizedFloat32 computes dst[i] = a[i] / b[i] for all i using NEON FDIV.
//
//go:noescape
func neonDivVectorizedFloat32(dst, a, b []float32)
