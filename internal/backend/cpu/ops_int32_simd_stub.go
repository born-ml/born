//go:build !amd64 || !goexperiment.simd

package cpu

// These functions are nil when SIMD is unavailable (non-amd64 or built
// without GOEXPERIMENT=simd).  They fall back to the scalar
// loop when nil.
var simdAddInplaceInt32 func(a, b []int32)
var simdSubInplaceInt32 func(a, b []int32)
var simdMulInplaceInt32 func(a, b []int32)

var simdAddVectorizedInt32 func(dst, a, b []int32)
var simdSubVectorizedInt32 func(dst, a, b []int32)
var simdMulVectorizedInt32 func(dst, a, b []int32)
