//go:build arm64 && !goexperiment.simd

#include "textflag.h"

// func gemmMicroKernel4x8NEON(c []float32, a []float32, b []float32, k int, cStride int)
//
// Computes a 4×8 register-blocked GEMM tile for ARM64 NEON, overwriting C.
//
// Layout:
//   a is a packed [k][4] panel:   a[kk*4+r]  (gemmMr == 4)
//   b is a packed [k][8] panel:   b[kk*8+col] (gemmNr == 8)
//   c is the output sub-tile; cStride is the C row stride in float32 elements.
//
// Accumulators (8 V registers, 4 rows × 2 vectors of 4×float32):
//   V0  = C[row0, col 0..3]   V1  = C[row0, col 4..7]
//   V2  = C[row1, col 0..3]   V3  = C[row1, col 4..7]
//   V4  = C[row2, col 0..3]   V5  = C[row2, col 4..7]
//   V6  = C[row3, col 0..3]   V7  = C[row3, col 4..7]
// Working registers:
//   V8  = broadcast of one a scalar (VLD1R, all 4 lanes equal)
//   V9  = b[kk*8+0..3]  (first half of B row)
//   V10 = b[kk*8+4..7]  (second half of B row)
//
// General-purpose registers:
//   R0 = c base pointer
//   R1 = a base pointer
//   R2 = b base pointer
//   R3 = k (inner-dimension loop counter)
//   R4 = cStride * 4 (C row stride in bytes)
//   R5 = scratch for store offsets
//
// Argument frame (ABI0, stack-based, each []float32 = 24 bytes):
//   c_base+0(FP)       = c data pointer
//   c_len+8(FP)        = c length  (unused in kernel)
//   c_cap+16(FP)       = c cap     (unused in kernel)
//   a_base+24(FP)      = a data pointer
//   a_len+32(FP)       = a length  (unused)
//   a_cap+40(FP)       = a cap     (unused)
//   b_base+48(FP)      = b data pointer
//   b_len+56(FP)       = b length  (unused)
//   b_cap+64(FP)       = b cap     (unused)
//   k+72(FP)           = k
//   cStride+80(FP)     = cStride
//
// Requires: ARMv8-A ASIMD (NEON) — mandatory on all arm64 hardware.
TEXT ·gemmMicroKernel4x8NEON(SB), NOSPLIT, $0-88
	MOVD   c_base+0(FP), R0
	MOVD   a_base+24(FP), R1
	MOVD   b_base+48(FP), R2
	MOVD   k+72(FP), R3
	MOVD   cStride+80(FP), R4
	LSL    $2, R4            // cStride bytes = cStride * sizeof(float32)

	// Zero all 8 accumulator registers.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	CBZ R3, kdone4  // k == 0: zero matrix, skip to store

kloop4:
	// Load B: two consecutive NEON vectors = 8 float32 = 32 bytes.
	VLD1 (R2), [V9.S4, V10.S4]
	ADD  $32, R2

	// Row 0: VLD1R broadcasts a[kk*4+0] to all 4 lanes of V8, then VFMLA accumulates.
	VLD1R (R1), [V8.S4]
	ADD   $4, R1
	VFMLA V9.S4, V8.S4, V0.S4
	VFMLA V10.S4, V8.S4, V1.S4

	// Row 1.
	VLD1R (R1), [V8.S4]
	ADD   $4, R1
	VFMLA V9.S4, V8.S4, V2.S4
	VFMLA V10.S4, V8.S4, V3.S4

	// Row 2.
	VLD1R (R1), [V8.S4]
	ADD   $4, R1
	VFMLA V9.S4, V8.S4, V4.S4
	VFMLA V10.S4, V8.S4, V5.S4

	// Row 3.
	VLD1R (R1), [V8.S4]
	ADD   $4, R1
	VFMLA V9.S4, V8.S4, V6.S4
	VFMLA V10.S4, V8.S4, V7.S4

	SUB  $1, R3
	CBNZ R3, kloop4

kdone4:
	// Store 4 rows × 8 floats using the C row stride.
	MOVD R0, R5
	VST1 [V0.S4, V1.S4], (R5)  // row 0: 8 floats at c[row0, 0..7]
	ADD  R4, R5
	VST1 [V2.S4, V3.S4], (R5)  // row 1
	ADD  R4, R5
	VST1 [V4.S4, V5.S4], (R5)  // row 2
	ADD  R4, R5
	VST1 [V6.S4, V7.S4], (R5)  // row 3

	RET

// func gemmMicroKernel1x8NEON(c []float32, a []float32, b []float32, k int)
//
// Computes a 1×8 tile C[0, col 0..7] = sum_kk a[kk]*b[kk*8+col] (overwriting C)
// from an unpacked scalar row a ([k]) and the packed B panel b ([k][8]).
//
// Accumulators:
//   V0 = C[0, col 0..3]
//   V1 = C[0, col 4..7]
//
// Argument frame (ABI0):
//   c_base+0(FP) ... k+72(FP)  (total 80 bytes; no cStride argument)
TEXT ·gemmMicroKernel1x8NEON(SB), NOSPLIT, $0-80
	MOVD c_base+0(FP), R0
	MOVD a_base+24(FP), R1
	MOVD b_base+48(FP), R2
	MOVD k+72(FP), R3

	// Zero accumulators.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

	CBZ R3, kdone1

kloop1:
	// Load B row: 8 float32 from packed panel.
	VLD1 (R2), [V9.S4, V10.S4]
	ADD  $32, R2

	// Broadcast a[kk] and accumulate.
	VLD1R (R1), [V8.S4]
	ADD   $4, R1
	VFMLA V9.S4, V8.S4, V0.S4
	VFMLA V10.S4, V8.S4, V1.S4

	SUB  $1, R3
	CBNZ R3, kloop1

kdone1:
	// Store 8 floats at c[0, 0..7].
	VST1 [V0.S4, V1.S4], (R0)

	RET
