//go:build arm64 && !goexperiment.simd

#include "textflag.h"

// ============================================================================
// ARM64 NEON element-wise float32 operations.
//
// Encoding note: FADD, FSUB, FMUL, FDIV on .4S vectors are not yet Plan 9
// mnemonics in Go 1.26 arm64 assembly, so they are emitted via WORD.
//
// WORD encodings (verified against ARM Architecture Reference Manual):
//   fadd vRd.4s, vRn.4s, vRm.4s: 0x4E000000 | (Rm<<16) | (0x1A<<11) | (Rn<<5) | Rd
//   fsub vRd.4s, vRn.4s, vRm.4s: 0x4EA00000 | (Rm<<16) | (0x1A<<11) | (Rn<<5) | Rd
//   fmul vRd.4s, vRn.4s, vRm.4s: 0x6E200000 | (Rm<<16) | (0x1B<<11) | (Rn<<5) | Rd
//   fdiv vRd.4s, vRn.4s, vRm.4s: 0x6E20F800 | (Rm<<16) | (Rn<<5) | Rd
//
// For all kernels below:
//   V0 = loaded a vector
//   V1 = loaded b vector
//   V2 = destination vector (vectorized variants only; inplace stores back to V0)
//
// The loops process 4 float32 per iteration. The scalar tail handles 0-3 remainder
// elements using FMOVS (scalar float moves).
// ============================================================================

// ============================================================================
// Inplace kernels: a[i] op= b[i]
// Each slice arg: 24 bytes (base ptr + len + cap). Two slices = 48 bytes total.
// ============================================================================

// func neonAddInplaceFloat32(a, b []float32)
// Computes a[i] += b[i] using NEON FADD (.4S, 4 float32 per vector).
TEXT ·neonAddInplaceFloat32(SB), NOSPLIT, $0-48
	MOVD a_base+0(FP), R0   // R0 = &a[0]
	MOVD a_len+8(FP), R2    // R2 = len(a)
	MOVD b_base+24(FP), R1  // R1 = &b[0]

	// Vector loop: 4 float32 per iteration (16 bytes).
	LSR  $2, R2, R3          // R3 = len/4 (number of vector iterations)
	CBZ  R3, vtail_add

vloop_add:
	VLD1 (R0), [V0.S4]       // V0 = a[i:i+4]
	VLD1 (R1), [V1.S4]       // V1 = b[i:i+4]
	// fadd v0.4s, v0.4s, v1.4s
	WORD $0x4E21D400
	VST1 [V0.S4], (R0)       // a[i:i+4] = V0
	ADD  $16, R0
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, vloop_add

vtail_add:
	// Scalar tail: process remaining 0-3 elements.
	AND  $3, R2, R3          // R3 = len % 4
	CBZ  R3, done_add

stail_add:
	FMOVS (R0), F0           // F0 = a[i]
	FMOVS (R1), F1           // F1 = b[i]
	FADDS F1, F0, F0         // F0 = F0 + F1
	FMOVS F0, (R0)           // a[i] = F0
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R3
	CBNZ  R3, stail_add

done_add:
	RET

// func neonSubInplaceFloat32(a, b []float32)
// Computes a[i] -= b[i] using NEON FSUB (.4S).
TEXT ·neonSubInplaceFloat32(SB), NOSPLIT, $0-48
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1

	LSR  $2, R2, R3
	CBZ  R3, vtail_sub

vloop_sub:
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	// fsub v0.4s, v0.4s, v1.4s
	WORD $0x4EA1D400
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, vloop_sub

vtail_sub:
	AND  $3, R2, R3
	CBZ  R3, done_sub

stail_sub:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FSUBS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R3
	CBNZ  R3, stail_sub

done_sub:
	RET

// func neonMulInplaceFloat32(a, b []float32)
// Computes a[i] *= b[i] using NEON FMUL (.4S).
TEXT ·neonMulInplaceFloat32(SB), NOSPLIT, $0-48
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1

	LSR  $2, R2, R3
	CBZ  R3, vtail_mul

vloop_mul:
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	// fmul v0.4s, v0.4s, v1.4s
	WORD $0x6E21DC00
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, vloop_mul

vtail_mul:
	AND  $3, R2, R3
	CBZ  R3, done_mul

stail_mul:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FMULS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R3
	CBNZ  R3, stail_mul

done_mul:
	RET

// func neonDivInplaceFloat32(a, b []float32)
// Computes a[i] /= b[i] using NEON FDIV (.4S).
TEXT ·neonDivInplaceFloat32(SB), NOSPLIT, $0-48
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1

	LSR  $2, R2, R3
	CBZ  R3, vtail_div

vloop_div:
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	// fdiv v0.4s, v0.4s, v1.4s
	WORD $0x6E21FC00
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, vloop_div

vtail_div:
	AND  $3, R2, R3
	CBZ  R3, done_div

stail_div:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FDIVS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R3
	CBNZ  R3, stail_div

done_div:
	RET

// ============================================================================
// Vectorized kernels: dst[i] = a[i] op b[i]
// Three slice args = 72 bytes total.
// ============================================================================

// func neonAddVectorizedFloat32(dst, a, b []float32)
// Computes dst[i] = a[i] + b[i] using NEON FADD (.4S).
TEXT ·neonAddVectorizedFloat32(SB), NOSPLIT, $0-72
	MOVD dst_base+0(FP), R0   // R0 = &dst[0]
	MOVD dst_len+8(FP), R3    // R3 = len(dst)
	MOVD a_base+24(FP), R1    // R1 = &a[0]
	MOVD b_base+48(FP), R2    // R2 = &b[0]

	LSR  $2, R3, R4            // R4 = len/4
	CBZ  R4, vtail_vadd

vloop_vadd:
	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	// fadd v2.4s, v0.4s, v1.4s
	WORD $0x4E21D402
	VST1 [V2.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $1, R4
	CBNZ R4, vloop_vadd

vtail_vadd:
	AND  $3, R3, R4
	CBZ  R4, done_vadd

stail_vadd:
	FMOVS (R1), F0
	FMOVS (R2), F1
	FADDS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	ADD   $4, R2
	SUB   $1, R4
	CBNZ  R4, stail_vadd

done_vadd:
	RET

// func neonSubVectorizedFloat32(dst, a, b []float32)
// Computes dst[i] = a[i] - b[i] using NEON FSUB (.4S).
TEXT ·neonSubVectorizedFloat32(SB), NOSPLIT, $0-72
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R3
	MOVD a_base+24(FP), R1
	MOVD b_base+48(FP), R2

	LSR  $2, R3, R4
	CBZ  R4, vtail_vsub

vloop_vsub:
	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	// fsub v2.4s, v0.4s, v1.4s
	WORD $0x4EA1D402
	VST1 [V2.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $1, R4
	CBNZ R4, vloop_vsub

vtail_vsub:
	AND  $3, R3, R4
	CBZ  R4, done_vsub

stail_vsub:
	FMOVS (R1), F0
	FMOVS (R2), F1
	FSUBS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	ADD   $4, R2
	SUB   $1, R4
	CBNZ  R4, stail_vsub

done_vsub:
	RET

// func neonMulVectorizedFloat32(dst, a, b []float32)
// Computes dst[i] = a[i] * b[i] using NEON FMUL (.4S).
TEXT ·neonMulVectorizedFloat32(SB), NOSPLIT, $0-72
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R3
	MOVD a_base+24(FP), R1
	MOVD b_base+48(FP), R2

	LSR  $2, R3, R4
	CBZ  R4, vtail_vmul

vloop_vmul:
	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	// fmul v2.4s, v0.4s, v1.4s
	WORD $0x6E21DC02
	VST1 [V2.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $1, R4
	CBNZ R4, vloop_vmul

vtail_vmul:
	AND  $3, R3, R4
	CBZ  R4, done_vmul

stail_vmul:
	FMOVS (R1), F0
	FMOVS (R2), F1
	FMULS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	ADD   $4, R2
	SUB   $1, R4
	CBNZ  R4, stail_vmul

done_vmul:
	RET

// func neonDivVectorizedFloat32(dst, a, b []float32)
// Computes dst[i] = a[i] / b[i] using NEON FDIV (.4S).
TEXT ·neonDivVectorizedFloat32(SB), NOSPLIT, $0-72
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R3
	MOVD a_base+24(FP), R1
	MOVD b_base+48(FP), R2

	LSR  $2, R3, R4
	CBZ  R4, vtail_vdiv

vloop_vdiv:
	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	// fdiv v2.4s, v0.4s, v1.4s
	WORD $0x6E21FC02
	VST1 [V2.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $1, R4
	CBNZ R4, vloop_vdiv

vtail_vdiv:
	AND  $3, R3, R4
	CBZ  R4, done_vdiv

stail_vdiv:
	FMOVS (R1), F0
	FMOVS (R2), F1
	FDIVS F1, F0, F0
	FMOVS F0, (R0)
	ADD   $4, R0
	ADD   $4, R1
	ADD   $4, R2
	SUB   $1, R4
	CBNZ  R4, stail_vdiv

done_vdiv:
	RET
