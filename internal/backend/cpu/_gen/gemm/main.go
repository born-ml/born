// Command gemm generates the vendored AVX2+FMA GEMM micro-kernel assembly for
// the CPU backend. It is run via `go generate` (see matmul_gemm_amd64.go) and
// lives in a separate module (_gen/gemm/go.mod) so avo never enters born's
// module graph. The generated artifacts (gemm_microkernel_amd64.s and its Go
// stub) are committed.
//
// The kernel computes a 4x16 tile of C = A @ B (row-major), holding all 8
// YMM accumulators in registers across the entire k-loop and storing C once
// at the end. This is the key difference from born's existing archsimd kernel,
// which reloads and stores the C block on every k iteration.
package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	TEXT("gemmMicroKernel4x16AVX2", NOSPLIT, "func(c, a, b []float32, k, n int)")
	Doc("gemmMicroKernel4x16AVX2 computes a 4x16 tile C[r,col] = sum_kk a[r*k+kk]*b[kk*n+col]",
		"(overwriting C) using AVX2+FMA with the C tile held in registers across the k-loop.",
		"c, a and b are the tile-local sub-slices; k and n are the A-row and B/C-row strides.")
	// The kernel reads/writes its slice arguments but never retains them, so the
	// pointers do not escape; this lets callers keep argument slices on the stack.
	Pragma("noescape")

	cBase := Load(Param("c").Base(), GP64())
	aBase := Load(Param("a").Base(), GP64())
	bBase := Load(Param("b").Base(), GP64())
	k := Load(Param("k"), GP64())
	n := Load(Param("n"), GP64())

	// Byte strides: kBytes = k*4 (A row stride), nBytes = n*4 (B and C row stride).
	kBytes := GP64()
	MOVQ(k, kBytes)
	SHLQ(Imm(2), kBytes)
	nBytes := GP64()
	MOVQ(n, nBytes)
	SHLQ(Imm(2), nBytes)

	// Running A-row pointers a0..a3 (rows 0..3 of the tile), advanced by 4 each kk.
	aPtr := make([]reg.GPVirtual, 4)
	aPtr[0] = GP64()
	MOVQ(aBase, aPtr[0])
	for r := 1; r < 4; r++ {
		aPtr[r] = GP64()
		MOVQ(aPtr[r-1], aPtr[r])
		ADDQ(kBytes, aPtr[r])
	}
	// Running B pointer, advanced by nBytes each kk.
	bPtr := GP64()
	MOVQ(bBase, bPtr)

	// Accumulators: acc[r][v], r in 0..3 rows, v in 0..1 (cols 0-7, 8-15). Zeroed.
	acc := [4][2]reg.VecVirtual{}
	for r := 0; r < 4; r++ {
		for v := 0; v < 2; v++ {
			acc[r][v] = YMM()
			VXORPS(acc[r][v], acc[r][v], acc[r][v])
		}
	}

	kCtr := GP64()
	MOVQ(k, kCtr)

	Label("kloop")
	CMPQ(kCtr, Imm(0))
	JE(LabelRef("kdone"))

	bVec0 := YMM()
	bVec1 := YMM()
	VMOVUPS(Mem{Base: bPtr}, bVec0)
	VMOVUPS(Mem{Base: bPtr, Disp: 32}, bVec1)

	aVec := YMM()
	for r := 0; r < 4; r++ {
		VBROADCASTSS(Mem{Base: aPtr[r]}, aVec)
		VFMADD231PS(bVec0, aVec, acc[r][0]) // acc = aVec*bVec0 + acc
		VFMADD231PS(bVec1, aVec, acc[r][1])
	}

	for r := 0; r < 4; r++ {
		ADDQ(Imm(4), aPtr[r])
	}
	ADDQ(nBytes, bPtr)
	DECQ(kCtr)
	JMP(LabelRef("kloop"))

	Label("kdone")
	// Store: running C-row pointers c0..c3 (advance by nBytes).
	cPtr := make([]reg.GPVirtual, 4)
	cPtr[0] = GP64()
	MOVQ(cBase, cPtr[0])
	for r := 1; r < 4; r++ {
		cPtr[r] = GP64()
		MOVQ(cPtr[r-1], cPtr[r])
		ADDQ(nBytes, cPtr[r])
	}
	for r := 0; r < 4; r++ {
		VMOVUPS(acc[r][0], Mem{Base: cPtr[r]})
		VMOVUPS(acc[r][1], Mem{Base: cPtr[r], Disp: 32})
	}

	// Clear the upper 128 bits of the YMM registers before returning to Go.
	// Without this, the dirty upper state forces the AVX->SSE transition penalty
	// on the surrounding (SSE-based) Go code. avo does not emit it automatically.
	VZEROUPPER()
	RET()

	emitKernel1x16()

	Generate()
}

// emitKernel1x16 generates the 1-row x 16-col micro-kernel. The driver uses it
// for the 1-3 remainder rows and for GEMV-shaped (m < 4) matmuls, so thin
// shapes are vectorized over the n dimension instead of falling to a naive
// scalar loop that loses to the cache-tiled scalar path.
func emitKernel1x16() {
	TEXT("gemmMicroKernel1x16AVX2", NOSPLIT, "func(c, a, b []float32, k, n int)")
	Doc("gemmMicroKernel1x16AVX2 computes a 1x16 tile C[0,col] = sum_kk a[kk]*b[kk*n+col]",
		"(overwriting C) using AVX2+FMA, with the two C accumulators held in",
		"registers across the k-loop. c and a are tile-local sub-slices; n is the",
		"B-row stride (a is a single row, so its stride k is unused).")
	Pragma("noescape")

	cBase := Load(Param("c").Base(), GP64())
	aBase := Load(Param("a").Base(), GP64())
	bBase := Load(Param("b").Base(), GP64())
	k := Load(Param("k"), GP64())
	n := Load(Param("n"), GP64())

	nBytes := GP64()
	MOVQ(n, nBytes)
	SHLQ(Imm(2), nBytes)

	acc0 := YMM()
	acc1 := YMM()
	VXORPS(acc0, acc0, acc0)
	VXORPS(acc1, acc1, acc1)

	aPtr := GP64()
	MOVQ(aBase, aPtr)
	bPtr := GP64()
	MOVQ(bBase, bPtr)
	kCtr := GP64()
	MOVQ(k, kCtr)

	Label("kloop1")
	CMPQ(kCtr, Imm(0))
	JE(LabelRef("kdone1"))

	aVec := YMM()
	VBROADCASTSS(Mem{Base: aPtr}, aVec)
	bVec0 := YMM()
	bVec1 := YMM()
	VMOVUPS(Mem{Base: bPtr}, bVec0)
	VMOVUPS(Mem{Base: bPtr, Disp: 32}, bVec1)
	VFMADD231PS(bVec0, aVec, acc0) // acc = a[kk]*bVec + acc
	VFMADD231PS(bVec1, aVec, acc1)

	ADDQ(Imm(4), aPtr)
	ADDQ(nBytes, bPtr)
	DECQ(kCtr)
	JMP(LabelRef("kloop1"))

	Label("kdone1")
	VMOVUPS(acc0, Mem{Base: cBase})
	VMOVUPS(acc1, Mem{Base: cBase, Disp: 32})
	VZEROUPPER()
	RET()
}
