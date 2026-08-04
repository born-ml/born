//go:build arm64 && !goexperiment.simd

package cpu

import (
	"sync"

	"golang.org/x/sys/cpu"
)

// mr, nr are the GEMM micro-kernel tile dimensions for ARM64 NEON and MUST match
// the packed-panel strides baked into the assembly (gemm_microkernel_arm64.s).
//
// ARM64 has 32 V registers (128-bit), each holding 4×float32 (.S4 arrangement).
// We use 8 accumulators (V0–V7) for a 4-row × 2-vector = 4×8 tile, leaving V8–V10
// for the broadcast scalar and two B vectors.
//
// Compare with the amd64 kernel (gemmMr=6, gemmNr=16 using 256-bit AVX2 YMM):
// NEON vectors are 128-bit (4 float32 each) so gemmNr=8 spans two V registers,
// and gemmMr=4 keeps 8 accumulators without spilling (ARM64's larger register file
// allows 6 on amd64, but 4 is optimal for the 4×8 blocking here).
const (
	gemmMr = 4
	gemmNr = 8
)

// gemmScratch holds per-call packing buffers. They are pooled and grown lazily so
// the GEMM fast path stays allocation-free across calls.
type gemmScratch struct {
	ap []float32 // packed A: [nBlocks][k][gemmMr]
	bp []float32 // packed B: [nTiles][k][gemmNr]
	bt []float32 // packed tail B: [k][gemmNr] (n%gemmNr cols, zero-padded)
}

var gemmScratchPool = sync.Pool{New: func() any { return new(gemmScratch) }}

// ensureCap returns *buf resliced to length n, growing the backing array only when
// the current capacity is insufficient.
func ensureCap(buf *[]float32, n int) []float32 {
	if cap(*buf) < n {
		*buf = make([]float32, n)
	}
	return (*buf)[:n]
}

// init wires the NEON GEMM kernel into the matmul dispatch when the CPU exposes
// ASIMD (Advanced SIMD / NEON). ASIMD is a mandatory extension of every ARMv8-A
// processor, so this path is active on virtually all arm64 hardware. CPUs that
// somehow report ASIMD absent leave gemmF32 nil and fall back to the scalar path.
func init() {
	if cpu.ARM64.HasASIMD {
		gemmF32 = gemmNEONF32
		gemmMinCols = gemmNr // 8 for NEON
	}
}

// gemmNEONF32 computes C[m,n] = A[m,k] @ B[k,n] (row-major, overwriting C).
//
// The dispatch strategy mirrors the amd64 kernel (see matmul_gemm_amd64.go):
//   - Column tiles outermost so each packed B panel stays L1-resident across row blocks.
//   - Full 4×8 tiles: gemmMicroKernel4x8NEON (all accumulators in registers over k).
//   - m < gemmMr (thin / GEMV): no packing pays off; stream B with native stride.
//   - n%gemmNr tail: zero-pad into a gemmNr-wide panel, run 1x8, copy valid columns.
func gemmNEONF32(c, a, b []float32, m, k, n int) {
	if k == 0 {
		clear(c[:m*n])
		return
	}

	mFull := m - m%gemmMr
	nFull := n - n%gemmNr
	nrem := n - nFull

	sc := gemmScratchPool.Get().(*gemmScratch)
	defer gemmScratchPool.Put(sc)

	switch {
	case nFull == 0:
		// All columns are in the tail; handled below.
	case mFull == 0:
		gemvStridedNEONF32(c, a, b, m, k, n, nFull)
	default:
		gemmPackedNEONF32(c, a, b, m, k, n, mFull, nFull, sc)
	}

	if nrem > 0 {
		gemmTailNEONF32(c, a, b, m, k, n, nFull, nrem, sc)
	}
}

// gemvStridedNEONF32 handles thin shapes (m < gemmMr) over the full gemmNr-wide
// column tiles. Each B element feeds only one output row, so packing B would double
// its traffic for no reuse; instead a single-row packed panel is built per call.
func gemvStridedNEONF32(c, a, b []float32, m, k, n, nFull int) {
	// Temporary buffer for one row of packed B ([k][gemmNr]).
	// Allocated once per thin-matrix call and reused across row iterations.
	bpRow := make([]float32, k*gemmNr)
	for i := 0; i < m; i++ {
		for j := 0; j < nFull; j += gemmNr {
			// Pack one column tile of B into bpRow.
			jt := j
			for kk := 0; kk < k; kk++ {
				copy(bpRow[kk*gemmNr:kk*gemmNr+gemmNr], b[kk*n+jt:kk*n+jt+gemmNr])
			}
			gemmMicroKernel1x8NEON(c[i*n+j:], a[i*k:], bpRow, k)
		}
	}
}

// gemmPackedNEONF32 runs the packed 4×8 path for the full gemmMr×gemmNr tile
// region (m >= gemmMr, n >= gemmNr). A and B are packed once into pooled scratch;
// column tiles are processed outermost so each packed B panel stays L2-resident
// across all row blocks that reuse it.
func gemmPackedNEONF32(c, a, b []float32, m, k, n, mFull, nFull int, sc *gemmScratch) {
	nTiles := nFull / gemmNr
	nBlocks := mFull / gemmMr

	bp := ensureCap(&sc.bp, nTiles*k*gemmNr)
	ap := ensureCap(&sc.ap, nBlocks*k*gemmMr)

	packB8(bp, b, k, n, nTiles)
	packA4(ap, a, k, nBlocks)

	for t := 0; t < nTiles; t++ {
		jt := t * gemmNr
		bpt := bp[t*k*gemmNr:]
		for bi := 0; bi < nBlocks; bi++ {
			gemmMicroKernel4x8NEON(c[bi*gemmMr*n+jt:], ap[bi*k*gemmMr:], bpt, k, n)
		}
		// Remainder rows [mFull, m): one per call, reusing the packed B panel.
		for i := mFull; i < m; i++ {
			gemmMicroKernel1x8NEON(c[i*n+jt:], a[i*k:], bpt, k)
		}
	}
}

// gemmTailNEONF32 computes the n%gemmNr column remainder [nFull, n) for all rows.
// The nrem (< gemmNr) tail columns of B are packed into a zero-padded [k][gemmNr]
// panel and run through the 1x8 kernel; only the nrem valid columns are stored.
func gemmTailNEONF32(c, a, b []float32, m, k, n, nFull, nrem int, sc *gemmScratch) {
	bt := ensureCap(&sc.bt, k*gemmNr)
	packTailB(bt, b, k, n, nFull, nrem)

	var scratch [gemmNr]float32
	for i := 0; i < m; i++ {
		gemmMicroKernel1x8NEON(scratch[:], a[i*k:], bt, k)
		copy(c[i*n+nFull:i*n+n], scratch[:nrem])
	}
}

// packB8 copies the full gemmNr-wide column tiles of B[k,n] into bp laid out as
// [nTiles][k][gemmNr] contiguous, so the micro-kernel reads each panel sequentially
// (stride gemmNr) instead of with B's column stride n.
func packB8(bp, b []float32, k, n, nTiles int) {
	for t := 0; t < nTiles; t++ {
		jt := t * gemmNr
		dst := bp[t*k*gemmNr:]
		for kk := 0; kk < k; kk++ {
			copy(dst[kk*gemmNr:kk*gemmNr+gemmNr], b[kk*n+jt:kk*n+jt+gemmNr])
		}
	}
}

// packTailB packs the nrem (< gemmNr) tail columns [nFull, n) of B[k,n] into bt as a
// contiguous [k][gemmNr] panel, zero-filling the unused columns so the 1×8 kernel
// reads a full 8-wide row. This mirrors the amd64 packTailB but uses gemmNr==8.
func packTailB(bt, b []float32, k, n, nFull, nrem int) {
	for kk := 0; kk < k; kk++ {
		d := bt[kk*gemmNr : kk*gemmNr+gemmNr : kk*gemmNr+gemmNr]
		copy(d[:nrem], b[kk*n+nFull:kk*n+nFull+nrem])
		for j := nrem; j < gemmNr; j++ {
			d[j] = 0
		}
	}
}

// packA4 copies the full gemmMr-tall row blocks of A[m,k] into ap laid out as
// [nBlocks][k][gemmMr] contiguous (transposed within each block), so the 4×8 kernel
// reads the gemmMr A values for a given k as one contiguous group.
//
// Specialized to gemmMr==4: the four source rows are sliced upfront so the inner
// loop carries a single bounds check per k instead of eight.
func packA4(ap, a []float32, k, nBlocks int) {
	for bi := 0; bi < nBlocks; bi++ {
		base := bi * gemmMr * k
		dst := ap[base : base+gemmMr*k]
		r0 := a[base+0*k : base+1*k]
		r1 := a[base+1*k : base+2*k]
		r2 := a[base+2*k : base+3*k]
		r3 := a[base+3*k : base+4*k]
		for kk := 0; kk < k; kk++ {
			d := dst[kk*gemmMr : kk*gemmMr+gemmMr : kk*gemmMr+gemmMr]
			d[0], d[1] = r0[kk], r1[kk]
			d[2], d[3] = r2[kk], r3[kk]
		}
	}
}
