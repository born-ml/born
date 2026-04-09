package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// BatchMatMul performs batched matrix multiplication with numpy-style broadcasting.
// Supports tensors with 2 or more dimensions. At least one input must be 3D or higher.
//
// The last two dimensions are treated as matrix dimensions: A: (..., M, K), B: (..., K, N)
// Output: (..., M, N). The inner dimension K must match exactly.
// Batch dimensions are broadcast following numpy rules (dimensions compatible if equal or one is 1).
//
// Examples:
//
//	[B, M, K] @ [B, K, N]       -> [B, M, N]      (no broadcast)
//	[1, M, K] @ [B, K, N]       -> [B, M, N]      (singleton broadcast)
//	[M, K]    @ [B, K, N]       -> [B, M, N]      (2D broadcast to batch)
//	[A, 1, M, K] @ [1, C, K, N] -> [A, C, M, N]  (multi-dim broadcast)
func (cpu *CPUBackend) BatchMatMul(a, b *tensor.RawTensor) *tensor.RawTensor {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 3 && len(bShape) < 3 {
		panic(fmt.Sprintf("BatchMatMul: at least one of the inputs must be > 2D, got %dD and %dD", len(aShape), len(bShape)))
	}

	outShape, needsBroadcast, err := tensor.BroadcastShapesMatMul(aShape, bShape)
	if err != nil {
		panic(fmt.Sprintf("BatchMatMul: %v", err))
	}

	result, err := tensor.NewRaw(outShape, a.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("BatchMatMul: failed to create result tensor: %v", err))
	}

	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-1]

	if !needsBroadcast {
		batchSize := 1
		for i := 0; i < len(outShape)-2; i++ {
			batchSize *= outShape[i]
		}
		batchMatmul(result, a, b, batchSize, m, k, n)
	} else {
		aBatchShape := aShape[:len(aShape)-2]
		bBatchShape := bShape[:len(bShape)-2]
		outBatchShape := outShape[:len(outShape)-2]
		batchMatmulBroadcast(result, a, b, outBatchShape, aBatchShape, bBatchShape, m, k, n)
	}

	return result
}

// batchMatmul performs batched matrix multiplication.
func batchMatmul(result, a, b *tensor.RawTensor, batchSize, m, k, n int) {
	switch a.DType() {
	case tensor.Float32:
		batchMatmulFloat32(result.AsFloat32(), a.AsFloat32(), b.AsFloat32(), batchSize, m, k, n)
	case tensor.Float64:
		batchMatmulFloat64(result.AsFloat64(), a.AsFloat64(), b.AsFloat64(), batchSize, m, k, n)
	default:
		panic(fmt.Sprintf("BatchMatMul: unsupported dtype %s", a.DType()))
	}
}

// batchMatmulBroadcast performs batched matrix multiplication with broadcast.
func batchMatmulBroadcast(
	result, a, b *tensor.RawTensor,
	outBatchShape, aBatchShape, bBatchShape tensor.Shape,
	m, k, n int,
) {
	switch a.DType() {
	case tensor.Float32:
		batchMatmulBroadcastFloat32(
			result.AsFloat32(), a.AsFloat32(), b.AsFloat32(),
			outBatchShape, aBatchShape, bBatchShape, m, k, n,
		)
	case tensor.Float64:
		batchMatmulBroadcastFloat64(
			result.AsFloat64(), a.AsFloat64(), b.AsFloat64(),
			outBatchShape, aBatchShape, bBatchShape, m, k, n,
		)
	default:
		panic(fmt.Sprintf("BatchMatMul: unsupported dtype %s", a.DType()))
	}
}

// batchMatmulFloat32 performs batched matrix multiplication for float32.
func batchMatmulFloat32(c, a, b []float32, batchSize, m, k, n int) {
	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batch := range batchSize {
		aOffset := batch * matrixSizeA
		bOffset := batch * matrixSizeB
		cOffset := batch * matrixSizeC

		matmulFloat32(c[cOffset:], a[aOffset:], b[bOffset:], m, k, n)
	}
}

// batchMatmulFloat64 performs batched matrix multiplication for float64.
func batchMatmulFloat64(c, a, b []float64, batchSize, m, k, n int) {
	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batch := range batchSize {
		aOffset := batch * matrixSizeA
		bOffset := batch * matrixSizeB
		cOffset := batch * matrixSizeC

		matmulFloat64(c[cOffset:], a[aOffset:], b[bOffset:], m, k, n)
	}
}

// batchMatmulBroadcastFloat32 performs batched matrix multiplication for float32 with broadcast.
func batchMatmulBroadcastFloat32(
	c, a, b []float32,
	outBatchShape, aBatchShape, bBatchShape tensor.Shape,
	m, k, n int,
) {
	outBatchStrides := outBatchShape.ComputeStrides()
	aBroadcastStrides := computeBroadcastStridesForShape(aBatchShape, outBatchShape)
	bBroadcastStrides := computeBroadcastStridesForShape(bBatchShape, outBatchShape)

	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batchIdx := range outBatchShape.NumElements() {
		aBatchFlat := computeFlatIndex(batchIdx, outBatchStrides, aBroadcastStrides)
		bBatchFlat := computeFlatIndex(batchIdx, outBatchStrides, bBroadcastStrides)

		aOffset := aBatchFlat * matrixSizeA
		bOffset := bBatchFlat * matrixSizeB
		cOffset := batchIdx * matrixSizeC

		matmulFloat32(c[cOffset:], a[aOffset:], b[bOffset:], m, k, n)
	}
}

// batchMatmulBroadcastFloat64 performs batched matrix multiplication for float64 with broadcast.
func batchMatmulBroadcastFloat64(
	c, a, b []float64,
	outBatchShape, aBatchShape, bBatchShape tensor.Shape,
	m, k, n int,
) {
	outBatchStrides := outBatchShape.ComputeStrides()
	aBroadcastStrides := computeBroadcastStridesForShape(aBatchShape, outBatchShape)
	bBroadcastStrides := computeBroadcastStridesForShape(bBatchShape, outBatchShape)

	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batchIdx := range outBatchShape.NumElements() {
		aBatchFlat := computeFlatIndex(batchIdx, outBatchStrides, aBroadcastStrides)
		bBatchFlat := computeFlatIndex(batchIdx, outBatchStrides, bBroadcastStrides)

		aOffset := aBatchFlat * matrixSizeA
		bOffset := bBatchFlat * matrixSizeB
		cOffset := batchIdx * matrixSizeC

		matmulFloat64(c[cOffset:], a[aOffset:], b[bOffset:], m, k, n)
	}
}
