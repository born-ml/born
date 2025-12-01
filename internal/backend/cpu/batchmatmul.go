package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// BatchMatMul performs batched matrix multiplication.
// Supports 3D and 4D tensors with batch dimensions.
//
// For 3D: [B, M, K] @ [B, K, N] -> [B, M, N]
// For 4D: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
//
// The last two dimensions are treated as matrix dimensions.
// All leading dimensions must match (batch dimensions).
func (cpu *CPUBackend) BatchMatMul(a, b *tensor.RawTensor) *tensor.RawTensor {
	aShape := a.Shape()
	bShape := b.Shape()
	ndim := len(aShape)

	// Validate dimensions
	if ndim < 3 {
		panic(fmt.Sprintf("BatchMatMul: inputs must be at least 3D, got %dD", ndim))
	}
	if len(bShape) != ndim {
		panic(fmt.Sprintf("BatchMatMul: dimension mismatch, got %dD and %dD", ndim, len(bShape)))
	}

	// Validate batch dimensions match
	for i := 0; i < ndim-2; i++ {
		if aShape[i] != bShape[i] {
			panic(fmt.Sprintf("BatchMatMul: batch dimension mismatch at dim %d: %d vs %d", i, aShape[i], bShape[i]))
		}
	}

	// Extract matrix dimensions
	m := aShape[ndim-2]
	k1 := aShape[ndim-1]
	k2 := bShape[ndim-2]
	n := bShape[ndim-1]

	if k1 != k2 {
		panic(fmt.Sprintf("BatchMatMul: inner dimension mismatch: %d vs %d", k1, k2))
	}

	// Compute batch size (product of all batch dims)
	batchSize := 1
	for i := 0; i < ndim-2; i++ {
		batchSize *= aShape[i]
	}

	// Output shape = batch dims + [M, N]
	outShape := make(tensor.Shape, ndim)
	copy(outShape, aShape[:ndim-2])
	outShape[ndim-2] = m
	outShape[ndim-1] = n

	// Create result tensor
	result, err := tensor.NewRaw(outShape, a.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("BatchMatMul: failed to create result tensor: %v", err))
	}

	// Dispatch to type-specific implementation
	switch a.DType() {
	case tensor.Float32:
		batchMatmulFloat32(result.AsFloat32(), a.AsFloat32(), b.AsFloat32(), batchSize, m, k1, n)
	case tensor.Float64:
		batchMatmulFloat64(result.AsFloat64(), a.AsFloat64(), b.AsFloat64(), batchSize, m, k1, n)
	default:
		panic(fmt.Sprintf("BatchMatMul: unsupported dtype %s", a.DType()))
	}

	return result
}

// batchMatmulFloat32 performs batched matrix multiplication for float32.
func batchMatmulFloat32(c, a, b []float32, batchSize, m, k, n int) {
	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batch := 0; batch < batchSize; batch++ {
		aOffset := batch * matrixSizeA
		bOffset := batch * matrixSizeB
		cOffset := batch * matrixSizeC

		// 2D matmul for this batch
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for kIdx := 0; kIdx < k; kIdx++ {
					sum += a[aOffset+i*k+kIdx] * b[bOffset+kIdx*n+j]
				}
				c[cOffset+i*n+j] = sum
			}
		}
	}
}

// batchMatmulFloat64 performs batched matrix multiplication for float64.
func batchMatmulFloat64(c, a, b []float64, batchSize, m, k, n int) {
	matrixSizeA := m * k
	matrixSizeB := k * n
	matrixSizeC := m * n

	for batch := 0; batch < batchSize; batch++ {
		aOffset := batch * matrixSizeA
		bOffset := batch * matrixSizeB
		cOffset := batch * matrixSizeC

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float64(0)
				for kIdx := 0; kIdx < k; kIdx++ {
					sum += a[aOffset+i*k+kIdx] * b[bOffset+kIdx*n+j]
				}
				c[cOffset+i*n+j] = sum
			}
		}
	}
}
