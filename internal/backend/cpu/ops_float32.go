package cpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Float32 inplace operations

func addInplaceFloat32(a, b []float32) {
	if simdAddInplaceFloat32 != nil {
		simdAddInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] += b[i]
	}
}

func subInplaceFloat32(a, b []float32) {
	if simdSubInplaceFloat32 != nil {
		simdSubInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] -= b[i]
	}
}

func mulInplaceFloat32(a, b []float32) {
	if simdMulInplaceFloat32 != nil {
		simdMulInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] *= b[i]
	}
}

func divInplaceFloat32(a, b []float32) {
	if simdDivInplaceFloat32 != nil {
		simdDivInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] /= b[i]
	}
}

// Float32 vectorized operations

func addVectorizedFloat32(dst, a, b []float32) {
	if simdAddVectorizedFloat32 != nil {
		simdAddVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func subVectorizedFloat32(dst, a, b []float32) {
	if simdSubVectorizedFloat32 != nil {
		simdSubVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func mulVectorizedFloat32(dst, a, b []float32) {
	if simdMulVectorizedFloat32 != nil {
		simdMulVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func divVectorizedFloat32(dst, a, b []float32) {
	if simdDivVectorizedFloat32 != nil {
		simdDivVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] / b[i]
	}
}

// Float32 broadcasting operations

func addBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	// Compute indices for broadcasting
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] + b[bIdx]
	}
}

func subBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] - b[bIdx]
	}
}

func mulBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] * b[bIdx]
	}
}

func divBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] / b[bIdx]
	}
}

// Transpose float32.
func transposeFloat32(dst, src []float32, shape tensor.Shape, axes []int) {
	ndim := len(shape)
	srcStrides := shape.ComputeStrides()

	// Compute destination shape and strides
	dstShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		dstShape[i] = shape[ax]
	}
	dstStrides := dstShape.ComputeStrides()

	// Transpose data
	n := shape.NumElements()
	for i := 0; i < n; i++ {
		// Compute multi-dimensional coordinates in source
		coords := make([]int, ndim)
		idx := i
		for dim := 0; dim < ndim; dim++ {
			coords[dim] = idx / srcStrides[dim]
			idx %= srcStrides[dim]
		}

		// Permute coordinates according to axes
		permutedCoords := make([]int, ndim)
		for dstDim, srcDim := range axes {
			permutedCoords[dstDim] = coords[srcDim]
		}

		// Compute flat index in destination
		dstIdx := 0
		for dim := 0; dim < ndim; dim++ {
			dstIdx += permutedCoords[dim] * dstStrides[dim]
		}

		dst[dstIdx] = src[i]
	}
}
