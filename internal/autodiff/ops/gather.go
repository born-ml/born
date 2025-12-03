package ops

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// GatherOp represents a gather operation that selects elements along a dimension.
//
// Forward: output = Gather(input, dim, index)
//
// Backward:
//
//	Scatter-add gradOutput to gradInput at positions specified by index.
//	gradInput is initialized to zeros and gradients are accumulated at indexed positions.
//
// Example:
//
//	input: [10, 20, 30, 40]
//	index: [2, 0, 3] along dim=0
//	output: [30, 10, 40]
//	gradOutput: [dL/d30, dL/d10, dL/d40]
//	gradInput: [dL/d10, 0, dL/d30, dL/d40]  (scattered back to original positions)
type GatherOp struct {
	input  *tensor.RawTensor // Input tensor
	dim    int               // Dimension along which gather happened
	index  *tensor.RawTensor // Index tensor (int32)
	output *tensor.RawTensor // Gathered output tensor
}

// NewGatherOp creates a new gather operation.
func NewGatherOp(input *tensor.RawTensor, dim int, index, output *tensor.RawTensor) *GatherOp {
	return &GatherOp{
		input:  input,
		dim:    dim,
		index:  index,
		output: output,
	}
}

// Inputs returns the input tensor.
// Note: index tensor doesn't need gradient.
func (op *GatherOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *GatherOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes gradients for the input tensor.
//
// Gradient computation:
//   - Create gradInput (same shape as input) initialized to zeros
//   - For each position in output, scatter-add gradOutput to gradInput[index]
//   - Multiple indices pointing to the same position accumulate gradients
//
// Algorithm:
//  1. Create zero-initialized gradInput with input shape
//  2. For each element in gradOutput:
//     - Read corresponding index value
//     - Add gradOutput element to gradInput at indexed position
//  3. Return gradInput
func (op *GatherOp) Backward(gradOutput *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	inputShape := op.input.Shape()
	ndim := len(inputShape)

	// Normalize dimension
	dim := op.dim
	if dim < 0 {
		dim = ndim + dim
	}

	// Create gradient for input (initialized to zero)
	gradInput, err := tensor.NewRaw(inputShape, gradOutput.DType(), backend.Device())
	if err != nil {
		panic(err)
	}

	// Zero-initialize gradient
	zeroInitialize(gradInput)

	// Scatter-add gradients
	scatterAddAlongDim(gradInput, gradOutput, op.index, dim)

	return []*tensor.RawTensor{gradInput}
}

// zeroInitialize initializes a tensor to zeros.
func zeroInitialize(t *tensor.RawTensor) {
	switch t.DType() {
	case tensor.Float32:
		data := t.AsFloat32()
		for i := range data {
			data[i] = 0
		}
	case tensor.Float64:
		data := t.AsFloat64()
		for i := range data {
			data[i] = 0
		}
	case tensor.Int32:
		data := t.AsInt32()
		for i := range data {
			data[i] = 0
		}
	case tensor.Int64:
		data := t.AsInt64()
		for i := range data {
			data[i] = 0
		}
	case tensor.Uint8:
		data := t.AsUint8()
		for i := range data {
			data[i] = 0
		}
	case tensor.Bool:
		data := t.AsBool()
		for i := range data {
			data[i] = false
		}
	default:
		panic("zeroInitialize: unsupported dtype")
	}
}

// scatterAddAlongDim scatter-adds values from src to dst at positions specified by index.
//
// Parameters:
//   - dst: destination tensor (gradInput)
//   - src: source tensor (gradOutput)
//   - index: index tensor specifying where to scatter
//   - dim: dimension along which to scatter
//
//nolint:cyclop // Complex but readable scatter operation with dtype handling
func scatterAddAlongDim(dst, src, index *tensor.RawTensor, dim int) {
	srcShape := src.Shape()
	dstShape := dst.Shape()
	indexShape := index.Shape()

	// Validate shapes match (except at dim)
	ndim := len(srcShape)
	for d := 0; d < ndim; d++ {
		if d != dim && srcShape[d] != indexShape[d] {
			panic("scatterAddAlongDim: src and index shapes must match except at dim")
		}
	}

	// Compute strides
	srcStrides := srcShape.ComputeStrides()
	dstStrides := dstShape.ComputeStrides()
	indexStrides := indexShape.ComputeStrides()

	numElements := src.NumElements()

	// Convert indices to int32 regardless of original dtype
	var indices []int32
	switch index.DType() {
	case tensor.Int32:
		indices = index.AsInt32()
	case tensor.Int64:
		// Convert int64 to int32
		src64 := index.AsInt64()
		indices = make([]int32, len(src64))
		for i, v := range src64 {
			indices[i] = int32(v)
		}
	case tensor.Float32:
		// Convert float32 to int32 (for boolean/comparison results)
		srcF := index.AsFloat32()
		indices = make([]int32, len(srcF))
		for i, v := range srcF {
			indices[i] = int32(v)
		}
	default:
		panic("scatterAddAlongDim: index must be int32, int64, or float32")
	}

	// Handle different data types
	switch src.DType() {
	case tensor.Float32:
		scatterAddFloat32(dst.AsFloat32(), src.AsFloat32(), indices, dim, numElements, srcShape, dstShape, srcStrides, dstStrides, indexStrides)
	case tensor.Float64:
		scatterAddFloat64(dst.AsFloat64(), src.AsFloat64(), indices, dim, numElements, srcShape, dstShape, srcStrides, dstStrides, indexStrides)
	case tensor.Int32:
		scatterAddInt32(dst.AsInt32(), src.AsInt32(), indices, dim, numElements, srcShape, dstShape, srcStrides, dstStrides, indexStrides)
	case tensor.Int64:
		scatterAddInt64(dst.AsInt64(), src.AsInt64(), indices, dim, numElements, srcShape, dstShape, srcStrides, dstStrides, indexStrides)
	case tensor.Uint8:
		scatterAddUint8(dst.AsUint8(), src.AsUint8(), indices, dim, numElements, srcShape, dstShape, srcStrides, dstStrides, indexStrides)
	default:
		panic("scatterAddAlongDim: unsupported dtype")
	}
}

// scatterAddFloat32 scatter-adds float32 values.
func scatterAddFloat32(dst, src []float32, indices []int32, dim, numElements int, srcShape, dstShape tensor.Shape, srcStrides, dstStrides, indexStrides []int) {
	for i := 0; i < numElements; i++ {
		// Compute multi-dimensional coordinates for src
		temp := i
		coords := make([]int, len(srcShape))
		for d := 0; d < len(srcShape); d++ {
			coords[d] = temp / srcStrides[d]
			temp %= srcStrides[d]
		}

		// Compute index position
		indexIdx := 0
		for d := 0; d < len(srcShape); d++ {
			indexIdx += coords[d] * indexStrides[d]
		}

		// Read index value
		idx := int(indices[indexIdx])

		// Validate index
		if idx < 0 || idx >= dstShape[dim] {
			panic(fmt.Sprintf("scatterAddAlongDim: index out of bounds: idx=%d, dstShape[%d]=%d, srcShape=%v, dstShape=%v",
				idx, dim, dstShape[dim], srcShape, dstShape))
		}

		// Compute destination index by replacing coord[dim] with idx
		dstIdx := 0
		for d := 0; d < len(srcShape); d++ {
			if d == dim {
				dstIdx += idx * dstStrides[d]
			} else {
				dstIdx += coords[d] * dstStrides[d]
			}
		}

		// Accumulate gradient
		dst[dstIdx] += src[i]
	}
}

// scatterAddFloat64 scatter-adds float64 values.
func scatterAddFloat64(dst, src []float64, indices []int32, dim, numElements int, srcShape, dstShape tensor.Shape, srcStrides, dstStrides, indexStrides []int) {
	for i := 0; i < numElements; i++ {
		temp := i
		coords := make([]int, len(srcShape))
		for d := 0; d < len(srcShape); d++ {
			coords[d] = temp / srcStrides[d]
			temp %= srcStrides[d]
		}

		indexIdx := 0
		for d := 0; d < len(srcShape); d++ {
			indexIdx += coords[d] * indexStrides[d]
		}

		idx := int(indices[indexIdx])
		if idx < 0 || idx >= dstShape[dim] {
			panic("scatterAddAlongDim: index out of bounds")
		}

		dstIdx := 0
		for d := 0; d < len(srcShape); d++ {
			if d == dim {
				dstIdx += idx * dstStrides[d]
			} else {
				dstIdx += coords[d] * dstStrides[d]
			}
		}

		dst[dstIdx] += src[i]
	}
}

// scatterAddInt32 scatter-adds int32 values.
func scatterAddInt32(dst, src, indices []int32, dim, numElements int, srcShape, dstShape tensor.Shape, srcStrides, dstStrides, indexStrides []int) {
	for i := 0; i < numElements; i++ {
		temp := i
		coords := make([]int, len(srcShape))
		for d := 0; d < len(srcShape); d++ {
			coords[d] = temp / srcStrides[d]
			temp %= srcStrides[d]
		}

		indexIdx := 0
		for d := 0; d < len(srcShape); d++ {
			indexIdx += coords[d] * indexStrides[d]
		}

		idx := int(indices[indexIdx])
		if idx < 0 || idx >= dstShape[dim] {
			panic("scatterAddAlongDim: index out of bounds")
		}

		dstIdx := 0
		for d := 0; d < len(srcShape); d++ {
			if d == dim {
				dstIdx += idx * dstStrides[d]
			} else {
				dstIdx += coords[d] * dstStrides[d]
			}
		}

		dst[dstIdx] += src[i]
	}
}

// scatterAddInt64 scatter-adds int64 values.
func scatterAddInt64(dst, src []int64, indices []int32, dim, numElements int, srcShape, dstShape tensor.Shape, srcStrides, dstStrides, indexStrides []int) {
	for i := 0; i < numElements; i++ {
		temp := i
		coords := make([]int, len(srcShape))
		for d := 0; d < len(srcShape); d++ {
			coords[d] = temp / srcStrides[d]
			temp %= srcStrides[d]
		}

		indexIdx := 0
		for d := 0; d < len(srcShape); d++ {
			indexIdx += coords[d] * indexStrides[d]
		}

		idx := int(indices[indexIdx])
		if idx < 0 || idx >= dstShape[dim] {
			panic("scatterAddAlongDim: index out of bounds")
		}

		dstIdx := 0
		for d := 0; d < len(srcShape); d++ {
			if d == dim {
				dstIdx += idx * dstStrides[d]
			} else {
				dstIdx += coords[d] * dstStrides[d]
			}
		}

		dst[dstIdx] += src[i]
	}
}

// scatterAddUint8 scatter-adds uint8 values.
func scatterAddUint8(dst, src []uint8, indices []int32, dim, numElements int, srcShape, dstShape tensor.Shape, srcStrides, dstStrides, indexStrides []int) {
	for i := 0; i < numElements; i++ {
		temp := i
		coords := make([]int, len(srcShape))
		for d := 0; d < len(srcShape); d++ {
			coords[d] = temp / srcStrides[d]
			temp %= srcStrides[d]
		}

		indexIdx := 0
		for d := 0; d < len(srcShape); d++ {
			indexIdx += coords[d] * indexStrides[d]
		}

		idx := int(indices[indexIdx])
		if idx < 0 || idx >= dstShape[dim] {
			panic("scatterAddAlongDim: index out of bounds")
		}

		dstIdx := 0
		for d := 0; d < len(srcShape); d++ {
			if d == dim {
				dstIdx += idx * dstStrides[d]
			} else {
				dstIdx += coords[d] * dstStrides[d]
			}
		}

		dst[dstIdx] += src[i]
	}
}
