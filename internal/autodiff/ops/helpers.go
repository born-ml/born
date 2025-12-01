package ops

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// reduceBroadcast reduces a gradient tensor to match the target shape.
// This is necessary when broadcasting was used in the forward pass.
//
// Example:
//
//	Forward: a[3,1] + b[3,4] -> c[3,4]  (a was broadcast along dim 1)
//	Backward: grad_c[3,4] -> grad_a[3,1] (sum along dim 1)
func reduceBroadcast(grad *tensor.RawTensor, targetShape tensor.Shape, backend tensor.Backend) *tensor.RawTensor {
	gradShape := grad.Shape()

	// If shapes already match, clone to avoid aliasing issues
	// (prevents inplace operations from modifying shared gradients)
	if gradShape.Equal(targetShape) {
		return grad.Clone()
	}

	// Handle scalar upstream gradient (empty shape)
	// This happens when gradient flows from scalar loss (e.g., CrossEntropy, MSE)
	// We need to broadcast scalar to target shape
	if len(gradShape) == 0 {
		// Extract scalar value
		var scalarValue float64
		switch grad.DType() {
		case tensor.Float32:
			scalarValue = float64(grad.AsFloat32()[0])
		case tensor.Float64:
			scalarValue = grad.AsFloat64()[0]
		default:
			panic(fmt.Sprintf("reduceBroadcast: unsupported dtype %s for scalar gradient", grad.DType()))
		}

		// Create tensor filled with scalar value in target shape
		return createScalar(targetShape, grad.DType(), scalarValue, grad.Device())
	}

	// Handle scalar target (empty shape)
	if len(targetShape) == 0 {
		// Sum all elements to scalar
		return sumAll(grad, backend)
	}

	// Handle broadcasting reduction
	// NumPy broadcasting aligns shapes from the right
	gradDims := len(gradShape)
	targetDims := len(targetShape)

	// If target has fewer dimensions, sum leading dimensions
	if targetDims < gradDims {
		dimsToSum := gradDims - targetDims
		result := grad
		for i := 0; i < dimsToSum; i++ {
			result = sumAlongDimension(result, 0, backend)
		}
		grad = result
		gradShape = grad.Shape()
	}

	// Now sum along dimensions where target is 1
	result := grad
	for i := 0; i < targetDims; i++ {
		//nolint:gosec // i is bounded by targetDims = len(targetShape), so targetShape[i] is always valid
		if targetShape[i] == 1 && gradShape[i] > 1 {
			result = sumAlongDimension(result, i, backend)
		}
	}

	// Reshape if necessary to match target shape exactly
	if !result.Shape().Equal(targetShape) {
		result = backend.Reshape(result, targetShape)
	}

	return result
}

// sumAll sums all elements of a tensor to a scalar.
func sumAll(t *tensor.RawTensor, _ tensor.Backend) *tensor.RawTensor {
	result, err := tensor.NewRaw(tensor.Shape{}, t.DType(), t.Device())
	if err != nil {
		panic(fmt.Sprintf("sumAll: failed to create result: %v", err))
	}

	switch t.DType() {
	case tensor.Float32:
		data := t.AsFloat32()
		var sum float32
		for _, v := range data {
			sum += v
		}
		result.AsFloat32()[0] = sum

	case tensor.Float64:
		data := t.AsFloat64()
		var sum float64
		for _, v := range data {
			sum += v
		}
		result.AsFloat64()[0] = sum

	default:
		panic(fmt.Sprintf("sumAll: unsupported dtype %s", t.DType()))
	}

	return result
}

// sumAlongDimension sums a tensor along the specified dimension.
func sumAlongDimension(t *tensor.RawTensor, dim int, _ tensor.Backend) *tensor.RawTensor {
	shape := t.Shape()
	if dim < 0 || dim >= len(shape) {
		panic(fmt.Sprintf("sumAlongDimension: invalid dimension %d for shape %v", dim, shape))
	}

	// Calculate output shape (dimension at 'dim' becomes 1)
	outShape := shape.Clone()
	outShape[dim] = 1

	result, err := tensor.NewRaw(outShape, t.DType(), t.Device())
	if err != nil {
		panic(fmt.Sprintf("sumAlongDimension: failed to create result: %v", err))
	}

	// Perform sum based on dtype
	switch t.DType() {
	case tensor.Float32:
		sumFloat32AlongDimension(t.AsFloat32(), result.AsFloat32(), shape, dim)
	case tensor.Float64:
		sumFloat64AlongDimension(t.AsFloat64(), result.AsFloat64(), shape, dim)
	default:
		panic(fmt.Sprintf("sumAlongDimension: unsupported dtype %s", t.DType()))
	}

	return result
}

// sumFloat32AlongDimension sums float32 data along a dimension.
func sumFloat32AlongDimension(data, result []float32, shape tensor.Shape, dim int) {
	// Initialize result to zero
	for i := range result {
		result[i] = 0
	}

	// Calculate strides
	strides := shape.ComputeStrides()
	dimStride := strides[dim]

	// Iterate over all elements and accumulate into result
	numElements := shape.NumElements()
	for i := 0; i < numElements; i++ {
		// Calculate which index in the reduced tensor this corresponds to
		reducedIdx := 0
		temp := i
		for d := len(shape) - 1; d >= 0; d-- {
			coord := temp / strides[d]
			temp %= strides[d]

			if d != dim {
				// Include this coordinate in the reduced index
				reducedStride := 1
				for dd := d + 1; dd < len(shape); dd++ {
					if dd != dim {
						reducedStride *= shape[dd]
					}
				}
				reducedIdx += coord * reducedStride
			}
		}

		// Clamp reducedIdx to valid range
		if reducedIdx >= len(result) {
			reducedIdx = i / dimStride % (len(result))
		}

		result[reducedIdx] += data[i]
	}
}

// sumFloat64AlongDimension sums float64 data along a dimension.
func sumFloat64AlongDimension(data, result []float64, shape tensor.Shape, dim int) {
	// Initialize result to zero
	for i := range result {
		result[i] = 0
	}

	// Calculate strides
	strides := shape.ComputeStrides()
	dimStride := strides[dim]

	// Iterate over all elements and accumulate into result
	numElements := shape.NumElements()
	for i := 0; i < numElements; i++ {
		// Calculate which index in the reduced tensor this corresponds to
		reducedIdx := 0
		temp := i
		for d := len(shape) - 1; d >= 0; d-- {
			coord := temp / strides[d]
			temp %= strides[d]

			if d != dim {
				// Include this coordinate in the reduced index
				reducedStride := 1
				for dd := d + 1; dd < len(shape); dd++ {
					if dd != dim {
						reducedStride *= shape[dd]
					}
				}
				reducedIdx += coord * reducedStride
			}
		}

		// Clamp reducedIdx to valid range
		if reducedIdx >= len(result) {
			reducedIdx = i / dimStride % (len(result))
		}

		result[reducedIdx] += data[i]
	}
}

// negateGradient returns -grad.
func negateGradient(grad *tensor.RawTensor, backend tensor.Backend) *tensor.RawTensor {
	// Create a tensor of zeros with same shape
	zeros, err := tensor.NewRaw(grad.Shape(), grad.DType(), backend.Device())
	if err != nil {
		panic(fmt.Sprintf("negateGradient: failed to create zeros: %v", err))
	}

	// Initialize zeros
	switch grad.DType() {
	case tensor.Float32:
		data := zeros.AsFloat32()
		for i := range data {
			data[i] = 0
		}
	case tensor.Float64:
		data := zeros.AsFloat64()
		for i := range data {
			data[i] = 0
		}
	}

	// Return 0 - grad
	return backend.Sub(zeros, grad)
}

// createScalar creates a tensor filled with a scalar value.
func createScalar(shape tensor.Shape, dtype tensor.DataType, value float64, device tensor.Device) *tensor.RawTensor {
	result, err := tensor.NewRaw(shape, dtype, device)
	if err != nil {
		panic(fmt.Sprintf("createScalar: %v", err))
	}

	switch dtype {
	case tensor.Float32:
		data := result.AsFloat32()
		val := float32(value)
		for i := range data {
			data[i] = val
		}
	case tensor.Float64:
		data := result.AsFloat64()
		for i := range data {
			data[i] = value
		}
	default:
		panic(fmt.Sprintf("createScalar: unsupported dtype %s", dtype))
	}

	return result
}
