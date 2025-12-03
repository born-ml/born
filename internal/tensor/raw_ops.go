//go:build !wasm

// Package tensor raw_ops provides type-specific tensor operations for ONNX inference.
// Type-specific implementations (Float32, Float64, Int32, Int64) are intentionally
// similar/duplicated for performance - generics would add overhead.
//
//nolint:dupl // Type-specific implementations are intentionally similar for performance
package tensor

import (
	"fmt"
	"math"
)

// ReLU applies the ReLU activation function element-wise: max(x, 0).
func ReLU(x *RawTensor) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("ReLU: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("ReLU: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			if in[i] > 0 {
				out[i] = in[i]
			}
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		for i := range in {
			if in[i] > 0 {
				out[i] = in[i]
			}
		}
	default:
		return nil, fmt.Errorf("ReLU: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// LeakyReLU applies leaky ReLU: max(x, alpha*x).
func LeakyReLU(x *RawTensor, alpha float32) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("LeakyReLU: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("LeakyReLU: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			if in[i] > 0 {
				out[i] = in[i]
			} else {
				out[i] = alpha * in[i]
			}
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		a := float64(alpha)
		for i := range in {
			if in[i] > 0 {
				out[i] = in[i]
			} else {
				out[i] = a * in[i]
			}
		}
	default:
		return nil, fmt.Errorf("LeakyReLU: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// PReLU applies parametric ReLU: max(x, slope*x) where slope is per-element or broadcasted.
//
//nolint:gocognit // PReLU has inherent complexity from dtype switching
func PReLU(x, slope *RawTensor) (*RawTensor, error) {
	if x == nil || slope == nil {
		return nil, fmt.Errorf("PReLU: input tensors cannot be nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("PReLU: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		sl := slope.AsFloat32()
		// Handle broadcasting: slope can be scalar or per-channel
		slopeSingle := len(sl) == 1
		for i := range in {
			s := sl[0]
			if !slopeSingle && i < len(sl) {
				s = sl[i%len(sl)]
			}
			if in[i] > 0 {
				out[i] = in[i]
			} else {
				out[i] = s * in[i]
			}
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		sl := slope.AsFloat64()
		slopeSingle := len(sl) == 1
		for i := range in {
			s := sl[0]
			if !slopeSingle && i < len(sl) {
				s = sl[i%len(sl)]
			}
			if in[i] > 0 {
				out[i] = in[i]
			} else {
				out[i] = s * in[i]
			}
		}
	default:
		return nil, fmt.Errorf("PReLU: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// Sigmoid applies the sigmoid activation function: 1/(1+exp(-x)).
func Sigmoid(x *RawTensor) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Sigmoid: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Sigmoid: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			out[i] = float32(1.0 / (1.0 + math.Exp(float64(-in[i]))))
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		for i := range in {
			out[i] = 1.0 / (1.0 + math.Exp(-in[i]))
		}
	default:
		return nil, fmt.Errorf("Sigmoid: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// Tanh applies the hyperbolic tangent activation function.
func Tanh(x *RawTensor) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Tanh: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Tanh: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			out[i] = float32(math.Tanh(float64(in[i])))
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		for i := range in {
			out[i] = math.Tanh(in[i])
		}
	default:
		return nil, fmt.Errorf("Tanh: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// Softmax applies softmax along the specified axis.
func Softmax(x *RawTensor, axis int) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Softmax: input tensor is nil")
	}

	// Handle negative axis
	if axis < 0 {
		axis = len(x.shape) + axis
	}
	if axis < 0 || axis >= len(x.shape) {
		return nil, fmt.Errorf("Softmax: axis %d out of range for tensor with %d dimensions", axis, len(x.shape))
	}

	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Softmax: %w", err)
	}

	switch x.dtype {
	case Float32:
		softmaxFloat32(x.AsFloat32(), result.AsFloat32(), x.shape, axis)
	case Float64:
		softmaxFloat64(x.AsFloat64(), result.AsFloat64(), x.shape, axis)
	default:
		return nil, fmt.Errorf("Softmax: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

func softmaxFloat32(in, out []float32, shape Shape, axis int) {
	// Calculate strides
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	axisSize := shape[axis]
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			// Find max for numerical stability
			maxVal := float32(-math.MaxFloat32)
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				if in[idx] > maxVal {
					maxVal = in[idx]
				}
			}
			// Compute exp and sum
			sum := float32(0)
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				out[idx] = float32(math.Exp(float64(in[idx] - maxVal)))
				sum += out[idx]
			}
			// Normalize
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				out[idx] /= sum
			}
		}
	}
}

func softmaxFloat64(in, out []float64, shape Shape, axis int) {
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	axisSize := shape[axis]
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			maxVal := -math.MaxFloat64
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				if in[idx] > maxVal {
					maxVal = in[idx]
				}
			}
			sum := 0.0
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				out[idx] = math.Exp(in[idx] - maxVal)
				sum += out[idx]
			}
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				out[idx] /= sum
			}
		}
	}
}

// LogSoftmax computes log(softmax(x)) along the specified axis.
func LogSoftmax(x *RawTensor, axis int) (*RawTensor, error) {
	s, err := Softmax(x, axis)
	if err != nil {
		return nil, err
	}

	switch s.dtype {
	case Float32:
		data := s.AsFloat32()
		for i := range data {
			data[i] = float32(math.Log(float64(data[i])))
		}
	case Float64:
		data := s.AsFloat64()
		for i := range data {
			data[i] = math.Log(data[i])
		}
	}
	return s, nil
}

// GELU applies the Gaussian Error Linear Unit activation.
// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
func GELU(x *RawTensor) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("GELU: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("GELU: %w", err)
	}

	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	const coeff = 0.044715

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			v := float64(in[i])
			inner := sqrt2OverPi * (v + coeff*v*v*v)
			out[i] = float32(0.5 * v * (1 + math.Tanh(inner)))
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		for i := range in {
			v := in[i]
			inner := sqrt2OverPi * (v + coeff*v*v*v)
			out[i] = 0.5 * v * (1 + math.Tanh(inner))
		}
	default:
		return nil, fmt.Errorf("GELU: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// SiLU applies the Sigmoid Linear Unit (Swish) activation: x * sigmoid(x).
func SiLU(x *RawTensor) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("SiLU: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("SiLU: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-in[i]))))
			out[i] = in[i] * sigmoid
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		for i := range in {
			sigmoid := 1.0 / (1.0 + math.Exp(-in[i]))
			out[i] = in[i] * sigmoid
		}
	default:
		return nil, fmt.Errorf("SiLU: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// Clip clamps values to the range [min, max].
func Clip(x *RawTensor, minVal, maxVal float32) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Clip: input tensor is nil")
	}
	result, err := NewRaw(x.shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Clip: %w", err)
	}

	switch x.dtype {
	case Float32:
		in := x.AsFloat32()
		out := result.AsFloat32()
		for i := range in {
			v := in[i]
			if v < minVal {
				v = minVal
			}
			if v > maxVal {
				v = maxVal
			}
			out[i] = v
		}
	case Float64:
		in := x.AsFloat64()
		out := result.AsFloat64()
		min64, max64 := float64(minVal), float64(maxVal)
		for i := range in {
			v := in[i]
			if v < min64 {
				v = min64
			}
			if v > max64 {
				v = max64
			}
			out[i] = v
		}
	default:
		return nil, fmt.Errorf("Clip: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

// Reshape returns a new tensor with the given shape (shares data if contiguous).
func Reshape(x *RawTensor, newShape Shape) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Reshape: input tensor is nil")
	}

	// Handle -1 dimension inference
	totalElements := x.NumElements()
	inferIdx := -1
	product := 1
	for i, dim := range newShape {
		switch {
		case dim == -1:
			if inferIdx >= 0 {
				return nil, fmt.Errorf("Reshape: can only have one -1 dimension")
			}
			inferIdx = i
		case dim <= 0:
			return nil, fmt.Errorf("Reshape: dimensions must be positive, got %d", dim)
		default:
			product *= dim
		}
	}

	actualShape := make(Shape, len(newShape))
	copy(actualShape, newShape)

	if inferIdx >= 0 {
		if product == 0 || totalElements%product != 0 {
			return nil, fmt.Errorf("Reshape: cannot infer dimension for shape %v from %d elements", newShape, totalElements)
		}
		actualShape[inferIdx] = totalElements / product
	}

	// Verify total elements match
	newTotal := 1
	for _, dim := range actualShape {
		newTotal *= dim
	}
	if newTotal != totalElements {
		return nil, fmt.Errorf("Reshape: cannot reshape %d elements to shape %v (%d elements)", totalElements, actualShape, newTotal)
	}

	// Create new tensor with same data but different shape
	result := x.Clone()
	result.shape = actualShape
	result.stride = actualShape.ComputeStrides()
	return result, nil
}

// TransposeAxes transposes dimensions according to the given permutation.
func TransposeAxes(x *RawTensor, axes ...int) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("TransposeAxes: input tensor is nil")
	}

	ndim := len(x.shape)

	// Default: reverse all dimensions
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = ndim - 1 - i
		}
	}

	if len(axes) != ndim {
		return nil, fmt.Errorf("TransposeAxes: axes length %d must match tensor dimensions %d", len(axes), ndim)
	}

	// Build new shape
	newShape := make(Shape, ndim)
	for i, ax := range axes {
		if ax < 0 || ax >= ndim {
			return nil, fmt.Errorf("TransposeAxes: axis %d out of range [0, %d)", ax, ndim)
		}
		newShape[i] = x.shape[ax]
	}

	result, err := NewRaw(newShape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("TransposeAxes: %w", err)
	}

	// Transpose data
	switch x.dtype {
	case Float32:
		transposeData(x.AsFloat32(), result.AsFloat32(), x.shape, newShape, axes)
	case Float64:
		transposeDataFloat64(x.AsFloat64(), result.AsFloat64(), x.shape, newShape, axes)
	case Int32:
		transposeDataInt32(x.AsInt32(), result.AsInt32(), x.shape, newShape, axes)
	case Int64:
		transposeDataInt64(x.AsInt64(), result.AsInt64(), x.shape, newShape, axes)
	default:
		return nil, fmt.Errorf("TransposeAxes: unsupported dtype %v", x.dtype)
	}
	return result, nil
}

func transposeData(in, out []float32, oldShape, newShape Shape, axes []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		// Compute new index
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		// Compute old flat index
		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldFlat += idx[j] * oldStrides[axes[j]]
		}

		// Compute new flat index
		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

func transposeDataFloat64(in, out []float64, oldShape, newShape Shape, axes []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldFlat += idx[j] * oldStrides[axes[j]]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

func transposeDataInt32(in, out []int32, oldShape, newShape Shape, axes []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldFlat += idx[j] * oldStrides[axes[j]]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

func transposeDataInt64(in, out []int64, oldShape, newShape Shape, axes []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldFlat += idx[j] * oldStrides[axes[j]]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

// Squeeze removes dimensions of size 1 at the specified axes.
//
// If no axes are specified, removes all dimensions of size 1.
//
//nolint:nestif // Squeeze has inherent complexity with axes handling
func Squeeze(x *RawTensor, axes ...int) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Squeeze: input tensor is nil")
	}

	newShape := make(Shape, 0, len(x.shape))

	if len(axes) == 0 {
		// Remove all dimensions of size 1
		for _, dim := range x.shape {
			if dim != 1 {
				newShape = append(newShape, dim)
			}
		}
	} else {
		// Only remove specified axes
		axisSet := make(map[int]bool)
		for _, ax := range axes {
			if ax < 0 {
				ax = len(x.shape) + ax
			}
			axisSet[ax] = true
		}
		for i, dim := range x.shape {
			if axisSet[i] {
				if dim != 1 {
					return nil, fmt.Errorf("Squeeze: cannot squeeze axis %d with size %d (must be 1)", i, dim)
				}
			} else {
				newShape = append(newShape, dim)
			}
		}
	}

	if len(newShape) == 0 {
		newShape = Shape{1} // Scalar
	}

	return Reshape(x, newShape)
}

// Unsqueeze adds dimensions of size 1 at the specified axes.
func Unsqueeze(x *RawTensor, axes ...int) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Unsqueeze: input tensor is nil")
	}

	if len(axes) == 0 {
		return nil, fmt.Errorf("Unsqueeze: at least one axis required")
	}

	newNdim := len(x.shape) + len(axes)
	newShape := make(Shape, newNdim)

	// Normalize axes and sort
	normalizedAxes := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax = newNdim + ax
		}
		if ax < 0 || ax >= newNdim {
			return nil, fmt.Errorf("Unsqueeze: axis %d out of range [0, %d)", axes[i], newNdim)
		}
		normalizedAxes[i] = ax
	}

	// Mark which positions are new axes
	axisSet := make(map[int]bool)
	for _, ax := range normalizedAxes {
		axisSet[ax] = true
	}

	// Build new shape
	oldIdx := 0
	for i := 0; i < newNdim; i++ {
		if axisSet[i] {
			newShape[i] = 1
		} else {
			newShape[i] = x.shape[oldIdx]
			oldIdx++
		}
	}

	return Reshape(x, newShape)
}

// Concat concatenates tensors along the specified dimension.
//
//nolint:gocyclo,cyclop // Concat validation has inherent complexity
func Concat(tensors []*RawTensor, axis int) (*RawTensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("Concat: no tensors provided")
	}
	if len(tensors) == 1 {
		return tensors[0].Clone(), nil
	}

	first := tensors[0]
	ndim := len(first.shape)

	// Handle negative axis
	if axis < 0 {
		axis = ndim + axis
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("Concat: axis %d out of range for %d dimensions", axis, ndim)
	}

	// Verify all tensors have compatible shapes
	for i, t := range tensors[1:] {
		if len(t.shape) != ndim {
			return nil, fmt.Errorf("Concat: tensor %d has %d dimensions, expected %d", i+1, len(t.shape), ndim)
		}
		if t.dtype != first.dtype {
			return nil, fmt.Errorf("Concat: tensor %d has dtype %v, expected %v", i+1, t.dtype, first.dtype)
		}
		for j := 0; j < ndim; j++ {
			if j != axis && t.shape[j] != first.shape[j] {
				return nil, fmt.Errorf("Concat: tensor %d has shape %v, incompatible with %v on axis %d", i+1, t.shape, first.shape, axis)
			}
		}
	}

	// Compute output shape
	newShape := make(Shape, ndim)
	copy(newShape, first.shape)
	for _, t := range tensors[1:] {
		newShape[axis] += t.shape[axis]
	}

	result, err := NewRaw(newShape, first.dtype, first.device)
	if err != nil {
		return nil, fmt.Errorf("Concat: %w", err)
	}

	// Copy data
	switch first.dtype {
	case Float32:
		concatFloat32(tensors, result, axis)
	case Float64:
		concatFloat64(tensors, result, axis)
	case Int64:
		concatInt64(tensors, result, axis)
	case Int32:
		concatInt32(tensors, result, axis)
	default:
		return nil, fmt.Errorf("Concat: unsupported dtype %v", first.dtype)
	}

	return result, nil
}

func concatFloat32(tensors []*RawTensor, result *RawTensor, axis int) {
	outData := result.AsFloat32()
	outShape := result.shape

	// Calculate sizes
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(outShape); i++ {
		innerSize *= outShape[i]
	}

	offset := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, t := range tensors {
			inData := t.AsFloat32()
			axisSize := t.shape[axis]
			copyLen := axisSize * innerSize
			inStart := outer * copyLen
			copy(outData[offset:offset+copyLen], inData[inStart:inStart+copyLen])
			offset += copyLen
		}
	}
}

func concatFloat64(tensors []*RawTensor, result *RawTensor, axis int) {
	outData := result.AsFloat64()
	outShape := result.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(outShape); i++ {
		innerSize *= outShape[i]
	}

	offset := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, t := range tensors {
			inData := t.AsFloat64()
			axisSize := t.shape[axis]
			copyLen := axisSize * innerSize
			inStart := outer * copyLen
			copy(outData[offset:offset+copyLen], inData[inStart:inStart+copyLen])
			offset += copyLen
		}
	}
}

func concatInt64(tensors []*RawTensor, result *RawTensor, axis int) {
	outData := result.AsInt64()
	outShape := result.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(outShape); i++ {
		innerSize *= outShape[i]
	}

	offset := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, t := range tensors {
			inData := t.AsInt64()
			axisSize := t.shape[axis]
			copyLen := axisSize * innerSize
			inStart := outer * copyLen
			copy(outData[offset:offset+copyLen], inData[inStart:inStart+copyLen])
			offset += copyLen
		}
	}
}

func concatInt32(tensors []*RawTensor, result *RawTensor, axis int) {
	outData := result.AsInt32()
	outShape := result.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(outShape); i++ {
		innerSize *= outShape[i]
	}

	offset := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, t := range tensors {
			inData := t.AsInt32()
			axisSize := t.shape[axis]
			copyLen := axisSize * innerSize
			inStart := outer * copyLen
			copy(outData[offset:offset+copyLen], inData[inStart:inStart+copyLen])
			offset += copyLen
		}
	}
}

// Split splits a tensor into multiple tensors along an axis.
//
//nolint:cyclop // Split logic has inherent complexity
func Split(x *RawTensor, axis int, splitSizes []int) ([]*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Split: input tensor is nil")
	}

	ndim := len(x.shape)
	if axis < 0 {
		axis = ndim + axis
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("Split: axis %d out of range for %d dimensions", axis, ndim)
	}

	axisSize := x.shape[axis]

	// Default: equal splits
	if len(splitSizes) == 0 {
		// Default to splitting into chunks of size 1
		splitSizes = make([]int, axisSize)
		for i := range splitSizes {
			splitSizes[i] = 1
		}
	}

	// Verify split sizes
	total := 0
	for _, s := range splitSizes {
		total += s
	}
	if total != axisSize {
		return nil, fmt.Errorf("Split: split sizes sum to %d, but axis has size %d", total, axisSize)
	}

	results := make([]*RawTensor, len(splitSizes))
	offset := 0

	for i, size := range splitSizes {
		newShape := make(Shape, ndim)
		copy(newShape, x.shape)
		newShape[axis] = size

		result, err := NewRaw(newShape, x.dtype, x.device)
		if err != nil {
			return nil, fmt.Errorf("Split: %w", err)
		}

		// Copy data slice
		switch x.dtype {
		case Float32:
			copySliceFloat32(x, result, axis, offset, size)
		case Float64:
			copySliceFloat64(x, result, axis, offset, size)
		case Int64:
			copySliceInt64(x, result, axis, offset, size)
		case Int32:
			copySliceInt32(x, result, axis, offset, size)
		default:
			return nil, fmt.Errorf("Split: unsupported dtype %v", x.dtype)
		}

		results[i] = result
		offset += size
	}

	return results, nil
}

func copySliceFloat32(src, dst *RawTensor, axis, offset, size int) {
	srcData := src.AsFloat32()
	dstData := dst.AsFloat32()
	srcShape := src.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= srcShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(srcShape); i++ {
		innerSize *= srcShape[i]
	}
	srcAxisSize := srcShape[axis]

	dstIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		for a := 0; a < size; a++ {
			for inner := 0; inner < innerSize; inner++ {
				srcIdx := outer*srcAxisSize*innerSize + (offset+a)*innerSize + inner
				dstData[dstIdx] = srcData[srcIdx]
				dstIdx++
			}
		}
	}
}

func copySliceFloat64(src, dst *RawTensor, axis, offset, size int) {
	srcData := src.AsFloat64()
	dstData := dst.AsFloat64()
	srcShape := src.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= srcShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(srcShape); i++ {
		innerSize *= srcShape[i]
	}
	srcAxisSize := srcShape[axis]

	dstIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		for a := 0; a < size; a++ {
			for inner := 0; inner < innerSize; inner++ {
				srcIdx := outer*srcAxisSize*innerSize + (offset+a)*innerSize + inner
				dstData[dstIdx] = srcData[srcIdx]
				dstIdx++
			}
		}
	}
}

func copySliceInt64(src, dst *RawTensor, axis, offset, size int) {
	srcData := src.AsInt64()
	dstData := dst.AsInt64()
	srcShape := src.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= srcShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(srcShape); i++ {
		innerSize *= srcShape[i]
	}
	srcAxisSize := srcShape[axis]

	dstIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		for a := 0; a < size; a++ {
			for inner := 0; inner < innerSize; inner++ {
				srcIdx := outer*srcAxisSize*innerSize + (offset+a)*innerSize + inner
				dstData[dstIdx] = srcData[srcIdx]
				dstIdx++
			}
		}
	}
}

func copySliceInt32(src, dst *RawTensor, axis, offset, size int) {
	srcData := src.AsInt32()
	dstData := dst.AsInt32()
	srcShape := src.shape

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= srcShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(srcShape); i++ {
		innerSize *= srcShape[i]
	}
	srcAxisSize := srcShape[axis]

	dstIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		for a := 0; a < size; a++ {
			for inner := 0; inner < innerSize; inner++ {
				srcIdx := outer*srcAxisSize*innerSize + (offset+a)*innerSize + inner
				dstData[dstIdx] = srcData[srcIdx]
				dstIdx++
			}
		}
	}
}

// Slice extracts a slice from a tensor.
//
//nolint:gocognit,gocyclo,cyclop,funlen // Slice has inherent complexity from multi-dim indexing and requires >60 statements
func Slice(x *RawTensor, starts, ends, axes, steps []int64) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Slice: input tensor is nil")
	}

	ndim := len(x.shape)

	// Default axes to all dimensions
	if len(axes) == 0 {
		axes = make([]int64, len(starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	}

	// Default steps to 1
	if len(steps) == 0 {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}

	// Build slice parameters for each dimension
	sliceStarts := make([]int, ndim)
	sliceEnds := make([]int, ndim)
	sliceSteps := make([]int, ndim)

	for i := 0; i < ndim; i++ {
		sliceStarts[i] = 0
		sliceEnds[i] = x.shape[i]
		sliceSteps[i] = 1
	}

	for i, ax := range axes {
		axis := int(ax)
		if axis < 0 {
			axis = ndim + axis
		}
		if axis < 0 || axis >= ndim {
			return nil, fmt.Errorf("Slice: axis %d out of range [0, %d)", ax, ndim)
		}

		start := int(starts[i])
		end := int(ends[i])
		step := int(steps[i])

		// Handle negative indices
		if start < 0 {
			start = x.shape[axis] + start
		}
		if end < 0 {
			end = x.shape[axis] + end
		}

		// Clamp to valid range
		if start < 0 {
			start = 0
		}
		if start > x.shape[axis] {
			start = x.shape[axis]
		}
		if end < 0 {
			end = 0
		}
		if end > x.shape[axis] {
			end = x.shape[axis]
		}

		sliceStarts[axis] = start
		sliceEnds[axis] = end
		sliceSteps[axis] = step
	}

	// Calculate output shape
	newShape := make(Shape, ndim)
	for i := 0; i < ndim; i++ {
		if sliceSteps[i] > 0 {
			newShape[i] = (sliceEnds[i] - sliceStarts[i] + sliceSteps[i] - 1) / sliceSteps[i]
		} else {
			newShape[i] = (sliceStarts[i] - sliceEnds[i] - sliceSteps[i] - 1) / (-sliceSteps[i])
		}
		if newShape[i] < 0 {
			newShape[i] = 0
		}
	}

	result, err := NewRaw(newShape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Slice: %w", err)
	}

	if result.NumElements() == 0 {
		return result, nil
	}

	// Copy data
	switch x.dtype {
	case Float32:
		sliceDataFloat32(x.AsFloat32(), result.AsFloat32(), x.shape, newShape, sliceStarts, sliceSteps)
	case Float64:
		sliceDataFloat64(x.AsFloat64(), result.AsFloat64(), x.shape, newShape, sliceStarts, sliceSteps)
	case Int64:
		sliceDataInt64(x.AsInt64(), result.AsInt64(), x.shape, newShape, sliceStarts, sliceSteps)
	case Int32:
		sliceDataInt32(x.AsInt32(), result.AsInt32(), x.shape, newShape, sliceStarts, sliceSteps)
	default:
		return nil, fmt.Errorf("Slice: unsupported dtype %v", x.dtype)
	}

	return result, nil
}

//nolint:dupl // Type-specific implementations for performance
func sliceDataFloat32(in, out []float32, oldShape, newShape Shape, starts, steps []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		// Compute new index
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		// Compute old index
		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldIdx := starts[j] + idx[j]*steps[j]
			oldFlat += oldIdx * oldStrides[j]
		}

		// Compute new flat index
		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

//nolint:dupl // Type-specific implementations for performance
func sliceDataFloat64(in, out []float64, oldShape, newShape Shape, starts, steps []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldIdx := starts[j] + idx[j]*steps[j]
			oldFlat += oldIdx * oldStrides[j]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

//nolint:dupl // Type-specific implementations for performance
func sliceDataInt64(in, out []int64, oldShape, newShape Shape, starts, steps []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldIdx := starts[j] + idx[j]*steps[j]
			oldFlat += oldIdx * oldStrides[j]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

//nolint:dupl // Type-specific implementations for performance
func sliceDataInt32(in, out []int32, oldShape, newShape Shape, starts, steps []int) {
	ndim := len(oldShape)
	oldStrides := oldShape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	total := 1
	for _, d := range newShape {
		total *= d
	}

	idx := make([]int, ndim)
	for i := 0; i < total; i++ {
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			idx[j] = tmp % newShape[j]
			tmp /= newShape[j]
		}

		oldFlat := 0
		for j := 0; j < ndim; j++ {
			oldIdx := starts[j] + idx[j]*steps[j]
			oldFlat += oldIdx * oldStrides[j]
		}

		newFlat := 0
		for j := 0; j < ndim; j++ {
			newFlat += idx[j] * newStrides[j]
		}

		out[newFlat] = in[oldFlat]
	}
}

// Gather selects elements along an axis according to indices.
//
//nolint:gocyclo,cyclop // Gather indexing has inherent complexity
func Gather(x, indices *RawTensor, axis int) (*RawTensor, error) {
	if x == nil || indices == nil {
		return nil, fmt.Errorf("Gather: input tensors cannot be nil")
	}

	ndim := len(x.shape)
	if axis < 0 {
		axis = ndim + axis
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("Gather: axis %d out of range for %d dimensions", axis, ndim)
	}

	// Output shape: x.shape[:axis] + indices.shape + x.shape[axis+1:]
	newShape := make(Shape, 0, ndim-1+len(indices.shape))
	for i := 0; i < axis; i++ {
		newShape = append(newShape, x.shape[i])
	}
	newShape = append(newShape, indices.shape...)
	for i := axis + 1; i < ndim; i++ {
		newShape = append(newShape, x.shape[i])
	}

	result, err := NewRaw(newShape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Gather: %w", err)
	}

	// Get indices as int
	var indexData []int
	switch indices.dtype {
	case Int32:
		idx32 := indices.AsInt32()
		indexData = make([]int, len(idx32))
		for i, v := range idx32 {
			indexData[i] = int(v)
		}
	case Int64:
		idx64 := indices.AsInt64()
		indexData = make([]int, len(idx64))
		for i, v := range idx64 {
			indexData[i] = int(v)
		}
	default:
		return nil, fmt.Errorf("Gather: indices must be int32 or int64, got %v", indices.dtype)
	}

	switch x.dtype {
	case Float32:
		gatherFloat32(x.AsFloat32(), result.AsFloat32(), x.shape, indices.shape, newShape, indexData, axis)
	case Float64:
		gatherFloat64(x.AsFloat64(), result.AsFloat64(), x.shape, indices.shape, newShape, indexData, axis)
	case Int64:
		gatherInt64(x.AsInt64(), result.AsInt64(), x.shape, indices.shape, newShape, indexData, axis)
	case Int32:
		gatherInt32(x.AsInt32(), result.AsInt32(), x.shape, indices.shape, newShape, indexData, axis)
	default:
		return nil, fmt.Errorf("Gather: unsupported dtype %v", x.dtype)
	}

	return result, nil
}

func gatherFloat32(in, out []float32, xShape, idxShape, outShape Shape, indices []int, axis int) {
	xStrides := xShape.ComputeStrides()
	outStrides := outShape.ComputeStrides()
	idxStrides := idxShape.ComputeStrides()

	total := 1
	for _, d := range outShape {
		total *= d
	}

	ndim := len(xShape)
	idxNdim := len(idxShape)

	for i := 0; i < total; i++ {
		// Decompose output index
		outIdx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			outIdx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		// Get the indices index (middle part of outIdx)
		idxFlat := 0
		for j := 0; j < idxNdim; j++ {
			idxFlat += outIdx[axis+j] * idxStrides[j]
		}
		gatherIdx := indices[idxFlat]

		// Build source index
		srcFlat := 0
		for j := 0; j < axis; j++ {
			srcFlat += outIdx[j] * xStrides[j]
		}
		srcFlat += gatherIdx * xStrides[axis]
		for j := axis + 1; j < ndim; j++ {
			srcFlat += outIdx[axis+idxNdim+j-axis-1] * xStrides[j]
		}

		// Build dest index
		dstFlat := 0
		for j := 0; j < len(outShape); j++ {
			dstFlat += outIdx[j] * outStrides[j]
		}

		out[dstFlat] = in[srcFlat]
	}
}

func gatherFloat64(in, out []float64, xShape, idxShape, outShape Shape, indices []int, axis int) {
	xStrides := xShape.ComputeStrides()
	outStrides := outShape.ComputeStrides()
	idxStrides := idxShape.ComputeStrides()

	total := 1
	for _, d := range outShape {
		total *= d
	}

	ndim := len(xShape)
	idxNdim := len(idxShape)

	for i := 0; i < total; i++ {
		outIdx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			outIdx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		idxFlat := 0
		for j := 0; j < idxNdim; j++ {
			idxFlat += outIdx[axis+j] * idxStrides[j]
		}
		gatherIdx := indices[idxFlat]

		srcFlat := 0
		for j := 0; j < axis; j++ {
			srcFlat += outIdx[j] * xStrides[j]
		}
		srcFlat += gatherIdx * xStrides[axis]
		for j := axis + 1; j < ndim; j++ {
			srcFlat += outIdx[axis+idxNdim+j-axis-1] * xStrides[j]
		}

		dstFlat := 0
		for j := 0; j < len(outShape); j++ {
			dstFlat += outIdx[j] * outStrides[j]
		}

		out[dstFlat] = in[srcFlat]
	}
}

func gatherInt64(in, out []int64, xShape, idxShape, outShape Shape, indices []int, axis int) {
	xStrides := xShape.ComputeStrides()
	outStrides := outShape.ComputeStrides()
	idxStrides := idxShape.ComputeStrides()

	total := 1
	for _, d := range outShape {
		total *= d
	}

	ndim := len(xShape)
	idxNdim := len(idxShape)

	for i := 0; i < total; i++ {
		outIdx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			outIdx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		idxFlat := 0
		for j := 0; j < idxNdim; j++ {
			idxFlat += outIdx[axis+j] * idxStrides[j]
		}
		gatherIdx := indices[idxFlat]

		srcFlat := 0
		for j := 0; j < axis; j++ {
			srcFlat += outIdx[j] * xStrides[j]
		}
		srcFlat += gatherIdx * xStrides[axis]
		for j := axis + 1; j < ndim; j++ {
			srcFlat += outIdx[axis+idxNdim+j-axis-1] * xStrides[j]
		}

		dstFlat := 0
		for j := 0; j < len(outShape); j++ {
			dstFlat += outIdx[j] * outStrides[j]
		}

		out[dstFlat] = in[srcFlat]
	}
}

func gatherInt32(in, out []int32, xShape, idxShape, outShape Shape, indices []int, axis int) {
	xStrides := xShape.ComputeStrides()
	outStrides := outShape.ComputeStrides()
	idxStrides := idxShape.ComputeStrides()

	total := 1
	for _, d := range outShape {
		total *= d
	}

	ndim := len(xShape)
	idxNdim := len(idxShape)

	for i := 0; i < total; i++ {
		outIdx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			outIdx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		idxFlat := 0
		for j := 0; j < idxNdim; j++ {
			idxFlat += outIdx[axis+j] * idxStrides[j]
		}
		gatherIdx := indices[idxFlat]

		srcFlat := 0
		for j := 0; j < axis; j++ {
			srcFlat += outIdx[j] * xStrides[j]
		}
		srcFlat += gatherIdx * xStrides[axis]
		for j := axis + 1; j < ndim; j++ {
			srcFlat += outIdx[axis+idxNdim+j-axis-1] * xStrides[j]
		}

		dstFlat := 0
		for j := 0; j < len(outShape); j++ {
			dstFlat += outIdx[j] * outStrides[j]
		}

		out[dstFlat] = in[srcFlat]
	}
}

// Flatten flattens tensor from axis onward into a single dimension.
func Flatten(x *RawTensor, axis int) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Flatten: input tensor is nil")
	}

	ndim := len(x.shape)
	if axis < 0 {
		axis = ndim + axis
	}
	if axis < 0 || axis > ndim {
		return nil, fmt.Errorf("Flatten: axis %d out of range [0, %d]", axis, ndim)
	}

	// Calculate new shape
	dim0 := 1
	for i := 0; i < axis; i++ {
		dim0 *= x.shape[i]
	}
	dim1 := 1
	for i := axis; i < ndim; i++ {
		dim1 *= x.shape[i]
	}

	newShape := Shape{dim0, dim1}
	if axis == 0 {
		newShape = Shape{1, dim1}
	}

	return Reshape(x, newShape)
}

// Expand broadcasts a tensor to a larger shape.
func Expand(x *RawTensor, targetShape Shape) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Expand: input tensor is nil")
	}

	// Validate shapes are compatible for broadcasting
	xShape := x.shape
	if len(targetShape) < len(xShape) {
		return nil, fmt.Errorf("Expand: target shape must have at least as many dimensions as input")
	}

	// Prepend 1s to source shape if needed
	paddedShape := make(Shape, len(targetShape))
	diff := len(targetShape) - len(xShape)
	for i := 0; i < diff; i++ {
		paddedShape[i] = 1
	}
	copy(paddedShape[diff:], xShape)

	// Validate broadcast compatibility
	for i := range targetShape {
		if paddedShape[i] != 1 && paddedShape[i] != targetShape[i] {
			return nil, fmt.Errorf("Expand: cannot expand dimension %d from %d to %d", i, paddedShape[i], targetShape[i])
		}
	}

	result, err := NewRaw(targetShape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Expand: %w", err)
	}

	switch x.dtype {
	case Float32:
		expandFloat32(x.AsFloat32(), result.AsFloat32(), paddedShape, targetShape)
	case Float64:
		expandFloat64(x.AsFloat64(), result.AsFloat64(), paddedShape, targetShape)
	case Int64:
		expandInt64(x.AsInt64(), result.AsInt64(), paddedShape, targetShape)
	case Int32:
		expandInt32(x.AsInt32(), result.AsInt32(), paddedShape, targetShape)
	default:
		return nil, fmt.Errorf("Expand: unsupported dtype %v", x.dtype)
	}

	return result, nil
}

func expandFloat32(in, out []float32, srcShape, dstShape Shape) {
	srcStrides := srcShape.ComputeStrides()
	ndim := len(dstShape)

	total := 1
	for _, d := range dstShape {
		total *= d
	}

	for i := 0; i < total; i++ {
		// Decompose output index
		dstIdx := make([]int, ndim)
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			dstIdx[j] = tmp % dstShape[j]
			tmp /= dstShape[j]
		}

		// Compute source index (with broadcasting)
		srcFlat := 0
		for j := 0; j < ndim; j++ {
			if srcShape[j] == 1 {
				// Broadcast: always use index 0
				continue
			}
			srcFlat += dstIdx[j] * srcStrides[j]
		}

		out[i] = in[srcFlat]
	}
}

func expandFloat64(in, out []float64, srcShape, dstShape Shape) {
	srcStrides := srcShape.ComputeStrides()
	ndim := len(dstShape)

	total := 1
	for _, d := range dstShape {
		total *= d
	}

	for i := 0; i < total; i++ {
		dstIdx := make([]int, ndim)
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			dstIdx[j] = tmp % dstShape[j]
			tmp /= dstShape[j]
		}

		srcFlat := 0
		for j := 0; j < ndim; j++ {
			if srcShape[j] == 1 {
				continue
			}
			srcFlat += dstIdx[j] * srcStrides[j]
		}

		out[i] = in[srcFlat]
	}
}

func expandInt64(in, out []int64, srcShape, dstShape Shape) {
	srcStrides := srcShape.ComputeStrides()
	ndim := len(dstShape)

	total := 1
	for _, d := range dstShape {
		total *= d
	}

	for i := 0; i < total; i++ {
		dstIdx := make([]int, ndim)
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			dstIdx[j] = tmp % dstShape[j]
			tmp /= dstShape[j]
		}

		srcFlat := 0
		for j := 0; j < ndim; j++ {
			if srcShape[j] == 1 {
				continue
			}
			srcFlat += dstIdx[j] * srcStrides[j]
		}

		out[i] = in[srcFlat]
	}
}

func expandInt32(in, out []int32, srcShape, dstShape Shape) {
	srcStrides := srcShape.ComputeStrides()
	ndim := len(dstShape)

	total := 1
	for _, d := range dstShape {
		total *= d
	}

	for i := 0; i < total; i++ {
		dstIdx := make([]int, ndim)
		tmp := i
		for j := ndim - 1; j >= 0; j-- {
			dstIdx[j] = tmp % dstShape[j]
			tmp /= dstShape[j]
		}

		srcFlat := 0
		for j := 0; j < ndim; j++ {
			if srcShape[j] == 1 {
				continue
			}
			srcFlat += dstIdx[j] * srcStrides[j]
		}

		out[i] = in[srcFlat]
	}
}

// Cast converts a tensor to a different data type.
func Cast(x *RawTensor, dtype DataType) (*RawTensor, error) {
	if x == nil {
		return nil, fmt.Errorf("Cast: input tensor is nil")
	}

	if x.dtype == dtype {
		return x.Clone(), nil
	}

	result, err := NewRaw(x.shape, dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Cast: %w", err)
	}

	switch x.dtype {
	case Float32:
		castFromFloat32(x.AsFloat32(), result, dtype)
	case Float64:
		castFromFloat64(x.AsFloat64(), result, dtype)
	case Int32:
		castFromInt32(x.AsInt32(), result, dtype)
	case Int64:
		castFromInt64(x.AsInt64(), result, dtype)
	case Uint8:
		castFromUint8(x.AsUint8(), result, dtype)
	case Bool:
		castFromBool(x.AsBool(), result, dtype)
	default:
		return nil, fmt.Errorf("Cast: unsupported source dtype %v", x.dtype)
	}

	return result, nil
}

func castFromFloat32(in []float32, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		copy(out.AsFloat32(), in)
	case Float64:
		dst := out.AsFloat64()
		for i, v := range in {
			dst[i] = float64(v)
		}
	case Int32:
		dst := out.AsInt32()
		for i, v := range in {
			dst[i] = int32(v)
		}
	case Int64:
		dst := out.AsInt64()
		for i, v := range in {
			dst[i] = int64(v)
		}
	case Uint8:
		dst := out.AsUint8()
		for i, v := range in {
			dst[i] = uint8(v)
		}
	case Bool:
		dst := out.AsBool()
		for i, v := range in {
			dst[i] = v != 0
		}
	}
}

func castFromFloat64(in []float64, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		dst := out.AsFloat32()
		for i, v := range in {
			dst[i] = float32(v)
		}
	case Float64:
		copy(out.AsFloat64(), in)
	case Int32:
		dst := out.AsInt32()
		for i, v := range in {
			dst[i] = int32(v)
		}
	case Int64:
		dst := out.AsInt64()
		for i, v := range in {
			dst[i] = int64(v)
		}
	case Uint8:
		dst := out.AsUint8()
		for i, v := range in {
			dst[i] = uint8(v)
		}
	case Bool:
		dst := out.AsBool()
		for i, v := range in {
			dst[i] = v != 0
		}
	}
}

// castFromInt32 converts int32 slice to target dtype.
//

func castFromInt32(in []int32, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		dst := out.AsFloat32()
		for i, v := range in {
			dst[i] = float32(v)
		}
	case Float64:
		dst := out.AsFloat64()
		for i, v := range in {
			dst[i] = float64(v)
		}
	case Int32:
		copy(out.AsInt32(), in)
	case Int64:
		dst := out.AsInt64()
		for i, v := range in {
			dst[i] = int64(v)
		}
	case Uint8:
		dst := out.AsUint8()
		for i, v := range in {
			dst[i] = uint8(v)
		}
	case Bool:
		dst := out.AsBool()
		for i, v := range in {
			dst[i] = v != 0
		}
	}
}

// castFromInt64 converts int64 slice to target dtype.
//

func castFromInt64(in []int64, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		dst := out.AsFloat32()
		for i, v := range in {
			dst[i] = float32(v)
		}
	case Float64:
		dst := out.AsFloat64()
		for i, v := range in {
			dst[i] = float64(v)
		}
	case Int32:
		dst := out.AsInt32()
		for i, v := range in {
			dst[i] = int32(v)
		}
	case Int64:
		copy(out.AsInt64(), in)
	case Uint8:
		dst := out.AsUint8()
		for i, v := range in {
			dst[i] = uint8(v)
		}
	case Bool:
		dst := out.AsBool()
		for i, v := range in {
			dst[i] = v != 0
		}
	}
}

func castFromUint8(in []uint8, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		dst := out.AsFloat32()
		for i, v := range in {
			dst[i] = float32(v)
		}
	case Float64:
		dst := out.AsFloat64()
		for i, v := range in {
			dst[i] = float64(v)
		}
	case Int32:
		dst := out.AsInt32()
		for i, v := range in {
			dst[i] = int32(v)
		}
	case Int64:
		dst := out.AsInt64()
		for i, v := range in {
			dst[i] = int64(v)
		}
	case Uint8:
		copy(out.AsUint8(), in)
	case Bool:
		dst := out.AsBool()
		for i, v := range in {
			dst[i] = v != 0
		}
	}
}

// castFromBool converts bool slice to target dtype.
//
//nolint:gocognit,gocyclo,cyclop // Type conversion has inherent complexity
func castFromBool(in []bool, out *RawTensor, dtype DataType) {
	switch dtype {
	case Float32:
		dst := out.AsFloat32()
		for i, v := range in {
			if v {
				dst[i] = 1
			}
		}
	case Float64:
		dst := out.AsFloat64()
		for i, v := range in {
			if v {
				dst[i] = 1
			}
		}
	case Int32:
		dst := out.AsInt32()
		for i, v := range in {
			if v {
				dst[i] = 1
			}
		}
	case Int64:
		dst := out.AsInt64()
		for i, v := range in {
			if v {
				dst[i] = 1
			}
		}
	case Uint8:
		dst := out.AsUint8()
		for i, v := range in {
			if v {
				dst[i] = 1
			}
		}
	case Bool:
		copy(out.AsBool(), in)
	}
}

// WhereRaw selects elements from x or y based on condition.
//
//nolint:gocyclo,cyclop // Conditional selection has inherent complexity
func WhereRaw(condition, x, y *RawTensor) (*RawTensor, error) {
	if condition == nil || x == nil || y == nil {
		return nil, fmt.Errorf("Where: input tensors cannot be nil")
	}

	// Broadcast shapes
	shape, _, err := BroadcastShapes(x.shape, y.shape)
	if err != nil {
		return nil, fmt.Errorf("Where: %w", err)
	}
	shape, _, err = BroadcastShapes(shape, condition.shape)
	if err != nil {
		return nil, fmt.Errorf("Where: %w", err)
	}

	result, err := NewRaw(shape, x.dtype, x.device)
	if err != nil {
		return nil, fmt.Errorf("Where: %w", err)
	}

	// Get condition as bool
	var cond []bool
	switch condition.dtype {
	case Bool:
		cond = condition.AsBool()
	case Float32:
		f := condition.AsFloat32()
		cond = make([]bool, len(f))
		for i, v := range f {
			cond[i] = v != 0
		}
	case Float64:
		f := condition.AsFloat64()
		cond = make([]bool, len(f))
		for i, v := range f {
			cond[i] = v != 0
		}
	case Int32:
		f := condition.AsInt32()
		cond = make([]bool, len(f))
		for i, v := range f {
			cond[i] = v != 0
		}
	case Int64:
		f := condition.AsInt64()
		cond = make([]bool, len(f))
		for i, v := range f {
			cond[i] = v != 0
		}
	case Uint8:
		f := condition.AsUint8()
		cond = make([]bool, len(f))
		for i, v := range f {
			cond[i] = v != 0
		}
	default:
		return nil, fmt.Errorf("Where: unsupported condition dtype %v", condition.dtype)
	}

	switch x.dtype {
	case Float32:
		whereFloat32(cond, x.AsFloat32(), y.AsFloat32(), result.AsFloat32(), condition.shape, x.shape, y.shape, shape)
	case Float64:
		whereFloat64(cond, x.AsFloat64(), y.AsFloat64(), result.AsFloat64(), condition.shape, x.shape, y.shape, shape)
	case Int64:
		whereInt64(cond, x.AsInt64(), y.AsInt64(), result.AsInt64(), condition.shape, x.shape, y.shape, shape)
	case Int32:
		whereInt32(cond, x.AsInt32(), y.AsInt32(), result.AsInt32(), condition.shape, x.shape, y.shape, shape)
	default:
		return nil, fmt.Errorf("Where: unsupported dtype %v", x.dtype)
	}

	return result, nil
}

func whereFloat32(cond []bool, x, y, out []float32, condShape, xShape, yShape, outShape Shape) {
	total := 1
	for _, d := range outShape {
		total *= d
	}

	condStrides := condShape.ComputeStrides()
	xStrides := xShape.ComputeStrides()
	yStrides := yShape.ComputeStrides()

	for i := 0; i < total; i++ {
		// Decompose index
		idx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			idx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		// Get broadcast indices
		condIdx := broadcastIndex(idx, condShape, condStrides)
		xIdx := broadcastIndex(idx, xShape, xStrides)
		yIdx := broadcastIndex(idx, yShape, yStrides)

		if cond[condIdx] {
			out[i] = x[xIdx]
		} else {
			out[i] = y[yIdx]
		}
	}
}

func whereFloat64(cond []bool, x, y, out []float64, condShape, xShape, yShape, outShape Shape) {
	total := 1
	for _, d := range outShape {
		total *= d
	}

	condStrides := condShape.ComputeStrides()
	xStrides := xShape.ComputeStrides()
	yStrides := yShape.ComputeStrides()

	for i := 0; i < total; i++ {
		idx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			idx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		condIdx := broadcastIndex(idx, condShape, condStrides)
		xIdx := broadcastIndex(idx, xShape, xStrides)
		yIdx := broadcastIndex(idx, yShape, yStrides)

		if cond[condIdx] {
			out[i] = x[xIdx]
		} else {
			out[i] = y[yIdx]
		}
	}
}

func whereInt64(cond []bool, x, y, out []int64, condShape, xShape, yShape, outShape Shape) {
	total := 1
	for _, d := range outShape {
		total *= d
	}

	condStrides := condShape.ComputeStrides()
	xStrides := xShape.ComputeStrides()
	yStrides := yShape.ComputeStrides()

	for i := 0; i < total; i++ {
		idx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			idx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		condIdx := broadcastIndex(idx, condShape, condStrides)
		xIdx := broadcastIndex(idx, xShape, xStrides)
		yIdx := broadcastIndex(idx, yShape, yStrides)

		if cond[condIdx] {
			out[i] = x[xIdx]
		} else {
			out[i] = y[yIdx]
		}
	}
}

func whereInt32(cond []bool, x, y, out []int32, condShape, xShape, yShape, outShape Shape) {
	total := 1
	for _, d := range outShape {
		total *= d
	}

	condStrides := condShape.ComputeStrides()
	xStrides := xShape.ComputeStrides()
	yStrides := yShape.ComputeStrides()

	for i := 0; i < total; i++ {
		idx := make([]int, len(outShape))
		tmp := i
		for j := len(outShape) - 1; j >= 0; j-- {
			idx[j] = tmp % outShape[j]
			tmp /= outShape[j]
		}

		condIdx := broadcastIndex(idx, condShape, condStrides)
		xIdx := broadcastIndex(idx, xShape, xStrides)
		yIdx := broadcastIndex(idx, yShape, yStrides)

		if cond[condIdx] {
			out[i] = x[xIdx]
		} else {
			out[i] = y[yIdx]
		}
	}
}

func broadcastIndex(idx []int, shape Shape, strides []int) int {
	result := 0
	diff := len(idx) - len(shape)
	for i := 0; i < len(shape); i++ {
		dimIdx := idx[diff+i]
		if shape[i] == 1 {
			dimIdx = 0 // Broadcast
		}
		result += dimIdx * strides[i]
	}
	return result
}

// FullRaw creates a RawTensor filled with a constant value.
func FullRaw(shape Shape, value float32, dtype DataType, device Device) (*RawTensor, error) {
	result, err := NewRaw(shape, dtype, device)
	if err != nil {
		return nil, fmt.Errorf("Full: %w", err)
	}

	switch dtype {
	case Float32:
		data := result.AsFloat32()
		for i := range data {
			data[i] = value
		}
	case Float64:
		data := result.AsFloat64()
		v := float64(value)
		for i := range data {
			data[i] = v
		}
	case Int32:
		data := result.AsInt32()
		v := int32(value)
		for i := range data {
			data[i] = v
		}
	case Int64:
		data := result.AsInt64()
		v := int64(value)
		for i := range data {
			data[i] = v
		}
	case Uint8:
		data := result.AsUint8()
		v := uint8(value)
		for i := range data {
			data[i] = v
		}
	case Bool:
		data := result.AsBool()
		v := value != 0
		for i := range data {
			data[i] = v
		}
	default:
		return nil, fmt.Errorf("Full: unsupported dtype %v", dtype)
	}

	return result, nil
}
