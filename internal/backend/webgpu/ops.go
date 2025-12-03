//go:build windows

package webgpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Add performs element-wise addition on GPU.
// Supports float32 and int32 dtypes.
// In LazyMode (default), returns a lazy tensor that keeps data on GPU.
func (b *Backend) Add(a, other *tensor.RawTensor) *tensor.RawTensor {
	shaderName, shaderCode := selectBinaryShader(a.DType(), "add", addShader, addShaderInt32)

	var result *tensor.RawTensor
	var err error

	if b.LazyMode {
		result, err = b.runBinaryOpLazy(a, other, shaderName, shaderCode)
	} else {
		result, err = b.runBinaryOp(a, other, shaderName, shaderCode)
	}

	if err != nil {
		panic("webgpu: Add: " + err.Error())
	}
	return result
}

// Sub performs element-wise subtraction on GPU.
// Supports float32 and int32 dtypes.
// In LazyMode (default), returns a lazy tensor that keeps data on GPU.
func (b *Backend) Sub(a, other *tensor.RawTensor) *tensor.RawTensor {
	shaderName, shaderCode := selectBinaryShader(a.DType(), "sub", subShader, subShaderInt32)

	var result *tensor.RawTensor
	var err error

	if b.LazyMode {
		result, err = b.runBinaryOpLazy(a, other, shaderName, shaderCode)
	} else {
		result, err = b.runBinaryOp(a, other, shaderName, shaderCode)
	}

	if err != nil {
		panic("webgpu: Sub: " + err.Error())
	}
	return result
}

// Mul performs element-wise multiplication on GPU.
// Supports float32 and int32 dtypes.
// In LazyMode (default), returns a lazy tensor that keeps data on GPU.
func (b *Backend) Mul(a, other *tensor.RawTensor) *tensor.RawTensor {
	shaderName, shaderCode := selectBinaryShader(a.DType(), "mul", mulShader, mulShaderInt32)

	var result *tensor.RawTensor
	var err error

	if b.LazyMode {
		result, err = b.runBinaryOpLazy(a, other, shaderName, shaderCode)
	} else {
		result, err = b.runBinaryOp(a, other, shaderName, shaderCode)
	}

	if err != nil {
		panic("webgpu: Mul: " + err.Error())
	}
	return result
}

// Div performs element-wise division on GPU.
// Supports float32 and int32 dtypes.
// In LazyMode (default), returns a lazy tensor that keeps data on GPU.
func (b *Backend) Div(a, other *tensor.RawTensor) *tensor.RawTensor {
	shaderName, shaderCode := selectBinaryShader(a.DType(), "div", divShader, divShaderInt32)

	var result *tensor.RawTensor
	var err error

	if b.LazyMode {
		result, err = b.runBinaryOpLazy(a, other, shaderName, shaderCode)
	} else {
		result, err = b.runBinaryOp(a, other, shaderName, shaderCode)
	}

	if err != nil {
		panic("webgpu: Div: " + err.Error())
	}
	return result
}

// selectBinaryShader selects the appropriate shader based on dtype.
func selectBinaryShader(dtype tensor.DataType, baseName, float32Shader, int32Shader string) (string, string) {
	switch dtype {
	case tensor.Float32:
		return baseName, float32Shader
	case tensor.Int32:
		return baseName + "Int32", int32Shader
	default:
		panic("webgpu: unsupported dtype for binary operation: " + dtype.String())
	}
}

// MatMul performs matrix multiplication on GPU.
func (b *Backend) MatMul(a, other *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error

	if b.LazyMode {
		result, err = b.runMatMulLazy(a, other)
	} else {
		result, err = b.runMatMul(a, other)
	}

	if err != nil {
		panic("webgpu: MatMul: " + err.Error())
	}
	return result
}

// BatchMatMul performs batched matrix multiplication on GPU.
// Supports 3D tensors [batch, M, K] @ [batch, K, N] -> [batch, M, N]
// and 4D tensors [batch, heads, M, K] @ [batch, heads, K, N].
func (b *Backend) BatchMatMul(a, other *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runBatchMatMulLazy(a, other)
	} else {
		result, err = b.runBatchMatMul(a, other)
	}
	if err != nil {
		panic("webgpu: BatchMatMul: " + err.Error())
	}
	return result
}

// Conv2D performs 2D convolution on GPU.
// Input shape: [batch, in_channels, height, width].
// Kernel shape: [out_channels, in_channels, kH, kW].
func (b *Backend) Conv2D(input, kernel *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	result, err := b.runConv2D(input, kernel, stride, padding)
	if err != nil {
		panic("webgpu: Conv2D: " + err.Error())
	}
	return result
}

// MaxPool2D performs 2D max pooling on GPU.
// Input shape: [batch, channels, height, width].
func (b *Backend) MaxPool2D(input *tensor.RawTensor, kernelSize, stride int) *tensor.RawTensor {
	result, err := b.runMaxPool2D(input, kernelSize, stride)
	if err != nil {
		panic("webgpu: MaxPool2D: " + err.Error())
	}
	return result
}

// Reshape returns a tensor with new shape.
// This is typically a metadata-only operation (zero-copy).
func (b *Backend) Reshape(t *tensor.RawTensor, newShape tensor.Shape) *tensor.RawTensor {
	if err := newShape.Validate(); err != nil {
		panic("webgpu: reshape: invalid shape: " + err.Error())
	}

	if t.NumElements() != newShape.NumElements() {
		panic("webgpu: reshape: incompatible number of elements")
	}

	// Reshape is a view operation - create new tensor with same data
	result, err := tensor.NewRaw(newShape, t.DType(), tensor.WebGPU)
	if err != nil {
		panic("webgpu: reshape: " + err.Error())
	}

	// Copy data (for now - TODO: make this zero-copy when GPU buffers are implemented)
	copy(result.Data(), t.Data())
	return result
}

// Transpose transposes the tensor by permuting its dimensions.
// GPU-accelerated for 2D (optimized) and ND tensors (up to 6D).
func (b *Backend) Transpose(t *tensor.RawTensor, axes ...int) *tensor.RawTensor {
	shape := t.Shape()
	ndim := len(shape)

	// Check for no-op case early
	if isNoOpTranspose(axes, ndim) {
		return t
	}

	// 2D transpose (matrix): use optimized 2D shader
	if ndim == 2 {
		validate2DTransposeAxes(axes)
		var result *tensor.RawTensor
		var err error
		if b.LazyMode {
			result, err = b.runTransposeLazy(t)
		} else {
			result, err = b.runTranspose(t)
		}
		if err != nil {
			panic("webgpu: Transpose: " + err.Error())
		}
		return result
	}

	// Multi-dimensional (3D, 4D, etc.): use GPU-accelerated ND transpose
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runTransposeNDLazy(t, axes)
	} else {
		result, err = b.runTransposeND(t, axes)
	}
	if err != nil {
		panic("webgpu: Transpose: " + err.Error())
	}
	return result
}

// isNoOpTranspose checks if transpose is a no-op.
func isNoOpTranspose(axes []int, ndim int) bool {
	if len(axes) == 0 {
		return false
	}
	if len(axes) != ndim {
		return false
	}
	for i, ax := range axes {
		if ax != i {
			return false
		}
	}
	return true
}

// validate2DTransposeAxes validates axes for 2D transpose.
func validate2DTransposeAxes(axes []int) {
	if len(axes) > 0 && (len(axes) != 2 || !isValid2DAxes(axes)) {
		panic("webgpu: Transpose: invalid axes for 2D tensor")
	}
}

// isValid2DAxes checks if axes are valid for 2D transpose.
func isValid2DAxes(axes []int) bool {
	return (axes[0] == 0 && axes[1] == 1) || (axes[0] == 1 && axes[1] == 0)
}

// ReLU applies ReLU activation: max(0, x).
func (b *Backend) ReLU(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runUnaryOp(x, "relu", reluShader)
	if err != nil {
		panic("webgpu: ReLU: " + err.Error())
	}
	return result
}

// Sigmoid applies sigmoid activation: 1 / (1 + exp(-x)).
func (b *Backend) Sigmoid(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runUnaryOp(x, "sigmoid", sigmoidShader)
	if err != nil {
		panic("webgpu: Sigmoid: " + err.Error())
	}
	return result
}

// Tanh applies tanh activation.
func (b *Backend) Tanh(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runUnaryOp(x, "tanh", tanhShader)
	if err != nil {
		panic("webgpu: Tanh: " + err.Error())
	}
	return result
}

// Exp computes element-wise exponential on GPU.
func (b *Backend) Exp(x *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runUnaryOpLazy(x, "exp", expShader)
	} else {
		result, err = b.runUnaryOp(x, "exp", expShader)
	}
	if err != nil {
		panic("webgpu: Exp: " + err.Error())
	}
	return result
}

// Sqrt computes element-wise square root on GPU.
func (b *Backend) Sqrt(x *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runUnaryOpLazy(x, "sqrt", sqrtShader)
	} else {
		result, err = b.runUnaryOp(x, "sqrt", sqrtShader)
	}
	if err != nil {
		panic("webgpu: Sqrt: " + err.Error())
	}
	return result
}

// Rsqrt computes element-wise reciprocal square root on GPU.
func (b *Backend) Rsqrt(x *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runUnaryOpLazy(x, "rsqrt", rsqrtShader)
	} else {
		result, err = b.runUnaryOp(x, "rsqrt", rsqrtShader)
	}
	if err != nil {
		panic("webgpu: Rsqrt: " + err.Error())
	}
	return result
}

// Cos computes element-wise cosine on GPU.
func (b *Backend) Cos(x *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runUnaryOpLazy(x, "cos", cosShader)
	} else {
		result, err = b.runUnaryOp(x, "cos", cosShader)
	}
	if err != nil {
		panic("webgpu: Cos: " + err.Error())
	}
	return result
}

// Sin computes element-wise sine on GPU.
func (b *Backend) Sin(x *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runUnaryOpLazy(x, "sin", sinShader)
	} else {
		result, err = b.runUnaryOp(x, "sin", sinShader)
	}
	if err != nil {
		panic("webgpu: Sin: " + err.Error())
	}
	return result
}

// SumDim sums along a dimension.
// Implemented on CPU as reduction operations are complex on GPU.
func (b *Backend) SumDim(x *tensor.RawTensor, dim int, keepDim bool) *tensor.RawTensor {
	shape := x.Shape()
	ndim := len(shape)

	// Normalize negative dimension
	if dim < 0 {
		dim = ndim + dim
	}

	if dim < 0 || dim >= ndim {
		panic("webgpu: SumDim: dimension out of range")
	}

	// Calculate output shape
	var outShape tensor.Shape
	if keepDim {
		outShape = shape.Clone()
		outShape[dim] = 1
	} else {
		outShape = make(tensor.Shape, 0, ndim-1)
		for i := 0; i < ndim; i++ {
			if i != dim {
				outShape = append(outShape, shape[i])
			}
		}
	}

	// Create result tensor
	result, err := tensor.NewRaw(outShape, x.DType(), tensor.WebGPU)
	if err != nil {
		panic("webgpu: SumDim: " + err.Error())
	}

	// Perform reduction on CPU
	if x.DType() == tensor.Float32 {
		sumDimFloat32(x.AsFloat32(), result.AsFloat32(), shape, dim)
	} else {
		panic("webgpu: SumDim: only float32 supported")
	}

	return result
}

// sumDimFloat32 performs dimension reduction for float32 tensors.
func sumDimFloat32(data, result []float32, shape tensor.Shape, dim int) {
	for i := range result {
		result[i] = 0
	}

	strides := shape.ComputeStrides()
	numElements := shape.NumElements()

	outShape := shape.Clone()
	outShape[dim] = 1
	outStrides := outShape.ComputeStrides()

	for i := 0; i < numElements; i++ {
		outIdx := 0
		temp := i
		for d := 0; d < len(shape); d++ {
			coord := temp / strides[d]
			temp %= strides[d]

			if d != dim {
				outIdx += coord * outStrides[d]
			}
		}
		result[outIdx] += data[i]
	}
}

// MeanDim computes mean along a dimension.
func (b *Backend) MeanDim(x *tensor.RawTensor, dim int, keepDim bool) *tensor.RawTensor {
	sumResult := b.SumDim(x, dim, keepDim)

	shape := x.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}

	divisor := float32(shape[dim])
	data := sumResult.AsFloat32()
	for i := range data {
		data[i] /= divisor
	}

	return sumResult
}

// Cat concatenates tensors along the specified dimension.
func (b *Backend) Cat(tensors []*tensor.RawTensor, dim int) *tensor.RawTensor {
	if len(tensors) == 0 {
		panic("webgpu: Cat: at least one tensor required")
	}

	shape := tensors[0].Shape()
	ndim := len(shape)
	dtype := tensors[0].DType()

	if dim < 0 {
		dim = ndim + dim
	}

	if dim < 0 || dim >= ndim {
		panic("webgpu: Cat: dimension out of range")
	}

	// Calculate total size along concat dimension
	totalDim := 0
	for _, t := range tensors {
		totalDim += t.Shape()[dim]
	}

	// Create output shape
	outShape := shape.Clone()
	outShape[dim] = totalDim

	result, err := tensor.NewRaw(outShape, dtype, tensor.WebGPU)
	if err != nil {
		panic("webgpu: Cat: " + err.Error())
	}

	// Concatenate on CPU
	if dtype == tensor.Float32 {
		catFloat32WebGPU(tensors, result, dim)
	} else {
		panic("webgpu: Cat: only float32 supported")
	}

	return result
}

func catFloat32WebGPU(tensors []*tensor.RawTensor, result *tensor.RawTensor, dim int) {
	outData := result.AsFloat32()
	outShape := result.Shape()
	outStrides := outShape.ComputeStrides()

	offset := 0
	for _, t := range tensors {
		data := t.AsFloat32()
		shape := t.Shape()
		strides := shape.ComputeStrides()
		numElements := shape.NumElements()

		for i := 0; i < numElements; i++ {
			outIdx := 0
			temp := i
			for d := 0; d < len(shape); d++ {
				coord := temp / strides[d]
				temp %= strides[d]

				if d == dim {
					coord += offset
				}
				outIdx += coord * outStrides[d]
			}
			outData[outIdx] = data[i]
		}
		offset += shape[dim]
	}
}

// Chunk splits tensor into n equal parts along the specified dimension.
func (b *Backend) Chunk(x *tensor.RawTensor, n, dim int) []*tensor.RawTensor {
	if n <= 0 {
		panic("webgpu: Chunk: n must be positive")
	}

	shape := x.Shape()
	ndim := len(shape)

	if dim < 0 {
		dim = ndim + dim
	}

	if dim < 0 || dim >= ndim {
		panic("webgpu: Chunk: dimension out of range")
	}

	dimSize := shape[dim]
	if dimSize%n != 0 {
		panic("webgpu: Chunk: dimension not divisible by n")
	}

	chunkSize := dimSize / n
	chunkShape := shape.Clone()
	chunkShape[dim] = chunkSize

	results := make([]*tensor.RawTensor, n)
	for i := 0; i < n; i++ {
		chunk, err := tensor.NewRaw(chunkShape, x.DType(), tensor.WebGPU)
		if err != nil {
			panic("webgpu: Chunk: " + err.Error())
		}
		results[i] = chunk
	}

	if x.DType() == tensor.Float32 {
		chunkFloat32WebGPU(x, results, dim, chunkSize)
	} else {
		panic("webgpu: Chunk: only float32 supported")
	}

	return results
}

func chunkFloat32WebGPU(x *tensor.RawTensor, results []*tensor.RawTensor, dim, chunkSize int) {
	data := x.AsFloat32()
	shape := x.Shape()
	strides := shape.ComputeStrides()
	numElements := shape.NumElements()

	for i := 0; i < numElements; i++ {
		temp := i
		coords := make([]int, len(shape))
		for d := 0; d < len(shape); d++ {
			coords[d] = temp / strides[d]
			temp %= strides[d]
		}

		chunkIdx := coords[dim] / chunkSize
		localCoord := coords[dim] % chunkSize

		outShape := results[chunkIdx].Shape()
		outStrides := outShape.ComputeStrides()
		outIdx := 0
		for d := 0; d < len(coords); d++ {
			if d == dim {
				outIdx += localCoord * outStrides[d]
			} else {
				outIdx += coords[d] * outStrides[d]
			}
		}

		results[chunkIdx].AsFloat32()[outIdx] = data[i]
	}
}

// Unsqueeze adds a dimension of size 1 at the specified position.
func (b *Backend) Unsqueeze(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	shape := x.Shape()
	ndim := len(shape)

	if dim < 0 {
		dim = ndim + 1 + dim
	}

	if dim < 0 || dim > ndim {
		panic("webgpu: Unsqueeze: dimension out of range")
	}

	newShape := make(tensor.Shape, ndim+1)
	for i := 0; i < dim; i++ {
		newShape[i] = shape[i]
	}
	newShape[dim] = 1
	for i := dim; i < ndim; i++ {
		newShape[i+1] = shape[i]
	}

	return b.Reshape(x, newShape)
}

// Squeeze removes a dimension of size 1 at the specified position.
func (b *Backend) Squeeze(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	shape := x.Shape()
	ndim := len(shape)

	if dim < 0 {
		dim = ndim + dim
	}

	if dim < 0 || dim >= ndim {
		panic("webgpu: Squeeze: dimension out of range")
	}

	if shape[dim] != 1 {
		panic("webgpu: Squeeze: dimension size must be 1")
	}

	newShape := make(tensor.Shape, 0, ndim-1)
	for i := 0; i < ndim; i++ {
		if i != dim {
			newShape = append(newShape, shape[i])
		}
	}

	return b.Reshape(x, newShape)
}
