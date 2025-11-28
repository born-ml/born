//go:build windows

package webgpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Add performs element-wise addition on GPU.
func (b *Backend) Add(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runBinaryOp(a, other, "add", addShader)
	if err != nil {
		panic("webgpu: Add: " + err.Error())
	}
	return result
}

// Sub performs element-wise subtraction on GPU.
func (b *Backend) Sub(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runBinaryOp(a, other, "sub", subShader)
	if err != nil {
		panic("webgpu: Sub: " + err.Error())
	}
	return result
}

// Mul performs element-wise multiplication on GPU.
func (b *Backend) Mul(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runBinaryOp(a, other, "mul", mulShader)
	if err != nil {
		panic("webgpu: Mul: " + err.Error())
	}
	return result
}

// Div performs element-wise division on GPU.
func (b *Backend) Div(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runBinaryOp(a, other, "div", divShader)
	if err != nil {
		panic("webgpu: Div: " + err.Error())
	}
	return result
}

// MatMul performs matrix multiplication on GPU.
func (b *Backend) MatMul(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runMatMul(a, other)
	if err != nil {
		panic("webgpu: MatMul: " + err.Error())
	}
	return result
}

// Conv2D performs 2D convolution on GPU.
// TODO: Implement WGSL compute shader for convolution.
func (b *Backend) Conv2D(_, _ *tensor.RawTensor, _, _ int) *tensor.RawTensor {
	panic("webgpu: Conv2D not implemented yet - see TASK-009")
}

// MaxPool2D performs 2D max pooling on GPU.
// TODO: Implement WGSL compute shader for max pooling.
func (b *Backend) MaxPool2D(_ *tensor.RawTensor, _, _ int) *tensor.RawTensor {
	panic("webgpu: MaxPool2D not implemented yet - see TASK-009")
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
// Currently supports only 2D tensors (matrix transpose).
// For multi-dimensional transpose with custom axes, use the general case (TODO).
func (b *Backend) Transpose(t *tensor.RawTensor, axes ...int) *tensor.RawTensor {
	// Simple 2D transpose
	if len(t.Shape()) == 2 && len(axes) == 0 {
		result, err := b.runTranspose(t)
		if err != nil {
			panic("webgpu: Transpose: " + err.Error())
		}
		return result
	}

	// General multi-dimensional transpose (TODO: implement on GPU)
	// For now, fall back to CPU implementation
	panic("webgpu: multi-dimensional transpose not implemented yet - only 2D is supported")
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

// Softmax applies softmax along the last dimension.
// Expects 2D input [batch_size, num_classes].
func (b *Backend) Softmax(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runSoftmax(x)
	if err != nil {
		panic("webgpu: Softmax: " + err.Error())
	}
	return result
}
