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

// BatchMatMul performs batched matrix multiplication on GPU.
// TODO: Implement WGSL compute shader for batched matmul.
func (b *Backend) BatchMatMul(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: BatchMatMul not implemented yet - see TASK-026")
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

// Exp computes element-wise exponential (stub - not implemented for WebGPU yet).
func (b *Backend) Exp(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Exp not implemented yet (TASK-013)")
}

// Sqrt computes element-wise square root (stub - not implemented for WebGPU yet).
func (b *Backend) Sqrt(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Sqrt not implemented yet (TASK-013)")
}

// Rsqrt computes element-wise reciprocal square root (stub - not implemented for WebGPU yet).
func (b *Backend) Rsqrt(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Rsqrt not implemented yet (TASK-013)")
}

// Cos computes element-wise cosine (stub - not implemented for WebGPU yet).
func (b *Backend) Cos(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Cos not implemented yet (TASK-013)")
}

// Sin computes element-wise sine (stub - not implemented for WebGPU yet).
func (b *Backend) Sin(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Sin not implemented yet (TASK-013)")
}

// SumDim sums along a dimension (stub - not implemented for WebGPU yet).
func (b *Backend) SumDim(_ *tensor.RawTensor, _ int, _ bool) *tensor.RawTensor {
	panic("webgpu: SumDim not implemented yet (TASK-014)")
}

// MeanDim computes mean along a dimension (stub - not implemented for WebGPU yet).
func (b *Backend) MeanDim(_ *tensor.RawTensor, _ int, _ bool) *tensor.RawTensor {
	panic("webgpu: MeanDim not implemented yet (TASK-014)")
}

// Cat concatenates tensors (stub - not implemented for WebGPU yet).
func (b *Backend) Cat(_ []*tensor.RawTensor, _ int) *tensor.RawTensor {
	panic("webgpu: Cat not implemented yet (TASK-015)")
}

// Chunk splits tensor (stub - not implemented for WebGPU yet).
func (b *Backend) Chunk(_ *tensor.RawTensor, _, _ int) []*tensor.RawTensor {
	panic("webgpu: Chunk not implemented yet (TASK-015)")
}

// Unsqueeze adds dimension (stub - not implemented for WebGPU yet).
func (b *Backend) Unsqueeze(_ *tensor.RawTensor, _ int) *tensor.RawTensor {
	panic("webgpu: Unsqueeze not implemented yet (TASK-015)")
}

// Squeeze removes dimension (stub - not implemented for WebGPU yet).
func (b *Backend) Squeeze(_ *tensor.RawTensor, _ int) *tensor.RawTensor {
	panic("webgpu: Squeeze not implemented yet (TASK-015)")
}
