package webgpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Add performs element-wise addition on GPU.
// TODO: Implement WGSL compute shader for addition.
func (b *Backend) Add(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Add not implemented yet - see TASK-009")
}

// Sub performs element-wise subtraction on GPU.
// TODO: Implement WGSL compute shader for subtraction.
func (b *Backend) Sub(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Sub not implemented yet - see TASK-009")
}

// Mul performs element-wise multiplication on GPU.
// TODO: Implement WGSL compute shader for multiplication.
func (b *Backend) Mul(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Mul not implemented yet - see TASK-009")
}

// Div performs element-wise division on GPU.
// TODO: Implement WGSL compute shader for division.
func (b *Backend) Div(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Div not implemented yet - see TASK-009")
}

// MatMul performs matrix multiplication on GPU.
// TODO: Implement WGSL compute shader for matmul.
func (b *Backend) MatMul(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: MatMul not implemented yet - see TASK-009")
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
// TODO: Implement WGSL compute shader for transpose.
func (b *Backend) Transpose(_ *tensor.RawTensor, _ ...int) *tensor.RawTensor {
	panic("webgpu: Transpose not implemented yet - see TASK-009")
}
