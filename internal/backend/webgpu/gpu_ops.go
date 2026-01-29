//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
package webgpu

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
	"github.com/gogpu/gputypes"
)

// AddGPU performs element-wise addition on GPU tensors.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) AddGPU(a, c *GPUTensor) *GPUTensor {
	return b.runBinaryOpGPU(a, c, "add", addShader)
}

// SubGPU performs element-wise subtraction on GPU tensors.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) SubGPU(a, c *GPUTensor) *GPUTensor {
	return b.runBinaryOpGPU(a, c, "sub", subShader)
}

// MulGPU performs element-wise multiplication on GPU tensors.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) MulGPU(a, c *GPUTensor) *GPUTensor {
	return b.runBinaryOpGPU(a, c, "mul", mulShader)
}

// DivGPU performs element-wise division on GPU tensors.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) DivGPU(a, c *GPUTensor) *GPUTensor {
	return b.runBinaryOpGPU(a, c, "div", divShader)
}

// runBinaryOpGPU executes binary operations on GPU tensors without CPU transfer.
// This is the core primitive for lazy GPU operations.
func (b *Backend) runBinaryOpGPU(a, c *GPUTensor, opName, shaderCode string) *GPUTensor {
	// Validate shapes
	if !a.shape.Equal(c.shape) {
		panic(fmt.Sprintf("webgpu: %s: shape mismatch: %v vs %v", opName, a.shape, c.shape))
	}

	// Validate dtypes
	if a.dtype != c.dtype {
		panic(fmt.Sprintf("webgpu: %s: dtype mismatch: %s vs %s", opName, a.dtype, c.dtype))
	}

	// Only float32 and int32 supported for now
	if a.dtype != tensor.Float32 && a.dtype != tensor.Int32 {
		panic(fmt.Sprintf("webgpu: %s: only float32 and int32 supported, got %s", opName, a.dtype))
	}

	// Select shader based on dtype
	shaderName := opName
	if a.dtype == tensor.Int32 {
		shaderName = opName + "Int32"
		switch opName {
		case "add":
			shaderCode = addShaderInt32
		case "sub":
			shaderCode = subShaderInt32
		case "mul":
			shaderCode = mulShaderInt32
		case "div":
			shaderCode = divShaderInt32
		default:
			panic(fmt.Sprintf("webgpu: %s: no int32 shader available", opName))
		}
	}

	numElements := a.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create output buffer (stays on GPU!)
	resultSize := a.ByteSize()
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, a.buffer, 0, resultSize),
		wgpu.BufferBindingEntry(1, c.buffer, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count: ceil(numElements / workgroupSize)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Return GPUTensor (NO readBuffer!)
	return &GPUTensor{
		buffer:     bufferResult,
		shape:      a.shape,
		dtype:      a.dtype,
		strides:    a.strides,
		backend:    b,
		computed:   true,
		bufferSize: resultSize,
	}
}

// MatMulGPU performs matrix multiplication on GPU tensors.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) MatMulGPU(a, c *GPUTensor) *GPUTensor {
	// Validate shapes (must be 2D)
	if len(a.shape) != 2 || len(c.shape) != 2 {
		panic(fmt.Sprintf("webgpu: MatMulGPU: expected 2D tensors, got shapes %v and %v", a.shape, c.shape))
	}

	// Validate matrix dimensions
	if a.shape[1] != c.shape[0] {
		panic(fmt.Sprintf("webgpu: MatMulGPU: incompatible matrix dimensions: [%d, %d] @ [%d, %d]",
			a.shape[0], a.shape[1], c.shape[0], c.shape[1]))
	}

	m := a.shape[0]
	k := a.shape[1]
	n := c.shape[1]

	// Output shape: [m, n]
	outShape := tensor.Shape{m, n}

	// Compile shader
	shader := b.compileShader("matmul", matmulShader)
	pipeline := b.getOrCreatePipeline("matmul", shader)

	// Create output buffer (stays on GPU!)

	resultSize := uint64(m * n * a.dtype.Size()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for dimensions
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(m)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(k)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[8:12], uint32(n)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, a.buffer, 0, a.bufferSize),
		wgpu.BufferBindingEntry(1, c.buffer, 0, c.bufferSize),
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Dispatch 2D workgroups: (m / tileSize, n / tileSize)
	const tileSize = 16

	workgroupsX := uint32((m + tileSize - 1) / tileSize) //nolint:gosec // G115: Cast operation - safe conversion.

	workgroupsY := uint32((n + tileSize - 1) / tileSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Return GPUTensor (NO readBuffer!)
	return &GPUTensor{
		buffer:     bufferResult,
		shape:      outShape,
		dtype:      a.dtype,
		strides:    outShape.ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: resultSize,
	}
}

// TransposeGPU transposes a 2D tensor on GPU.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) TransposeGPU(t *GPUTensor, axes ...int) *GPUTensor {
	shape := t.shape
	ndim := len(shape)

	// Validate 2D for now
	if ndim != 2 {
		panic(fmt.Sprintf("webgpu: TransposeGPU: only 2D tensors supported for now, got %dD", ndim))
	}

	// Validate axes
	if len(axes) > 0 && (len(axes) != 2 || !isValid2DAxes(axes)) {
		panic("webgpu: TransposeGPU: invalid axes for 2D tensor")
	}

	m := shape[0]
	n := shape[1]

	// Output shape: [n, m]
	outShape := tensor.Shape{n, m}

	// Compile shader
	shader := b.compileShader("transpose", transposeShader)
	pipeline := b.getOrCreatePipeline("transpose", shader)

	// Create output buffer (stays on GPU!)

	resultSize := uint64(m * n * t.dtype.Size()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for dimensions
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(m)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(n)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, t.buffer, 0, t.bufferSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Dispatch 2D workgroups: (n / tileSize, m / tileSize)
	const tileSize = 16

	workgroupsX := uint32((n + tileSize - 1) / tileSize) //nolint:gosec // G115: Cast operation - safe conversion.

	workgroupsY := uint32((m + tileSize - 1) / tileSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Return GPUTensor (NO readBuffer!)
	return &GPUTensor{
		buffer:     bufferResult,
		shape:      outShape,
		dtype:      t.dtype,
		strides:    outShape.ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: resultSize,
	}
}

// ReLUGPU applies ReLU activation on GPU: max(0, x).
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) ReLUGPU(t *GPUTensor) *GPUTensor {
	return b.runUnaryOpGPU(t, "relu", reluShader)
}

// SigmoidGPU applies sigmoid activation on GPU: 1 / (1 + exp(-x)).
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) SigmoidGPU(t *GPUTensor) *GPUTensor {
	return b.runUnaryOpGPU(t, "sigmoid", sigmoidShader)
}

// TanhGPU applies tanh activation on GPU.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) TanhGPU(t *GPUTensor) *GPUTensor {
	return b.runUnaryOpGPU(t, "tanh", tanhShader)
}

// SoftmaxGPU applies softmax activation along the specified dimension.
// For now, only last dimension (dim=-1) is supported efficiently on GPU.
// Data stays on GPU - no CPU transfer occurs.
func (b *Backend) SoftmaxGPU(t *GPUTensor, dim int) *GPUTensor {
	shape := t.shape
	ndim := len(shape)

	// Normalize negative dimension
	if dim < 0 {
		dim = ndim + dim
	}

	if dim < 0 || dim >= ndim {
		panic("webgpu: SoftmaxGPU: dimension out of range")
	}

	// For now, only support last dimension
	if dim != ndim-1 {
		panic("webgpu: SoftmaxGPU: only last dimension supported for now")
	}

	// Calculate batch size and feature size
	batchSize := 1
	for i := 0; i < ndim-1; i++ {
		batchSize *= shape[i]
	}
	featureSize := shape[ndim-1]

	// Compile shader
	shader := b.compileShader("softmax", softmaxShader)
	pipeline := b.getOrCreatePipeline("softmax", shader)

	// Create output buffer (stays on GPU!)
	resultSize := t.ByteSize()
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(batchSize)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(featureSize)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, t.buffer, 0, t.bufferSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Dispatch workgroups: one per batch element

	workgroups := uint32(batchSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Return GPUTensor (NO readBuffer!)
	return &GPUTensor{
		buffer:     bufferResult,
		shape:      t.shape,
		dtype:      t.dtype,
		strides:    t.strides,
		backend:    b,
		computed:   true,
		bufferSize: resultSize,
	}
}

// UploadTensor uploads a CPU tensor to GPU memory.
// Returns a GPUTensor that can be used for lazy GPU operations.
func (b *Backend) UploadTensor(raw *tensor.RawTensor) *GPUTensor {
	// Calculate aligned buffer size (WebGPU requires 4-byte alignment)
	numElements := raw.NumElements()
	bytesPerElement := raw.DType().Size()
	actualByteSize := numElements * bytesPerElement

	//nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	alignedSize := uint64((actualByteSize + 3) &^ 3) // Round up to 4-byte boundary

	// Create GPU buffer
	buffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:             alignedSize,
		Usage:            gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		MappedAtCreation: wgpu.True,
	})

	// Copy data to GPU
	mappedPtr := buffer.GetMappedRange(0, alignedSize)
	mappedSlice := unsafe.Slice((*byte)(mappedPtr), alignedSize) //nolint:gosec // G103: Required for GPU buffer access
	copy(mappedSlice, raw.Data()[:actualByteSize])
	buffer.Unmap()

	return &GPUTensor{
		buffer:     buffer,
		shape:      raw.Shape(),
		dtype:      raw.DType(),
		strides:    raw.Shape().ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: alignedSize,
	}
}

// runUnaryOpGPU executes unary operations on GPU tensors without CPU transfer.
func (b *Backend) runUnaryOpGPU(t *GPUTensor, opName, shaderCode string) *GPUTensor {
	numElements := t.NumElements()

	// Compile shader
	shader := b.compileShader(opName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(opName, shader)

	// Create output buffer (stays on GPU!)
	resultSize := t.ByteSize()
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, t.buffer, 0, t.bufferSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count: ceil(numElements / workgroupSize)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Return GPUTensor (NO readBuffer!)
	return &GPUTensor{
		buffer:     bufferResult,
		shape:      t.shape,
		dtype:      t.dtype,
		strides:    t.strides,
		backend:    b,
		computed:   true,
		bufferSize: resultSize,
	}
}
