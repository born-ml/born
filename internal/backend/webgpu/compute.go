//go:build windows

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
	"github.com/gogpu/gputypes"
)

// compileShader compiles WGSL shader code into a ShaderModule.
// Results are cached in the Backend's shaders map.
func (b *Backend) compileShader(name, code string) *wgpu.ShaderModule {
	b.mu.RLock()
	if shader, exists := b.shaders[name]; exists {
		b.mu.RUnlock()
		return shader
	}
	b.mu.RUnlock()

	// Compile shader
	shader := b.device.CreateShaderModuleWGSL(code)

	// Cache it
	b.mu.Lock()
	b.shaders[name] = shader
	b.mu.Unlock()

	return shader
}

// getOrCreatePipeline returns a cached ComputePipeline or creates a new one.
func (b *Backend) getOrCreatePipeline(name string, shader *wgpu.ShaderModule) *wgpu.ComputePipeline {
	b.mu.RLock()
	if pipeline, exists := b.pipelines[name]; exists {
		b.mu.RUnlock()
		return pipeline
	}
	b.mu.RUnlock()

	// Create compute pipeline with auto layout (nil layout)
	pipeline := b.device.CreateComputePipelineSimple(nil, shader, "main")

	// Cache it
	b.mu.Lock()
	b.pipelines[name] = pipeline
	b.mu.Unlock()

	return pipeline
}

// createBuffer creates a GPU buffer and optionally uploads initial data.
func (b *Backend) createBuffer(data []byte, usage gputypes.BufferUsage) *wgpu.Buffer {
	size := uint64(len(data))

	// Create buffer with MappedAtCreation for initial data upload
	buffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage:            usage,
		Size:             size,
		MappedAtCreation: wgpu.True,
	})

	// Copy data to mapped buffer
	mappedPtr := buffer.GetMappedRange(0, size)

	mappedSlice := unsafe.Slice((*byte)(mappedPtr), size) //nolint:gosec // G103: Required for GPU buffer access
	copy(mappedSlice, data)
	buffer.Unmap()

	return buffer
}

// createUniformBuffer creates a uniform buffer with proper alignment.
// Uniform buffers require 16-byte alignment for struct fields.
func (b *Backend) createUniformBuffer(data []byte) *wgpu.Buffer {
	// Ensure 16-byte alignment
	size := uint64(len(data))
	alignedSize := (size + 15) &^ 15 // Round up to 16-byte boundary

	buffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage:            gputypes.BufferUsageUniform | gputypes.BufferUsageCopyDst,
		Size:             alignedSize,
		MappedAtCreation: wgpu.True,
	})

	// Copy data (padding is handled by aligned size)
	mappedPtr := buffer.GetMappedRange(0, alignedSize)

	mappedSlice := unsafe.Slice((*byte)(mappedPtr), alignedSize) //nolint:gosec // G103: Required for GPU buffer access
	copy(mappedSlice, data)
	buffer.Unmap()

	return buffer
}

// readBuffer reads data back from a GPU buffer to CPU memory.
// Uses a staging buffer since storage buffers can't be mapped directly.
func (b *Backend) readBuffer(srcBuffer *wgpu.Buffer, size uint64) ([]byte, error) {
	// Flush all pending commands first to ensure GPU operations are complete.
	// This is critical for command batching - we need all compute passes to finish
	// before we can read their results.
	b.flushCommands()

	// Create staging buffer for reading (MAP_READ | COPY_DST)
	stagingBuffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst,
		Size:  size,
	})
	defer stagingBuffer.Release()

	// Copy from GPU buffer to staging buffer
	encoder := b.device.CreateCommandEncoder(nil)
	encoder.CopyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, size)
	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Map staging buffer for reading
	err := stagingBuffer.MapAsync(b.device, wgpu.MapModeRead, 0, size)
	if err != nil {
		return nil, fmt.Errorf("failed to map staging buffer: %w", err)
	}

	// Get mapped range and copy data
	mappedPtr := stagingBuffer.GetMappedRange(0, size)

	mappedSlice := unsafe.Slice((*byte)(mappedPtr), size) //nolint:gosec // G103: Required for GPU buffer access
	result := make([]byte, size)
	copy(result, mappedSlice)

	// Unmap buffer
	stagingBuffer.Unmap()

	return result, nil
}

// runBinaryOp executes a binary element-wise operation (add, sub, mul, div) on GPU.
// Supports NumPy-style broadcasting. Supports float32 and int32 dtypes.
func (b *Backend) runBinaryOp(a, other *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate inputs - must have same dtype
	if a.DType() != other.DType() {
		return nil, fmt.Errorf("webgpu: dtype mismatch: %s vs %s", a.DType(), other.DType())
	}

	// Only float32 and int32 are supported
	dtype := a.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: only float32 and int32 are supported, got %s", dtype)
	}

	// Handle broadcasting if shapes don't match
	if !a.Shape().Equal(other.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(a.Shape(), other.Shape())
		if !ok {
			return nil, fmt.Errorf("webgpu: shapes not broadcastable: %v vs %v", a.Shape(), other.Shape())
		}
		// Expand tensors to broadcasted shape
		if !a.Shape().Equal(broadcastedShape) {
			a = b.Expand(a, broadcastedShape)
		}
		if !other.Shape().Equal(broadcastedShape) {
			other = b.Expand(other, broadcastedShape)
		}
	}

	numElements := a.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferOther := b.createBuffer(other.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferOther.Release()

	resultSize := uint64(a.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferOther, 0, resultSize),
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

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(a.Shape(), a.DType(), tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runComparisonOp executes a binary comparison operation (greater, lower, equal, etc.) on GPU.
// Always returns float32 result (0.0 for false, 1.0 for true).
// Converts int32 inputs to float32 before comparison.
func (b *Backend) runComparisonOp(a, other *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate inputs - must have same dtype
	if a.DType() != other.DType() {
		return nil, fmt.Errorf("webgpu: dtype mismatch: %s vs %s", a.DType(), other.DType())
	}

	// Only float32 and int32 are supported
	dtype := a.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: only float32 and int32 are supported, got %s", dtype)
	}

	// Convert int32 to float32 for comparison shaders (they only support f32)
	if dtype == tensor.Int32 {
		var err error
		a, err = int32ToFloat32(a)
		if err != nil {
			return nil, err
		}
		other, err = int32ToFloat32(other)
		if err != nil {
			return nil, err
		}
	}

	// Handle broadcasting if shapes don't match
	if !a.Shape().Equal(other.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(a.Shape(), other.Shape())
		if !ok {
			return nil, fmt.Errorf("webgpu: shapes not broadcastable: %v vs %v", a.Shape(), other.Shape())
		}
		// Expand tensors to broadcasted shape
		if !a.Shape().Equal(broadcastedShape) {
			a = b.Expand(a, broadcastedShape)
		}
		if !other.Shape().Equal(broadcastedShape) {
			other = b.Expand(other, broadcastedShape)
		}
	}

	numElements := a.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferOther := b.createBuffer(other.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferOther.Release()

	resultSize := uint64(a.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferOther, 0, resultSize),
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

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor - ALWAYS float32 for comparison results (0.0/1.0 for bool)
	result, err := tensor.NewRaw(a.Shape(), tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runUnaryOp executes a unary element-wise operation (relu, sigmoid, tanh, neg, exp, log, sqrt) on GPU.
func (b *Backend) runUnaryOp(input *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate input
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", input.DType())
	}

	numElements := input.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(input.Shape(), input.DType(), tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runMatMul executes matrix multiplication C = A @ B on GPU.
// A is [M, K], B is [K, N], C is [M, N].
func (b *Backend) runMatMul(a, other *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate inputs
	if a.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", a.DType())
	}
	if len(a.Shape()) != 2 || len(other.Shape()) != 2 {
		return nil, fmt.Errorf("webgpu: matmul requires 2D tensors, got %v and %v", a.Shape(), other.Shape())
	}

	M := uint32(a.Shape()[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	K := uint32(a.Shape()[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	N := uint32(other.Shape()[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	if other.Shape()[0] != int(K) {
		return nil, fmt.Errorf("webgpu: matmul shape mismatch: [%d,%d] @ [%d,%d]", M, K, other.Shape()[0], N)
	}

	// Compile shader
	shader := b.compileShader("matmul", matmulShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("matmul", shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferOther := b.createBuffer(other.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferOther.Release()

	resultShape := tensor.Shape{int(M), int(N)}

	//nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	resultSize := uint64(int(M) * int(N) * 4) // float32 = 4 bytes
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (M, K, N: u32 each)
	params := make([]byte, 16) // 16-byte aligned (3 u32 = 12 bytes, padded to 16)
	binary.LittleEndian.PutUint32(params[0:4], M)
	binary.LittleEndian.PutUint32(params[4:8], K)
	binary.LittleEndian.PutUint32(params[8:12], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, uint64(a.ByteSize())),         //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferOther, 0, uint64(other.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count for 2D workgroups (16x16 per workgroup)
	workgroupsX := uint32(math.Ceil(float64(N) / 16.0))
	workgroupsY := uint32(math.Ceil(float64(M) / 16.0))
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runTranspose executes 2D matrix transpose on GPU.
func (b *Backend) runTranspose(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate input
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", input.DType())
	}
	if len(input.Shape()) != 2 {
		return nil, fmt.Errorf("webgpu: transpose requires 2D tensor, got %v", input.Shape())
	}

	rows := uint32(input.Shape()[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	cols := uint32(input.Shape()[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	// Compile shader
	shader := b.compileShader("transpose", transposeShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("transpose", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (rows, cols: u32 each)
	params := make([]byte, 16) // 16-byte aligned
	binary.LittleEndian.PutUint32(params[0:4], rows)
	binary.LittleEndian.PutUint32(params[4:8], cols)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count for 2D workgroups (16x16 per workgroup)
	workgroupsX := uint32(math.Ceil(float64(cols) / 16.0))
	workgroupsY := uint32(math.Ceil(float64(rows) / 16.0))
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor with transposed shape
	resultShape := tensor.Shape{int(cols), int(rows)}
	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runScalarOp executes a scalar operation (MulScalar, AddScalar, etc.) on GPU.
// The shader params contain both size (u32) and scalar value (f32).
func (b *Backend) runScalarOp(input *tensor.RawTensor, scalar float32, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate input
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", input.DType())
	}

	numElements := input.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32, scalar: f32)
	params := make([]byte, 16) // 16-byte aligned

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	binary.LittleEndian.PutUint32(params[4:8], math.Float32bits(scalar))
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(input.Shape(), input.DType(), tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runSoftmax executes softmax along the last dimension on GPU.
// Input shape: [batch_size, num_classes].
func (b *Backend) runSoftmax(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate input
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", input.DType())
	}
	if len(input.Shape()) != 2 {
		return nil, fmt.Errorf("webgpu: softmax requires 2D tensor, got %v", input.Shape())
	}

	batchSize := uint32(input.Shape()[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	numClasses := uint32(input.Shape()[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	// Compile shader
	shader := b.compileShader("softmax", softmaxShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("softmax", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (batch_size, num_classes: u32 each)
	params := make([]byte, 16) // 16-byte aligned
	binary.LittleEndian.PutUint32(params[0:4], batchSize)
	binary.LittleEndian.PutUint32(params[4:8], numClasses)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Each workgroup handles one row (batch sample)
	workgroups := (batchSize + workgroupSize - 1) / workgroupSize
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(input.Shape(), tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runBatchMatMul executes batched matrix multiplication on GPU.
// Supports 3D [batch, M, K] @ [batch, K, N] -> [batch, M, N]
// and 4D [batch, heads, M, K] @ [batch, heads, K, N] -> [batch, heads, M, N].
func (b *Backend) runBatchMatMul(a, other *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate inputs
	if a.DType() != tensor.Float32 || other.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported")
	}

	shapeA := a.Shape()
	shapeB := other.Shape()

	if len(shapeA) != len(shapeB) || (len(shapeA) != 3 && len(shapeA) != 4) {
		return nil, fmt.Errorf("webgpu: BatchMatMul requires 3D or 4D tensors with matching dimensions")
	}

	var batch, M, K, N uint32
	var resultShape tensor.Shape

	if len(shapeA) == 3 {
		// 3D: [batch, M, K] @ [batch, K, N]

		batch = uint32(shapeA[0]) //nolint:gosec // G115: Cast operation - safe conversion.

		M = uint32(shapeA[1]) //nolint:gosec // G115: Cast operation - safe conversion.

		K = uint32(shapeA[2]) //nolint:gosec // G115: Cast operation - safe conversion.

		N = uint32(shapeB[2]) //nolint:gosec // G115: Cast operation - safe conversion.
		resultShape = tensor.Shape{int(batch), int(M), int(N)}
	} else {
		// 4D: [batch, heads, M, K] @ [batch, heads, K, N]
		// Treat as [batch*heads, M, K] @ [batch*heads, K, N]

		batch = uint32(shapeA[0] * shapeA[1]) //nolint:gosec // G115: Cast operation - safe conversion.

		M = uint32(shapeA[2]) //nolint:gosec // G115: Cast operation - safe conversion.

		K = uint32(shapeA[3]) //nolint:gosec // G115: Cast operation - safe conversion.

		N = uint32(shapeB[3]) //nolint:gosec // G115: Cast operation - safe conversion.
		resultShape = tensor.Shape{shapeA[0], shapeA[1], int(M), int(N)}
	}

	// Compile shader
	shader := b.compileShader("batchMatMul", batchMatMulShader)
	pipeline := b.getOrCreatePipeline("batchMatMul", shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferB := b.createBuffer(other.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferB.Release()

	resultSize := uint64(batch) * uint64(M) * uint64(N) * 4 // float32 = 4 bytes
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:4], batch)
	binary.LittleEndian.PutUint32(params[4:8], M)
	binary.LittleEndian.PutUint32(params[8:12], K)
	binary.LittleEndian.PutUint32(params[12:16], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, uint64(a.ByteSize())),     //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferB, 0, uint64(other.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Dispatch: (N+7)/8 x (M+7)/8 x batch
	workgroupsX := (N + 7) / 8
	workgroupsY := (M + 7) / 8
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, batch)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runConv2D executes 2D convolution on GPU.
// Input shape: [batch, in_channels, height, width].
// Kernel shape: [out_channels, in_channels, kH, kW].
func (b *Backend) runConv2D(input, kernel *tensor.RawTensor, stride, padding int) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 || kernel.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported")
	}

	inShape := input.Shape()
	kShape := kernel.Shape()

	if len(inShape) != 4 || len(kShape) != 4 {
		return nil, fmt.Errorf("webgpu: Conv2D requires 4D tensors")
	}

	batchSize := uint32(inShape[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	inChannels := uint32(inShape[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	inHeight := uint32(inShape[2]) //nolint:gosec // G115: Cast operation - safe conversion.

	inWidth := uint32(inShape[3]) //nolint:gosec // G115: Cast operation - safe conversion.

	outChannels := uint32(kShape[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	kernelH := uint32(kShape[2]) //nolint:gosec // G115: Cast operation - safe conversion.

	kernelW := uint32(kShape[3]) //nolint:gosec // G115: Cast operation - safe conversion.

	// Calculate output dimensions

	outHeight := (inHeight+2*uint32(padding)-kernelH)/uint32(stride) + 1 //nolint:gosec // G115: Cast operation - safe conversion.

	outWidth := (inWidth+2*uint32(padding)-kernelW)/uint32(stride) + 1 //nolint:gosec // G115: Cast operation - safe conversion.

	resultShape := tensor.Shape{int(batchSize), int(outChannels), int(outHeight), int(outWidth)}

	// Compile shader
	shader := b.compileShader("conv2d", conv2dShader)
	pipeline := b.getOrCreatePipeline("conv2d", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	bufferKernel := b.createBuffer(kernel.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferKernel.Release()

	resultSize := uint64(batchSize) * uint64(outChannels) * uint64(outHeight) * uint64(outWidth) * 4
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (36 bytes, padded to 48)
	params := make([]byte, 48)
	binary.LittleEndian.PutUint32(params[0:4], batchSize)
	binary.LittleEndian.PutUint32(params[4:8], inChannels)
	binary.LittleEndian.PutUint32(params[8:12], inHeight)
	binary.LittleEndian.PutUint32(params[12:16], inWidth)
	binary.LittleEndian.PutUint32(params[16:20], outChannels)
	binary.LittleEndian.PutUint32(params[20:24], kernelH)
	binary.LittleEndian.PutUint32(params[24:28], kernelW)

	binary.LittleEndian.PutUint32(params[28:32], uint32(stride)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[32:36], uint32(padding)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())),   //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferKernel, 0, uint64(kernel.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 48),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroupsX := (outWidth + 7) / 8
	workgroupsY := (outHeight + 7) / 8
	workgroupsZ := batchSize * outChannels
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runMaxPool2D executes 2D max pooling on GPU.
// Input shape: [batch, channels, height, width].
func (b *Backend) runMaxPool2D(input *tensor.RawTensor, kernelSize, stride int) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported")
	}

	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, fmt.Errorf("webgpu: MaxPool2D requires 4D tensor")
	}

	batchSize := uint32(inShape[0]) //nolint:gosec // G115: Cast operation - safe conversion.

	channels := uint32(inShape[1]) //nolint:gosec // G115: Cast operation - safe conversion.

	inHeight := uint32(inShape[2]) //nolint:gosec // G115: Cast operation - safe conversion.

	inWidth := uint32(inShape[3]) //nolint:gosec // G115: Cast operation - safe conversion.

	kSize := uint32(kernelSize) //nolint:gosec // G115: Cast operation - safe conversion.

	outHeight := (inHeight-kSize)/uint32(stride) + 1 //nolint:gosec // G115: Cast operation - safe conversion.

	outWidth := (inWidth-kSize)/uint32(stride) + 1 //nolint:gosec // G115: Cast operation - safe conversion.

	resultShape := tensor.Shape{int(batchSize), int(channels), int(outHeight), int(outWidth)}

	// Compile shader
	shader := b.compileShader("maxPool2d", maxPool2dShader)
	pipeline := b.getOrCreatePipeline("maxPool2d", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(batchSize) * uint64(channels) * uint64(outHeight) * uint64(outWidth) * 4
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params
	params := make([]byte, 32)
	binary.LittleEndian.PutUint32(params[0:4], batchSize)
	binary.LittleEndian.PutUint32(params[4:8], channels)
	binary.LittleEndian.PutUint32(params[8:12], inHeight)
	binary.LittleEndian.PutUint32(params[12:16], inWidth)
	binary.LittleEndian.PutUint32(params[16:20], kSize)
	binary.LittleEndian.PutUint32(params[20:24], kSize)

	binary.LittleEndian.PutUint32(params[24:28], uint32(stride)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 32),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroupsX := (outWidth + 7) / 8
	workgroupsY := (outHeight + 7) / 8
	workgroupsZ := batchSize * channels
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runSum executes global sum reduction on GPU.
// Supports float32 and int32 dtypes.
func (b *Backend) runSum(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	dtype := input.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: only float32 and int32 are supported, got %s", dtype)
	}

	numElements := input.NumElements()

	// For small tensors, use CPU
	if numElements < 1024 {
		return b.runSumCPU(input)
	}

	// GPU parallel reduction
	return b.runSumGPU(input)
}

// runSumCPU executes sum on CPU for small tensors.
func (b *Backend) runSumCPU(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	dtype := input.DType()

	switch dtype {
	case tensor.Float32:
		data := input.AsFloat32()
		var sum float32
		for _, v := range data {
			sum += v
		}
		result, err := tensor.NewRaw(tensor.Shape{}, tensor.Float32, tensor.WebGPU)
		if err != nil {
			return nil, err
		}
		result.AsFloat32()[0] = sum
		return result, nil

	case tensor.Int32:
		data := input.AsInt32()
		var sum int32
		for _, v := range data {
			sum += v
		}
		result, err := tensor.NewRaw(tensor.Shape{}, tensor.Int32, tensor.WebGPU)
		if err != nil {
			return nil, err
		}
		result.AsInt32()[0] = sum
		return result, nil

	default:
		return nil, fmt.Errorf("webgpu: unsupported dtype for Sum: %s", dtype)
	}
}

// runSumGPU executes sum on GPU for large tensors.
func (b *Backend) runSumGPU(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	dtype := input.DType()
	numElements := input.NumElements()

	// Select shader based on dtype
	var shaderName string
	var shaderCode string
	switch dtype {
	case tensor.Float32:
		shaderName = "globalSum"
		shaderCode = globalSumShader
	case tensor.Int32:
		shaderName = "globalSumInt32"
		shaderCode = globalSumShaderInt32
	default:
		return nil, fmt.Errorf("webgpu: unsupported dtype for Sum: %s", dtype)
	}

	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	// Calculate number of workgroups needed

	numWorkgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	partialSumsSize := uint64(numWorkgroups) * 4

	bufferPartialSums := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  partialSumsSize,
	})
	defer bufferPartialSums.Release()

	// Create uniform buffer for params
	params := make([]byte, 16)

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferPartialSums, 0, partialSumsSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)
	computePass.DispatchWorkgroups(numWorkgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read partial sums and sum on CPU
	partialData, err := b.readBuffer(bufferPartialSums, partialSumsSize)
	if err != nil {
		return nil, err
	}

	// Sum partial results on CPU based on dtype
	switch dtype {
	case tensor.Float32:
		var sum float32
		for i := uint32(0); i < numWorkgroups; i++ {
			sum += math.Float32frombits(binary.LittleEndian.Uint32(partialData[i*4 : i*4+4]))
		}
		result, err := tensor.NewRaw(tensor.Shape{}, tensor.Float32, tensor.WebGPU)
		if err != nil {
			return nil, err
		}
		result.AsFloat32()[0] = sum
		return result, nil

	case tensor.Int32:
		var sum int32
		for i := uint32(0); i < numWorkgroups; i++ {
			sum += int32(binary.LittleEndian.Uint32(partialData[i*4 : i*4+4])) //nolint:gosec // G115: Cast operation - safe conversion.
		}
		result, err := tensor.NewRaw(tensor.Shape{}, tensor.Int32, tensor.WebGPU)
		if err != nil {
			return nil, err
		}
		result.AsInt32()[0] = sum
		return result, nil

	default:
		return nil, fmt.Errorf("webgpu: unsupported dtype for Sum: %s", dtype)
	}
}

// runArgmax executes argmax along last dimension on GPU.
func (b *Backend) runArgmax(input *tensor.RawTensor, dim int) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported")
	}

	shape := input.Shape()
	ndim := len(shape)

	// Normalize dimension
	if dim < 0 {
		dim = ndim + dim
	}

	// Currently only supports last dimension
	if dim != ndim-1 {
		return nil, fmt.Errorf("webgpu: Argmax currently only supports last dimension (dim=-1)")
	}

	// Calculate batch size (product of all dimensions except last)
	batchSize := 1
	for i := 0; i < ndim-1; i++ {
		batchSize *= shape[i]
	}
	dimSize := shape[ndim-1]

	// Result shape: remove last dimension
	var resultShape tensor.Shape
	if ndim > 1 {
		resultShape = make(tensor.Shape, ndim-1)
		copy(resultShape, shape[:ndim-1])
	} else {
		resultShape = tensor.Shape{1}
	}

	shader := b.compileShader("argmax", argmaxShader)
	pipeline := b.getOrCreatePipeline("argmax", shader)

	// Create buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(batchSize) * 4 //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer
	params := make([]byte, 16)

	binary.LittleEndian.PutUint32(params[0:4], uint32(batchSize)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(dimSize)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((batchSize + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	result, err := tensor.NewRaw(resultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runEmbedding performs embedding lookup on GPU.
// weight: [num_embeddings, embedding_dim], indices: [...], output: [..., embedding_dim].
func (b *Backend) runEmbedding(weight, indices *tensor.RawTensor) (*tensor.RawTensor, error) {
	if weight.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: Embedding weight must be float32, got %s", weight.DType())
	}
	if indices.DType() != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: Embedding indices must be int32, got %s", indices.DType())
	}
	if len(weight.Shape()) != 2 {
		return nil, fmt.Errorf("webgpu: Embedding weight must be 2D, got %v", weight.Shape())
	}

	numEmbeddings := weight.Shape()[0]
	embeddingDim := weight.Shape()[1]
	numIndices := indices.NumElements()

	// Output shape: [...indices_shape, embedding_dim]
	indicesShape := indices.Shape()
	outputShape := make(tensor.Shape, len(indicesShape)+1)
	copy(outputShape, indicesShape)
	outputShape[len(outputShape)-1] = embeddingDim

	shader := b.compileShader("embedding", embeddingShader)
	pipeline := b.getOrCreatePipeline("embedding", shader)

	// Create buffers
	bufferWeight := b.createBuffer(weight.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferWeight.Release()

	bufferIndices := b.createBuffer(indices.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferIndices.Release()

	resultSize := uint64(numIndices) * uint64(embeddingDim) * 4 //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create params
	params := make([]byte, 16)

	binary.LittleEndian.PutUint32(params[0:4], uint32(numIndices)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(embeddingDim)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[8:12], uint32(numEmbeddings)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferWeight, 0, uint64(weight.ByteSize())),   //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferIndices, 0, uint64(indices.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)
	totalElements := numIndices * embeddingDim
	workgroups := uint32((totalElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	result, err := tensor.NewRaw(outputShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// boolToFloat32 converts a bool tensor to float32 for GPU operations.
// WGSL doesn't have native bool arrays, so we use float32 (0.0 = false, 1.0 = true).
func boolToFloat32(condition *tensor.RawTensor) (*tensor.RawTensor, error) {
	result, err := tensor.NewRaw(condition.Shape(), tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}
	boolData := condition.Data()
	floatData := result.AsFloat32()
	for i := 0; i < condition.NumElements(); i++ {
		if boolData[i] != 0 {
			floatData[i] = 1.0
		} else {
			floatData[i] = 0.0
		}
	}
	return result, nil
}

// int32ToFloat32 converts an int32 tensor to float32 for GPU operations.
// Used for condition tensors in Where operations.
func int32ToFloat32(condition *tensor.RawTensor) (*tensor.RawTensor, error) {
	result, err := tensor.NewRaw(condition.Shape(), tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}
	intData := condition.AsInt32()
	floatData := result.AsFloat32()
	for i, v := range intData {
		floatData[i] = float32(v)
	}
	return result, nil
}

// runWhere performs conditional element selection on GPU.
// result[i] = condition[i] != 0 ? x[i] : y[i].
// Supports float32 and int32 data types. Condition can be bool, float32, or int32.
//
//nolint:funlen,gocognit,gocyclo,cyclop // Complex GPU operation with dtype handling and broadcasting
func (b *Backend) runWhere(condition, x, y *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Convert condition to float32 for GPU
	var condFloat32 *tensor.RawTensor
	var err error
	switch condition.DType() {
	case tensor.Bool:
		condFloat32, err = boolToFloat32(condition)
		if err != nil {
			return nil, err
		}
	case tensor.Float32:
		condFloat32 = condition
	case tensor.Int32:
		// Convert int32 condition to float32
		condFloat32, err = int32ToFloat32(condition)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("webgpu: Where condition must be bool, float32, or int32, got %s", condition.DType())
	}

	// x and y must have same dtype
	if x.DType() != y.DType() {
		return nil, fmt.Errorf("webgpu: Where requires x and y with same dtype, got %s and %s", x.DType(), y.DType())
	}

	// Only float32 and int32 supported
	dtype := x.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: Where supports float32 and int32, got %s", dtype)
	}

	// Handle broadcasting - compute output shape from all 3 tensors (like Burn)
	outShape := condFloat32.Shape()

	// Broadcast condition with x
	if !condFloat32.Shape().Equal(x.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(condFloat32.Shape(), x.Shape())
		if !ok {
			return nil, fmt.Errorf("webgpu: Where condition and x shapes not broadcastable: %v vs %v", condFloat32.Shape(), x.Shape())
		}
		outShape = broadcastedShape
	}

	// Broadcast outShape with y
	if !outShape.Equal(y.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(outShape, y.Shape())
		if !ok {
			return nil, fmt.Errorf("webgpu: Where output and y shapes not broadcastable: %v vs %v", outShape, y.Shape())
		}
		outShape = broadcastedShape
	}

	// Expand all tensors to output shape
	if !condFloat32.Shape().Equal(outShape) {
		condFloat32 = b.Expand(condFloat32, outShape)
	}
	if !x.Shape().Equal(outShape) {
		x = b.Expand(x, outShape)
	}
	if !y.Shape().Equal(outShape) {
		y = b.Expand(y, outShape)
	}

	numElements := condFloat32.NumElements()

	// Select shader based on dtype
	var shaderName, shaderCode string
	if dtype == tensor.Int32 {
		shaderName = "whereInt32"
		shaderCode = whereShaderInt32
	} else {
		shaderName = "where"
		shaderCode = whereShader
	}

	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create buffers
	bufferCondition := b.createBuffer(condFloat32.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferCondition.Release()

	bufferX := b.createBuffer(x.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferX.Release()

	bufferY := b.createBuffer(y.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferY.Release()

	resultSizeWhere := uint64(x.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResultWhere := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSizeWhere,
	})
	defer bufferResultWhere.Release()

	// Create uniform buffer
	paramsWhere := make([]byte, 16)

	binary.LittleEndian.PutUint32(paramsWhere[0:4], uint32(numElements)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParamsWhere := b.createUniformBuffer(paramsWhere)
	defer bufferParamsWhere.Release()

	// Create bind group

	condSizeWhere := uint64(condFloat32.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bindGroupLayoutWhere := pipeline.GetBindGroupLayout(0)
	bindGroupWhere := b.device.CreateBindGroupSimple(bindGroupLayoutWhere, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferCondition, 0, condSizeWhere),
		wgpu.BufferBindingEntry(1, bufferX, 0, resultSizeWhere),
		wgpu.BufferBindingEntry(2, bufferY, 0, resultSizeWhere),
		wgpu.BufferBindingEntry(3, bufferResultWhere, 0, resultSizeWhere),
		wgpu.BufferBindingEntry(4, bufferParamsWhere, 0, 16),
	})
	defer bindGroupWhere.Release()

	// Execute
	encoderWhere := b.device.CreateCommandEncoder(nil)
	computePassWhere := encoderWhere.BeginComputePass(nil)

	computePassWhere.SetPipeline(pipeline)
	computePassWhere.SetBindGroup(0, bindGroupWhere, nil)

	workgroupsWhere := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePassWhere.DispatchWorkgroups(workgroupsWhere, 1, 1)
	computePassWhere.End()

	cmdBufferWhere := encoderWhere.Finish(nil)
	b.queue.Submit(cmdBufferWhere)

	// Read result
	resultDataWhere, err := b.readBuffer(bufferResultWhere, resultSizeWhere)
	if err != nil {
		return nil, err
	}

	resultWhere, err := tensor.NewRaw(x.Shape(), dtype, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(resultWhere.Data(), resultDataWhere)
	return resultWhere, nil
}

// runGather gathers elements along a dimension using indices.
// For dim=-1 (last dimension): input[..., indices[...]] -> result[...].
// input: float32 tensor, indices: int32 tensor (like PyTorch/NumPy).
func (b *Backend) runGather(input *tensor.RawTensor, dim int, indices *tensor.RawTensor) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: Gather input must be float32, got %s", input.DType())
	}
	if indices.DType() != tensor.Int32 {
		return nil, fmt.Errorf("webgpu: Gather indices must be int32, got %s", indices.DType())
	}

	inShape := input.Shape()
	idxShape := indices.Shape()
	ndim := len(inShape)

	// Normalize dimension
	if dim < 0 {
		dim = ndim + dim
	}

	// For non-last dimensions: transpose → gather → transpose back
	if dim != ndim-1 {
		return b.gatherNonLastDim(input, dim, indices)
	}

	// Calculate batch size (product of all dimensions except last)
	gatherBatchSize := 1
	for i := 0; i < ndim-1; i++ {
		gatherBatchSize *= inShape[i]
	}
	inputDim := inShape[ndim-1]

	// Output K is the size of the last dimension of indices
	outputK := idxShape[len(idxShape)-1]

	// Result shape: batch dimensions + outputK
	gatherResultShape := make(tensor.Shape, ndim)
	copy(gatherResultShape, inShape[:ndim-1])
	gatherResultShape[ndim-1] = outputK

	shaderGather := b.compileShader("gather", gatherShader)
	pipelineGather := b.getOrCreatePipeline("gather", shaderGather)

	// Create buffers
	bufferInputGather := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInputGather.Release()

	bufferIndices := b.createBuffer(indices.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferIndices.Release()

	gatherResultSize := uint64(gatherBatchSize) * uint64(outputK) * 4 //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResultGather := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  gatherResultSize,
	})
	defer bufferResultGather.Release()

	// Create uniform buffer
	paramsGather := make([]byte, 16)

	binary.LittleEndian.PutUint32(paramsGather[0:4], uint32(gatherBatchSize)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(paramsGather[4:8], uint32(inputDim)) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(paramsGather[8:12], uint32(outputK)) //nolint:gosec // G115: Cast operation - safe conversion.
	bufferParamsGather := b.createUniformBuffer(paramsGather)
	defer bufferParamsGather.Release()

	// Create bind group
	bindGroupLayoutGather := pipelineGather.GetBindGroupLayout(0)

	bindGroupGather := b.device.CreateBindGroupSimple(bindGroupLayoutGather, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInputGather, 0, uint64(input.ByteSize())), //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(1, bufferIndices, 0, uint64(indices.ByteSize())),   //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
		wgpu.BufferBindingEntry(2, bufferResultGather, 0, gatherResultSize),
		wgpu.BufferBindingEntry(3, bufferParamsGather, 0, 16),
	})
	defer bindGroupGather.Release()

	// Execute
	encoderGather := b.device.CreateCommandEncoder(nil)
	computePassGather := encoderGather.BeginComputePass(nil)

	computePassGather.SetPipeline(pipelineGather)
	computePassGather.SetBindGroup(0, bindGroupGather, nil)
	totalOutputGather := gatherBatchSize * outputK

	workgroupsGather := uint32((totalOutputGather + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: Cast operation - safe conversion.
	computePassGather.DispatchWorkgroups(workgroupsGather, 1, 1)
	computePassGather.End()

	cmdBufferGather := encoderGather.Finish(nil)
	b.queue.Submit(cmdBufferGather)

	// Read result
	resultDataGather, err := b.readBuffer(bufferResultGather, gatherResultSize)
	if err != nil {
		return nil, err
	}

	resultGather, err := tensor.NewRaw(gatherResultShape, tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(resultGather.Data(), resultDataGather)
	return resultGather, nil
}

// gatherNonLastDim handles Gather on non-last dimensions via transpose.
// Approach: transpose input & indices → gather on last dim → transpose back.
func (b *Backend) gatherNonLastDim(input *tensor.RawTensor, dim int, indices *tensor.RawTensor) (*tensor.RawTensor, error) {
	inShape := input.Shape()
	ndim := len(inShape)

	// Build transpose axes: move dim to last position
	// e.g., for dim=1, ndim=3: [0, 2, 1] (swap 1 and 2)
	axes := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		axes[i] = i
	}
	axes[dim] = ndim - 1
	axes[ndim-1] = dim

	// Transpose input: move target dim to last
	transposedInput := b.Transpose(input, axes...)

	// Transpose indices with same axes (indices must match input dimensions)
	transposedIndices := b.transposeInt32(indices, axes)

	// Gather on last dimension
	gathered, err := b.runGather(transposedInput, -1, transposedIndices)
	if err != nil {
		return nil, err
	}

	// Transpose back
	result := b.Transpose(gathered, axes...) // Same axes work for inverse (swap is symmetric)
	return result, nil
}

// transposeInt32 transposes an int32 tensor.
func (b *Backend) transposeInt32(t *tensor.RawTensor, axes []int) *tensor.RawTensor {
	shape := t.Shape()
	ndim := len(shape)

	// Compute new shape
	newShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		newShape[i] = shape[ax]
	}

	// Create result
	result, err := tensor.NewRaw(newShape, tensor.Int32, tensor.WebGPU)
	if err != nil {
		panic("webgpu: transposeInt32: " + err.Error())
	}

	// Transpose on CPU
	srcData := t.AsInt32()
	dstData := result.AsInt32()
	srcStrides := shape.ComputeStrides()
	dstStrides := newShape.ComputeStrides()
	numElements := shape.NumElements()

	for i := 0; i < numElements; i++ {
		// Convert flat index to dst coordinates
		dstIdx := i
		coords := make([]int, ndim)
		for d := 0; d < ndim; d++ {
			coords[d] = dstIdx / dstStrides[d]
			dstIdx %= dstStrides[d]
		}

		// Map to src index
		srcIdx := 0
		for d := 0; d < ndim; d++ {
			srcIdx += coords[d] * srcStrides[axes[d]]
		}

		dstData[i] = srcData[srcIdx]
	}

	return result
}

// runTransposeND executes N-dimensional matrix transpose on GPU.
// Supports up to 6D tensors with arbitrary axes permutation.
//
//nolint:gocognit,gocyclo,cyclop,funlen // Complex GPU setup logic - unavoidable for parameter packing
func (b *Backend) runTransposeND(input *tensor.RawTensor, axes []int) (*tensor.RawTensor, error) {
	shape := input.Shape()
	ndim := len(shape)

	if ndim > 6 {
		return nil, fmt.Errorf("webgpu: transposeND supports up to 6D tensors, got %dD", ndim)
	}

	// Default axes: reverse all dimensions
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := 0; i < ndim; i++ {
			axes[i] = ndim - 1 - i
		}
	}

	if len(axes) != ndim {
		return nil, fmt.Errorf("webgpu: transpose axes length must match tensor dimensions")
	}

	// Validate axes
	seen := make(map[int]bool)
	for _, ax := range axes {
		if ax < 0 || ax >= ndim {
			return nil, fmt.Errorf("webgpu: axis %d out of range for %dD tensor", ax, ndim)
		}
		if seen[ax] {
			return nil, fmt.Errorf("webgpu: duplicate axis %d", ax)
		}
		seen[ax] = true
	}

	// Compute new shape
	newShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		newShape[i] = shape[ax]
	}

	// Choose shader based on dtype
	var shaderName, shaderCode string
	switch input.DType() {
	case tensor.Float32:
		shaderName = "transposeND"
		shaderCode = transposeNDShader
	case tensor.Int32:
		shaderName = "transposeND_int32"
		shaderCode = transposeNDShaderInt32
	default:
		return nil, fmt.Errorf("webgpu: transposeND unsupported dtype %s", input.DType())
	}

	// Compile shader and get pipeline
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params
	// Layout: ndim, total_elements, shapes[6], input_strides[6], output_strides[6], axes[6]
	params := make([]byte, 4*26) // 26 u32 values * 4 bytes
	inputStrides := shape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	binary.LittleEndian.PutUint32(params[0:4], uint32(ndim))

	binary.LittleEndian.PutUint32(params[4:8], uint32(shape.NumElements())) //nolint:gosec // G115: Cast operation - safe conversion.

	// Pack input shape (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(shape) {
			binary.LittleEndian.PutUint32(params[8+i*4:12+i*4], uint32(shape[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			binary.LittleEndian.PutUint32(params[32+i*4:36+i*4], uint32(inputStrides[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			binary.LittleEndian.PutUint32(params[56+i*4:60+i*4], uint32(outputStrides[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[56+i*4:60+i*4], 1)
		}
	}

	// Pack axes (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(axes) {
			binary.LittleEndian.PutUint32(params[80+i*4:84+i*4], uint32(axes[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[80+i*4:84+i*4], 0)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	paramsSize := uint64(len(params))
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, paramsSize),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count (1D workgroups, 256 threads each)

	numElements := uint32(shape.NumElements()) //nolint:gosec // G115: Cast operation - safe conversion.
	workgroups := uint32(math.Ceil(float64(numElements) / 256.0))
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor with transposed shape
	result, err := tensor.NewRaw(newShape, input.DType(), tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}

// runExpand executes NumPy-style broadcasting on GPU.
// Expands tensor to new shape by broadcasting dimensions of size 1.
//
//nolint:gocognit,gocyclo,cyclop,funlen // Complex GPU setup logic - unavoidable for parameter packing
func (b *Backend) runExpand(input *tensor.RawTensor, newShape tensor.Shape) (*tensor.RawTensor, error) {
	shape := input.Shape()

	// Validate shapes are compatible for broadcasting
	if len(newShape) < len(shape) {
		return nil, fmt.Errorf("webgpu: expand new shape must have at least as many dimensions")
	}

	if len(newShape) > 6 {
		return nil, fmt.Errorf("webgpu: expand supports up to 6D tensors, got %dD", len(newShape))
	}

	// Pad source shape to match destination dimensions
	dimDiff := len(newShape) - len(shape)
	paddedShape := make(tensor.Shape, len(newShape))
	for i := 0; i < dimDiff; i++ {
		paddedShape[i] = 1
	}
	for i := 0; i < len(shape); i++ {
		paddedShape[dimDiff+i] = shape[i]
	}

	// Validate broadcasting compatibility
	for i := 0; i < len(newShape); i++ {
		if paddedShape[i] != 1 && paddedShape[i] != newShape[i] {
			return nil, fmt.Errorf("webgpu: expand incompatible shapes: %v -> %v", shape, newShape)
		}
	}

	// Choose shader based on dtype
	var shaderName, shaderCode string
	switch input.DType() {
	case tensor.Float32:
		shaderName = "expand"
		shaderCode = expandShader
	case tensor.Int32:
		shaderName = "expand_int32"
		shaderCode = expandShaderInt32
	default:
		return nil, fmt.Errorf("webgpu: expand unsupported dtype %s", input.DType())
	}

	// Compile shader and get pipeline
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	defer bufferInput.Release()

	// Calculate result size
	resultNumElements := newShape.NumElements()

	elementSize := uint64(input.DType().Size()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.

	resultSize := uint64(resultNumElements) * elementSize //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.

	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params
	// Layout: ndim, total_elements, input_shape[6], input_strides[6], output_strides[6]
	params := make([]byte, 4*20) // 20 u32 values * 4 bytes
	inputStrides := paddedShape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	binary.LittleEndian.PutUint32(params[0:4], uint32(len(newShape))) //nolint:gosec // G115: Cast operation - safe conversion.

	binary.LittleEndian.PutUint32(params[4:8], uint32(resultNumElements)) //nolint:gosec // G115: Cast operation - safe conversion.

	// Pack input shape (6 slots) - use paddedShape
	for i := 0; i < 6; i++ {
		if i < len(paddedShape) {
			binary.LittleEndian.PutUint32(params[8+i*4:12+i*4], uint32(paddedShape[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			binary.LittleEndian.PutUint32(params[32+i*4:36+i*4], uint32(inputStrides[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			binary.LittleEndian.PutUint32(params[56+i*4:60+i*4], uint32(outputStrides[i])) //nolint:gosec // G115: Cast operation - safe conversion.
		} else {
			binary.LittleEndian.PutUint32(params[56+i*4:60+i*4], 1)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	inputSize := uint64(input.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations.
	paramsSize2 := uint64(len(params))
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, inputSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, paramsSize2),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroup count (1D workgroups, 256 threads each)
	workgroups := uint32(math.Ceil(float64(resultNumElements) / 256.0))
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result back from GPU
	resultData, err := b.readBuffer(bufferResult, resultSize)
	if err != nil {
		return nil, err
	}

	// Create result tensor
	result, err := tensor.NewRaw(newShape, input.DType(), tensor.WebGPU)
	if err != nil {
		return nil, err
	}

	copy(result.Data(), resultData)
	return result, nil
}
