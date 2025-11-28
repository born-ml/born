//go:build windows

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
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
func (b *Backend) createBuffer(data []byte, usage wgpu.BufferUsage) *wgpu.Buffer {
	size := uint64(len(data))

	// Create buffer with MappedAtCreation for initial data upload
	buffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage:            usage,
		Size:             size,
		MappedAtCreation: wgpu.True,
	})

	// Copy data to mapped buffer
	mappedPtr := buffer.GetMappedRange(0, size)
	//nolint:gosec // unsafe.Slice for zero-copy conversion from unsafe.Pointer
	mappedSlice := unsafe.Slice((*byte)(mappedPtr), size)
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
		Usage:            wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		Size:             alignedSize,
		MappedAtCreation: wgpu.True,
	})

	// Copy data (padding is handled by aligned size)
	mappedPtr := buffer.GetMappedRange(0, alignedSize)
	//nolint:gosec // unsafe.Slice for zero-copy conversion from unsafe.Pointer
	mappedSlice := unsafe.Slice((*byte)(mappedPtr), alignedSize)
	copy(mappedSlice, data)
	buffer.Unmap()

	return buffer
}

// readBuffer reads data back from a GPU buffer to CPU memory.
// Uses a staging buffer since storage buffers can't be mapped directly.
func (b *Backend) readBuffer(srcBuffer *wgpu.Buffer, size uint64) ([]byte, error) {
	// Create staging buffer for reading (MAP_READ | COPY_DST)
	stagingBuffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
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
	//nolint:gosec // unsafe.Slice for zero-copy conversion from unsafe.Pointer
	mappedSlice := unsafe.Slice((*byte)(mappedPtr), size)
	result := make([]byte, size)
	copy(result, mappedSlice)

	// Unmap buffer
	stagingBuffer.Unmap()

	return result, nil
}

// runBinaryOp executes a binary element-wise operation (add, sub, mul, div) on GPU.
func (b *Backend) runBinaryOp(a, other *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate inputs
	if a.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: only float32 is supported, got %s", a.DType())
	}
	if !a.Shape().Equal(other.Shape()) {
		return nil, fmt.Errorf("webgpu: shape mismatch: %v vs %v", a.Shape(), other.Shape())
	}

	numElements := a.NumElements()

	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferOther := b.createBuffer(other.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferOther.Release()

	//nolint:gosec // G115: Safe conversion, ByteSize() returns non-negative int
	resultSize := uint64(a.ByteSize())
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned
	//nolint:gosec // G115: Safe conversion, NumElements() returns non-negative int
	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements))
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
	//nolint:gosec // G115: Safe conversion, workgroup count is non-negative
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
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
	bufferInput := b.createBuffer(input.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferInput.Release()

	//nolint:gosec // G115: Safe conversion, ByteSize() returns non-negative int
	resultSize := uint64(input.ByteSize())
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})
	defer bufferResult.Release()

	// Create uniform buffer for params (size: u32)
	params := make([]byte, 16) // 16-byte aligned
	//nolint:gosec // G115: Safe conversion, NumElements() returns non-negative int
	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements))
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
	//nolint:gosec // G115: Safe conversion, workgroup count is non-negative
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
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

	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	M := uint32(a.Shape()[0])
	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	K := uint32(a.Shape()[1])
	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	N := uint32(other.Shape()[1])

	if other.Shape()[0] != int(K) {
		return nil, fmt.Errorf("webgpu: matmul shape mismatch: [%d,%d] @ [%d,%d]", M, K, other.Shape()[0], N)
	}

	// Compile shader
	shader := b.compileShader("matmul", matmulShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("matmul", shader)

	// Create GPU buffers
	bufferA := b.createBuffer(a.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferA.Release()

	bufferOther := b.createBuffer(other.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferOther.Release()

	resultShape := tensor.Shape{int(M), int(N)}
	//nolint:gosec // G115: Safe conversion, matrix dimensions are non-negative
	resultSize := uint64(int(M) * int(N) * 4) // float32 = 4 bytes
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
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
	//nolint:gosec // G115: Safe conversions, ByteSize() returns non-negative int
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, uint64(a.ByteSize())),
		wgpu.BufferBindingEntry(1, bufferOther, 0, uint64(other.ByteSize())),
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

	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	rows := uint32(input.Shape()[0])
	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	cols := uint32(input.Shape()[1])

	// Compile shader
	shader := b.compileShader("transpose", transposeShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("transpose", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferInput.Release()

	//nolint:gosec // G115: Safe conversion, ByteSize() returns non-negative int
	resultSize := uint64(input.ByteSize())
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
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

	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	batchSize := uint32(input.Shape()[0])
	//nolint:gosec // G115: Safe conversions, shape dimensions are non-negative
	numClasses := uint32(input.Shape()[1])

	// Compile shader
	shader := b.compileShader("softmax", softmaxShader)

	// Get or create pipeline
	pipeline := b.getOrCreatePipeline("softmax", shader)

	// Create GPU buffers
	bufferInput := b.createBuffer(input.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferInput.Release()

	//nolint:gosec // G115: Safe conversion, ByteSize() returns non-negative int
	resultSize := uint64(input.ByteSize())
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
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
