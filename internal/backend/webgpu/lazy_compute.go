//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
package webgpu

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
)

// createLazyResult creates a lazy RawTensor that keeps data on GPU.
// The buffer is NOT released - ownership is transferred to the lazy tensor.
// Data will be transferred from GPU only when Data() is called.
//
// This is the key optimization for Phase 3 Integration:
// - No readBuffer() call during operation.
// - Data stays on GPU until explicitly needed.
// - Chained operations can be batched.
func (b *Backend) createLazyResult(buffer *wgpu.Buffer, bufferSize uint64, shape tensor.Shape, dtype tensor.DataType) (*tensor.RawTensor, error) {
	// Create lazy GPU data with backend reference for later readBuffer()
	gpuData := tensor.NewLazyGPUData(unsafe.Pointer(buffer), bufferSize, b) //nolint:gosec // G103: Required for GPU buffer tracking

	// Create lazy tensor - CPU buffer allocated but not filled
	result, err := tensor.NewLazyRaw(shape, dtype, tensor.WebGPU, gpuData)
	if err != nil {
		// If tensor creation fails, release the GPU buffer
		buffer.Release()
		return nil, err
	}

	return result, nil
}

// runBinaryOpLazy executes a binary element-wise operation and returns a LAZY tensor.
// The result stays on GPU until Data() is called.
func (b *Backend) runBinaryOpLazy(a, other *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	// Validate inputs - must have same dtype
	if a.DType() != other.DType() {
		return nil, errDTypeMismatch(a.DType(), other.DType())
	}

	// Only float32 and int32 are supported
	dtype := a.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, errUnsupportedDType(dtype)
	}

	// Handle broadcasting if shapes don't match
	if !a.Shape().Equal(other.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(a.Shape(), other.Shape())
		if !ok {
			return nil, errBroadcastFailed(a.Shape(), other.Shape())
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

	// Create GPU buffers for inputs (these can be released after submission)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferOther := b.createBufferFromTensor(other)
	defer bufferOther.Release()

	resultSize := uint64(a.ByteSize())
	// Create result buffer - NO defer Release! Ownership transfers to lazy tensor
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	params := b.createParamsBuffer(numElements)
	defer params.Release()

	// Get bind group layout and create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferOther, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(3, params, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	// Create LAZY result - NO readBuffer() call!
	// Data will be transferred from GPU only when Data() is called
	return b.createLazyResult(bufferResult, resultSize, a.Shape(), a.DType())
}

// copyGPUBuffer creates a GPU-to-GPU copy without CPU round-trip.
// This is critical for LazyMode performance - avoids GPU→CPU→GPU transfers.
func (b *Backend) copyGPUBuffer(srcBuffer *wgpu.Buffer, size uint64) *wgpu.Buffer {
	dstBuffer := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  size,
	})

	encoder := b.device.CreateCommandEncoder(nil)
	encoder.CopyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, size)
	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return dstBuffer
}

// createBufferFromTensor creates a GPU buffer from a RawTensor.
// If the tensor already has GPU data (lazy), performs GPU→GPU copy (no CPU round-trip!).
func (b *Backend) createBufferFromTensor(t *tensor.RawTensor) *wgpu.Buffer {
	// Check if tensor already has GPU data
	if gpuData := t.GPUData(); gpuData != nil && !gpuData.IsRealized() {
		// Tensor has unrealized GPU data - use GPU→GPU copy
		existingBuffer := (*wgpu.Buffer)(gpuData.BufferPtr())
		return b.copyGPUBuffer(existingBuffer, gpuData.Size())
	}

	// CPU tensor - upload data to GPU
	return b.createBuffer(t.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
}

// createParamsBuffer creates a uniform buffer with element count parameter.
func (b *Backend) createParamsBuffer(numElements int) *wgpu.Buffer {
	params := make([]byte, 16) // 16-byte aligned
	putUint32LE(params[0:4], uint32(numElements))
	return b.createUniformBuffer(params)
}

// errDTypeMismatch returns an error for dtype mismatch.
func errDTypeMismatch(a, other tensor.DataType) error {
	return &lazyError{msg: "dtype mismatch: " + a.String() + " vs " + other.String()}
}

func errUnsupportedDType(dtype tensor.DataType) error {
	return &lazyError{msg: "unsupported dtype: " + dtype.String() + " (only float32 and int32)"}
}

func errBroadcastFailed(_, _ tensor.Shape) error {
	return &lazyError{msg: "shapes not broadcastable"}
}

type lazyError struct {
	msg string
}

func (e *lazyError) Error() string {
	return "webgpu: " + e.msg
}

// putUint32LE writes a uint32 to a byte slice in little-endian order.
func putUint32LE(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

// =============================================================================
// Extended Lazy Operations (Phase 3.1)
// =============================================================================

// runMatMulLazy executes matrix multiplication C = A @ B on GPU with lazy result.
// A is [M, K], B is [K, N], C is [M, N].
func (b *Backend) runMatMulLazy(a, other *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate inputs
	if a.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "matmul: only float32 is supported, got " + a.DType().String()}
	}
	if len(a.Shape()) != 2 || len(other.Shape()) != 2 {
		return nil, &lazyError{msg: "matmul: requires 2D tensors"}
	}

	M := uint32(a.Shape()[0])
	K := uint32(a.Shape()[1])
	N := uint32(other.Shape()[1])

	if other.Shape()[0] != int(K) {
		return nil, &lazyError{msg: "matmul: shape mismatch"}
	}

	// Compile shader
	shader := b.compileShader("matmul", matmulShader)
	pipeline := b.getOrCreatePipeline("matmul", shader)

	// Create GPU buffers (support lazy chaining)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferOther := b.createBufferFromTensor(other)
	defer bufferOther.Release()

	resultShape := tensor.Shape{int(M), int(N)}
	resultSize := uint64(int(M) * int(N) * 4)

	// Create result buffer - NO defer Release! Ownership transfers to lazy tensor
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create params buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], M)
	putUint32LE(params[4:8], K)
	putUint32LE(params[8:12], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

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

	// 2D workgroups (16x16 per workgroup)
	workgroupsX := (N + 15) / 16
	workgroupsY := (M + 15) / 16
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	// Return LAZY result - NO readBuffer!
	return b.createLazyResult(bufferResult, resultSize, resultShape, tensor.Float32)
}

// runUnaryOpLazy executes a unary operation (exp, sqrt, cos, sin, etc.) with lazy result.
func (b *Backend) runUnaryOpLazy(x *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	if x.DType() != tensor.Float32 {
		return nil, &lazyError{msg: shaderName + ": only float32 is supported"}
	}

	numElements := x.NumElements()
	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer (support lazy chaining)
	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	resultSize := uint64(x.ByteSize())
	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create params buffer
	params := b.createParamsBuffer(numElements)
	defer params.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferX, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, params, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, x.Shape(), tensor.Float32)
}

// runScalarOpLazy executes a scalar operation (mul, add, sub, div by scalar) with lazy result.
func (b *Backend) runScalarOpLazy(x *tensor.RawTensor, scalar float32, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	if x.DType() != tensor.Float32 {
		return nil, &lazyError{msg: shaderName + ": only float32 is supported"}
	}

	numElements := x.NumElements()
	// Compile shader
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer
	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	resultSize := uint64(x.ByteSize())
	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create params buffer with scalar value
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(numElements))
	putFloat32LE(params[4:8], scalar)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferX, 0, resultSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, x.Shape(), tensor.Float32)
}

// putFloat32LE writes a float32 to a byte slice in little-endian order.
func putFloat32LE(b []byte, v float32) {
	bits := *(*uint32)(unsafe.Pointer(&v)) //nolint:gosec // G103: Required for float bit conversion
	putUint32LE(b, bits)
}

// runBatchMatMulLazy executes batched matrix multiplication on GPU with lazy result.
// Supports 3D [batch, M, K] @ [batch, K, N] and 4D [batch, heads, M, K] @ [batch, heads, K, N].
func (b *Backend) runBatchMatMulLazy(a, other *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate inputs
	if a.DType() != tensor.Float32 || other.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "batchMatMul: only float32 is supported"}
	}

	shapeA := a.Shape()
	shapeB := other.Shape()

	if len(shapeA) != len(shapeB) || (len(shapeA) != 3 && len(shapeA) != 4) {
		return nil, &lazyError{msg: "batchMatMul: requires 3D or 4D tensors with matching dimensions"}
	}

	var batch, M, K, N uint32
	var resultShape tensor.Shape

	if len(shapeA) == 3 {
		// 3D: [batch, M, K] @ [batch, K, N]
		batch = uint32(shapeA[0])
		M = uint32(shapeA[1])
		K = uint32(shapeA[2])
		N = uint32(shapeB[2])
		resultShape = tensor.Shape{int(batch), int(M), int(N)}
	} else {
		// 4D: [batch, heads, M, K] @ [batch, heads, K, N]
		batch = uint32(shapeA[0] * shapeA[1])
		M = uint32(shapeA[2])
		K = uint32(shapeA[3])
		N = uint32(shapeB[3])
		resultShape = tensor.Shape{shapeA[0], shapeA[1], int(M), int(N)}
	}

	// Compile shader
	shader := b.compileShader("batchMatMul", batchMatMulShader)
	pipeline := b.getOrCreatePipeline("batchMatMul", shader)

	// Create GPU buffers (support lazy chaining)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferB := b.createBufferFromTensor(other)
	defer bufferB.Release()

	resultSize := uint64(batch) * uint64(M) * uint64(N) * 4 // float32 = 4 bytes

	// Create result buffer - NO defer Release! Ownership transfers to lazy tensor
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	params := make([]byte, 16)
	putUint32LE(params[0:4], batch)
	putUint32LE(params[4:8], M)
	putUint32LE(params[8:12], K)
	putUint32LE(params[12:16], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferA, 0, uint64(a.ByteSize())),
		wgpu.BufferBindingEntry(1, bufferB, 0, uint64(other.ByteSize())),
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
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	// Return LAZY result - NO readBuffer!
	return b.createLazyResult(bufferResult, resultSize, resultShape, tensor.Float32)
}

// runTransposeLazy executes 2D matrix transpose with lazy result.
func (b *Backend) runTransposeLazy(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "transpose: only float32 is supported"}
	}
	if len(input.Shape()) != 2 {
		return nil, &lazyError{msg: "transpose: requires 2D tensor"}
	}

	rows := uint32(input.Shape()[0])
	cols := uint32(input.Shape()[1])

	// Compile shader
	shader := b.compileShader("transpose", transposeShader)
	pipeline := b.getOrCreatePipeline("transpose", shader)

	// Create input buffer
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultShape := tensor.Shape{int(cols), int(rows)}
	resultSize := uint64(input.ByteSize())
	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create params buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], rows)
	putUint32LE(params[4:8], cols)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
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

	workgroupsX := (cols + 15) / 16
	workgroupsY := (rows + 15) / 16
	computePass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, resultShape, tensor.Float32)
}

// runSoftmaxLazy executes softmax on GPU with lazy result.
// Input must be 2D [batch, classes].
func (b *Backend) runSoftmaxLazy(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	// Validate input
	if input.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "softmax: only float32 is supported"}
	}
	if len(input.Shape()) != 2 {
		return nil, &lazyError{msg: "softmax: requires 2D tensor"}
	}

	batchSize := uint32(input.Shape()[0])
	numClasses := uint32(input.Shape()[1])

	// Compile shader
	shader := b.compileShader("softmax", softmaxShader)
	pipeline := b.getOrCreatePipeline("softmax", shader)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize())
	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	params := make([]byte, 16)
	putUint32LE(params[0:4], batchSize)
	putUint32LE(params[4:8], numClasses)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
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
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, input.Shape(), tensor.Float32)
}

// runTransposeNDLazy executes N-dimensional transpose on GPU with lazy result.
// Supports up to 6D tensors with arbitrary axes permutation.
//
//nolint:gocognit,gocyclo,cyclop,funlen // Complex GPU setup logic - unavoidable for parameter packing
func (b *Backend) runTransposeNDLazy(input *tensor.RawTensor, axes []int) (*tensor.RawTensor, error) {
	shape := input.Shape()
	ndim := len(shape)

	if ndim > 6 {
		return nil, &lazyError{msg: "transposeND: supports up to 6D tensors"}
	}

	// Default axes: reverse all dimensions
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := 0; i < ndim; i++ {
			axes[i] = ndim - 1 - i
		}
	}

	if len(axes) != ndim {
		return nil, &lazyError{msg: "transposeND: axes length must match tensor dimensions"}
	}

	// Validate axes
	seen := make(map[int]bool)
	for _, ax := range axes {
		if ax < 0 || ax >= ndim {
			return nil, &lazyError{msg: "transposeND: axis out of range"}
		}
		if seen[ax] {
			return nil, &lazyError{msg: "transposeND: duplicate axis"}
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
		return nil, &lazyError{msg: "transposeND: unsupported dtype " + input.DType().String()}
	}

	// Compile shader and get pipeline
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize())

	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	// Layout: ndim, total_elements, shapes[6], input_strides[6], output_strides[6], axes[6]
	params := make([]byte, 4*26) // 26 u32 values * 4 bytes
	inputStrides := shape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	putUint32LE(params[0:4], uint32(ndim))
	putUint32LE(params[4:8], uint32(shape.NumElements()))

	// Pack input shape (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(shape) {
			putUint32LE(params[8+i*4:12+i*4], uint32(shape[i]))
		} else {
			putUint32LE(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			putUint32LE(params[32+i*4:36+i*4], uint32(inputStrides[i]))
		} else {
			putUint32LE(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			putUint32LE(params[56+i*4:60+i*4], uint32(outputStrides[i]))
		} else {
			putUint32LE(params[56+i*4:60+i*4], 1)
		}
	}

	// Pack axes (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(axes) {
			putUint32LE(params[80+i*4:84+i*4], uint32(axes[i]))
		} else {
			putUint32LE(params[80+i*4:84+i*4], 0)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
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
	numElements := uint32(shape.NumElements())
	workgroups := (numElements + 255) / 256
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, newShape, input.DType())
}

// runExpandLazy broadcasts tensor to new shape with lazy result.
// Supports up to 6D tensors.
//
//nolint:gocognit,gocyclo,cyclop,funlen // Complex GPU setup logic - unavoidable for parameter packing
func (b *Backend) runExpandLazy(input *tensor.RawTensor, newShape tensor.Shape) (*tensor.RawTensor, error) {
	shape := input.Shape()

	// Validate shapes are compatible for broadcasting
	if len(newShape) < len(shape) {
		return nil, &lazyError{msg: "expand: new shape must have at least as many dimensions"}
	}

	if len(newShape) > 6 {
		return nil, &lazyError{msg: "expand: supports up to 6D tensors"}
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
			return nil, &lazyError{msg: "expand: incompatible shapes"}
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
		return nil, &lazyError{msg: "expand: unsupported dtype " + input.DType().String()}
	}

	// Compile shader and get pipeline
	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	// Calculate result size
	resultNumElements := newShape.NumElements()
	elementSize := uint64(input.DType().Size())
	resultSize := uint64(resultNumElements) * elementSize

	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer for params
	params := make([]byte, 4*20) // 20 u32 values * 4 bytes
	inputStrides := paddedShape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	putUint32LE(params[0:4], uint32(len(newShape)))
	putUint32LE(params[4:8], uint32(resultNumElements))

	// Pack input shape (6 slots) - use paddedShape
	for i := 0; i < 6; i++ {
		if i < len(paddedShape) {
			putUint32LE(params[8+i*4:12+i*4], uint32(paddedShape[i]))
		} else {
			putUint32LE(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			putUint32LE(params[32+i*4:36+i*4], uint32(inputStrides[i]))
		} else {
			putUint32LE(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			putUint32LE(params[56+i*4:60+i*4], uint32(outputStrides[i]))
		} else {
			putUint32LE(params[56+i*4:60+i*4], 1)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	inputSize := uint64(input.ByteSize())
	paramsSize := uint64(len(params))
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, inputSize),
		wgpu.BufferBindingEntry(1, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferParams, 0, paramsSize),
	})
	defer bindGroup.Release()

	// Execute compute pass
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((resultNumElements + 255) / 256)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, newShape, input.DType())
}

// runGatherLazy executes Gather operation with lazy result.
// Input must be float32, indices must be int32.
func (b *Backend) runGatherLazy(input *tensor.RawTensor, dim int, indices *tensor.RawTensor) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "gather: input must be float32"}
	}
	if indices.DType() != tensor.Int32 {
		return nil, &lazyError{msg: "gather: indices must be int32"}
	}

	inShape := input.Shape()
	idxShape := indices.Shape()
	ndim := len(inShape)

	// Normalize dimension
	if dim < 0 {
		dim = ndim + dim
	}

	// For non-last dimensions: use non-lazy path (involves multiple operations)
	if dim != ndim-1 {
		// Fall back to non-lazy for complex transpose chain
		return b.gatherNonLastDim(input, dim, indices)
	}

	// Calculate batch size
	gatherBatchSize := 1
	for i := 0; i < ndim-1; i++ {
		gatherBatchSize *= inShape[i]
	}
	inputDim := inShape[ndim-1]
	outputK := idxShape[len(idxShape)-1]

	// Result shape
	gatherResultShape := make(tensor.Shape, ndim)
	copy(gatherResultShape, inShape[:ndim-1])
	gatherResultShape[ndim-1] = outputK

	shader := b.compileShader("gather", gatherShader)
	pipeline := b.getOrCreatePipeline("gather", shader)

	// Create buffers (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	bufferIndices := b.createBufferFromTensor(indices)
	defer bufferIndices.Release()

	gatherResultSize := uint64(gatherBatchSize) * uint64(outputK) * 4
	// Create result buffer - NO defer Release!
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  gatherResultSize,
	})

	// Create uniform buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(gatherBatchSize))
	putUint32LE(params[4:8], uint32(inputDim))
	putUint32LE(params[8:12], uint32(outputK))
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())),
		wgpu.BufferBindingEntry(1, bufferIndices, 0, uint64(indices.ByteSize())),
		wgpu.BufferBindingEntry(2, bufferResult, 0, gatherResultSize),
		wgpu.BufferBindingEntry(3, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	totalOutput := gatherBatchSize * outputK
	workgroups := uint32((totalOutput + workgroupSize - 1) / workgroupSize)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, gatherResultSize, gatherResultShape, tensor.Float32)
}

// runWhereLazy executes conditional selection on GPU and returns a LAZY tensor.
// result[i] = condition[i] != 0 ? x[i] : y[i].
// The result stays on GPU until Data() is called.
//
//nolint:gocyclo,cyclop,funlen // Conditional selection with broadcasting has inherent complexity
func (b *Backend) runWhereLazy(condition, x, y *tensor.RawTensor) (*tensor.RawTensor, error) {
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
		condFloat32, err = int32ToFloat32(condition)
		if err != nil {
			return nil, err
		}
	default:
		return nil, errUnsupportedDType(condition.DType())
	}

	// x and y must have same dtype
	if x.DType() != y.DType() {
		return nil, errDTypeMismatch(x.DType(), y.DType())
	}

	// Only float32 and int32 supported
	dtype := x.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, errUnsupportedDType(dtype)
	}

	// Handle broadcasting - compute output shape from all 3 tensors
	outShape := condFloat32.Shape()

	// Broadcast condition with x
	if !condFloat32.Shape().Equal(x.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(condFloat32.Shape(), x.Shape())
		if !ok {
			return nil, errBroadcastFailed(condFloat32.Shape(), x.Shape())
		}
		outShape = broadcastedShape
	}

	// Broadcast outShape with y
	if !outShape.Equal(y.Shape()) {
		broadcastedShape, ok, _ := tensor.BroadcastShapes(outShape, y.Shape())
		if !ok {
			return nil, errBroadcastFailed(outShape, y.Shape())
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

	// Create buffers (from lazy tensors if needed)
	bufferCondition := b.createBufferFromTensor(condFloat32)
	defer bufferCondition.Release()

	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	bufferY := b.createBufferFromTensor(y)
	defer bufferY.Release()

	resultSize := uint64(x.ByteSize())
	// Create result buffer - NO defer Release! Ownership transfers to lazy tensor
	bufferResult := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  resultSize,
	})

	// Create uniform buffer
	params := make([]byte, 16)

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements))
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	condSize := uint64(condFloat32.ByteSize())
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferCondition, 0, condSize),
		wgpu.BufferBindingEntry(1, bufferX, 0, resultSize),
		wgpu.BufferBindingEntry(2, bufferY, 0, resultSize),
		wgpu.BufferBindingEntry(3, bufferResult, 0, resultSize),
		wgpu.BufferBindingEntry(4, bufferParams, 0, 16),
	})
	defer bindGroup.Release()

	// Execute
	encoder := b.device.CreateCommandEncoder(nil)
	computePass := encoder.BeginComputePass(nil)

	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	return b.createLazyResult(bufferResult, resultSize, outShape, dtype)
}

// runSumLazy executes sum reduction and returns a LAZY tensor.
// For Sum, the result is scalar (4 bytes), so lazy mode has minimal benefit.
// However, this avoids blocking the GPU pipeline during chained operations.
//
//nolint:funlen // Parallel reduction requires multiple stages
func (b *Backend) runSumLazy(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	dtype := input.DType()
	if dtype != tensor.Float32 && dtype != tensor.Int32 {
		return nil, errUnsupportedDType(dtype)
	}

	numElements := input.NumElements()

	// For small tensors, use CPU (no benefit from lazy mode)
	if numElements < 1024 {
		return b.runSumCPU(input)
	}

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
		return nil, errUnsupportedDType(dtype)
	}

	shader := b.compileShader(shaderName, shaderCode)
	pipeline := b.getOrCreatePipeline(shaderName, shader)

	// Create input buffer (from lazy tensor if needed)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	// Calculate number of workgroups needed
	numWorkgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
	partialSumsSize := uint64(numWorkgroups) * 4

	bufferPartialSums := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		Size:  partialSumsSize,
	})
	defer bufferPartialSums.Release()

	// Create uniform buffer for params
	params := make([]byte, 16)

	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements))
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)

	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferInput, 0, uint64(input.ByteSize())),
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
	b.queueCommand(cmdBuffer) // Batch instead of immediate submit

	// For Sum, we need to read partial sums and aggregate on CPU.
	// This is unavoidable for parallel reduction.
	// Read partial sums.
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
			sum += int32(binary.LittleEndian.Uint32(partialData[i*4 : i*4+4]))
		}
		result, err := tensor.NewRaw(tensor.Shape{}, tensor.Int32, tensor.WebGPU)
		if err != nil {
			return nil, err
		}
		result.AsInt32()[0] = sum
		return result, nil

	default:
		return nil, errUnsupportedDType(dtype)
	}
}
