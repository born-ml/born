//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
)

// createLazyResult creates a lazy RawTensor backed by a GPU staging buffer.
// The stagingBuf must have MapRead | CopyDst usage and must already have been
// populated via CopyBufferToBuffer inside the same encoder as the compute pass.
//
// Ownership of stagingBuf is transferred to the lazy tensor:
// - It is NOT released here — the caller must NOT defer-release it.
// - It will be released when LazyGPUData.Release() is called (GC or explicit).
//
// When Data() is called on the result tensor, ReadGPUBuffer() Maps the staging
// buffer directly — no additional copy encoder is needed.
func (b *Backend) createLazyResult(stagingBuf *wgpu.Buffer, bufferSize uint64, shape tensor.Shape, dtype tensor.DataType) (*tensor.RawTensor, error) {
	// Create lazy GPU data referencing the staging (MapRead) buffer.
	gpuData := tensor.NewLazyGPUData(unsafe.Pointer(stagingBuf), bufferSize, b) //nolint:gosec // G103: Required for GPU buffer tracking

	// Create lazy tensor — CPU buffer allocated but not filled until Data() is called.
	result, err := tensor.NewLazyRaw(shape, dtype, tensor.WebGPU, gpuData)
	if err != nil {
		// If tensor creation fails, release the staging buffer.
		stagingBuf.Release()
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

	shader := b.compileShader(shaderName, shaderCode)
	entry := b.getOrCreatePipeline(shaderName, shader, bglBinary)

	// Create GPU buffers for inputs (these can be released after submission)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferOther := b.createBufferFromTensor(other)
	defer bufferOther.Release()

	resultSize := uint64(a.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate result buffer: written by the compute shader, source for the copy.
	// Released at end of function — the encoder holds its own reference.
	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runBinaryOpLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	// Staging buffer: CopyBufferToBuffer destination + MapRead for CPU readback.
	// CopySrc is needed so this buffer can be re-used as source when chaining
	// lazy ops (createBufferFromTensor → copyGPUBuffer).
	// Ownership transfers to the lazy tensor — NO defer Release.
	stagingBuf, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runBinaryOpLazy: create staging buffer: %w", err)
	}

	// Create uniform buffer for params
	params := b.createParamsBuffer(numElements)
	defer params.Release()

	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferA, resultSize),
		bufBinding(bufferOther, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(params, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runBinaryOpLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	// finishAndSubmitLazy appends CopyBufferToBuffer, finishes the encoder, and submits
	// immediately. Immediate submit is required so lastSubmissionIndex is updated BEFORE
	// the deferred bufferResult.Release() fires (buffers deferred at index 0 get
	// destroyed on the next DestroyQueue.Triage call, corrupting in-flight GPU copies).
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, a.Shape(), a.DType(), "runBinaryOpLazy")
}

// createStagingBuffer creates a MapRead | CopyDst buffer of the given size.
// Returns the staging buffer; the caller is responsible for releasing it
// (or transferring ownership to a lazy tensor via createLazyResult).
func (b *Backend) createStagingBuffer(size uint64) (*wgpu.Buffer, error) {
	buf, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst | gputypes.BufferUsageCopySrc,
		Size:  size,
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: failed to create staging buffer: %w", err)
	}
	return buf, nil
}

// finishAndSubmitLazy finalizes encoder (appending a copy from resultBuf to stagingBuf),
// submits the command buffer IMMEDIATELY, and returns a lazy tensor.
// stagingBuf ownership is transferred to the returned tensor (no caller release).
// resultBuf is NOT transferred — caller must defer-release it.
//
// IMPORTANT: This function submits immediately (not batched) to ensure the submission
// index is updated BEFORE the caller's deferred resultBuf.Release() fires.
// If we batched (queueCommand), the defer would fire with lastSubmissionIndex=0,
// causing the DestroyQueue to destroy resultBuf before the GPU copy reads it.
// See: BUG-LAZY-DEFER-RELEASE (buffers deferred at index 0 destroyed on next triage).
func (b *Backend) finishAndSubmitLazy(
	encoder *wgpu.CommandEncoder,
	resultBuf *wgpu.Buffer,
	stagingBuf *wgpu.Buffer,
	resultSize uint64,
	shape tensor.Shape,
	dtype tensor.DataType,
	opName string,
) (*tensor.RawTensor, error) {
	// CopyBufferToBuffer in the SAME encoder, after the compute pass ended,
	// before Finish(). This is the unified encoder pattern.
	encoder.CopyBufferToBuffer(resultBuf, 0, stagingBuf, 0, resultSize)
	cmdBuffer, err := encoder.Finish()
	if err != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("%s: finish encoder: %w", opName, err)
	}
	// Submit immediately so lastSubmissionIndex is updated before the caller's
	// defer resultBuf.Release() fires. This prevents premature buffer destruction.
	if _, err := b.queue.Submit(cmdBuffer); err != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("%s: submit: %w", opName, err)
	}

	return b.createLazyResult(stagingBuf, resultSize, shape, dtype)
}

// copyGPUBuffer creates a GPU-to-GPU copy without CPU round-trip.
// This is critical for LazyMode performance - avoids GPU→CPU→GPU transfers.
func (b *Backend) copyGPUBuffer(srcBuffer *wgpu.Buffer, size uint64) *wgpu.Buffer {
	dstBuffer, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  size,
	})
	if err != nil {
		panic(fmt.Sprintf("webgpu: copyGPUBuffer: failed to create dst buffer: %v", err))
	}

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		panic(fmt.Sprintf("webgpu: copyGPUBuffer: failed to create encoder: %v", encErr))
	}
	encoder.CopyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, size)
	cmdBuffer, finErr := encoder.Finish()
	if finErr != nil {
		panic(fmt.Sprintf("webgpu: copyGPUBuffer: failed to finish encoder: %v", finErr))
	}
	if _, err := b.queue.Submit(cmdBuffer); err != nil {
		panic(fmt.Sprintf("webgpu: copyGPUBuffer: submit failed: %v", err))
	}

	return dstBuffer
}

// createBufferFromTensor creates a GPU buffer from a RawTensor.
// If the tensor already has GPU data (lazy), performs GPU→GPU copy (no CPU round-trip!).
func (b *Backend) createBufferFromTensor(t *tensor.RawTensor) *wgpu.Buffer {
	// Check if tensor already has GPU data
	if gpuData := t.GPUData(); gpuData != nil && !gpuData.IsRealized() {
		// Tensor has unrealized GPU data - use GPU→GPU copy.
		// KeepAlive prevents GC from collecting gpuData (and running its
		// finalizer which releases the buffer) while copyGPUBuffer uses it.
		existingBuffer := (*wgpu.Buffer)(gpuData.BufferPtr())
		result := b.copyGPUBuffer(existingBuffer, gpuData.Size())
		runtime.KeepAlive(gpuData)
		return result
	}

	// CPU tensor - upload data to GPU
	return b.createBuffer(t.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
}

// createParamsBuffer creates a uniform buffer with element count parameter.
func (b *Backend) createParamsBuffer(numElements int) *wgpu.Buffer {
	params := make([]byte, 16)                    // 16-byte aligned
	putUint32LE(params[0:4], uint32(numElements)) //nolint:gosec // G115: integer overflow conversion int -> uint32
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
	b[0] = byte(v)       //nolint:gosec // G115: intentional uint32-to-byte truncation for LE encoding
	b[1] = byte(v >> 8)  //nolint:gosec // G115: intentional uint32-to-byte truncation for LE encoding
	b[2] = byte(v >> 16) //nolint:gosec // G115: intentional uint32-to-byte truncation for LE encoding
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

	M := uint32(a.Shape()[0])     //nolint:gosec // G115: safe, tensor dims are small positive ints
	K := uint32(a.Shape()[1])     //nolint:gosec // G115: safe, tensor dims are small positive ints
	N := uint32(other.Shape()[1]) //nolint:gosec // G115: safe, tensor dims are small positive ints

	if other.Shape()[0] != int(K) {
		return nil, &lazyError{msg: "matmul: shape mismatch"}
	}

	shader := b.compileShader("matmul", matmulShader)
	entry := b.getOrCreatePipeline("matmul", shader, bglBinary)

	// Create GPU buffers (support lazy chaining)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferOther := b.createBufferFromTensor(other)
	defer bufferOther.Release()

	resultShape := tensor.Shape{int(M), int(N)}
	resultSize := uint64(int(M) * int(N) * 4) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Storage buffer for compute output; released at end of function.
	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runMatMulLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runMatMulLazy: %w", err)
	}

	// Create params buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], M)
	putUint32LE(params[4:8], K)
	putUint32LE(params[8:12], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	sizeA := uint64(a.ByteSize())         //nolint:gosec // G115: integer overflow conversion int -> uint64
	sizeOther := uint64(other.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferA, sizeA),
		bufBinding(bufferOther, sizeOther),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runMatMulLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	// 2D workgroups (16x16 per workgroup)
	workgroupsX := (N + 15) / 16
	workgroupsY := (M + 15) / 16
	computePass.Dispatch(workgroupsX, workgroupsY, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, resultShape, tensor.Float32, "runMatMulLazy")
}

// runUnaryOpLazy executes a unary operation (exp, sqrt, cos, sin, etc.) with lazy result.
func (b *Backend) runUnaryOpLazy(x *tensor.RawTensor, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	if x.DType() != tensor.Float32 {
		return nil, &lazyError{msg: shaderName + ": only float32 is supported"}
	}

	numElements := x.NumElements()
	shader := b.compileShader(shaderName, shaderCode)
	entry := b.getOrCreatePipeline(shaderName, shader, bglUnary)

	// Create input buffer (support lazy chaining)
	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	resultSize := uint64(x.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runUnaryOpLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runUnaryOpLazy: %w", err)
	}

	// Create params buffer
	params := b.createParamsBuffer(numElements)
	defer params.Release()

	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferX, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(params, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runUnaryOpLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, x.Shape(), tensor.Float32, "runUnaryOpLazy")
}

// runScalarOpLazy executes a scalar operation (mul, add, sub, div by scalar) with lazy result.
func (b *Backend) runScalarOpLazy(x *tensor.RawTensor, scalar float32, shaderName, shaderCode string) (*tensor.RawTensor, error) {
	if x.DType() != tensor.Float32 {
		return nil, &lazyError{msg: shaderName + ": only float32 is supported"}
	}

	numElements := x.NumElements()
	shader := b.compileShader(shaderName, shaderCode)
	entry := b.getOrCreatePipeline(shaderName, shader, bglUnary)

	// Create input buffer
	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	resultSize := uint64(x.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runScalarOpLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runScalarOpLazy: %w", err)
	}

	// Create params buffer with scalar value
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(numElements)) //nolint:gosec // G115: integer overflow conversion int -> uint32
	putFloat32LE(params[4:8], scalar)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferX, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runScalarOpLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, x.Shape(), tensor.Float32, "runScalarOpLazy")
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
		batch = uint32(shapeA[0]) //nolint:gosec // G115: safe, tensor dims are small positive ints
		M = uint32(shapeA[1])     //nolint:gosec // G115: safe, tensor dims are small positive ints
		K = uint32(shapeA[2])     //nolint:gosec // G115: safe, tensor dims are small positive ints
		N = uint32(shapeB[2])     //nolint:gosec // G115: safe, tensor dims are small positive ints
		resultShape = tensor.Shape{int(batch), int(M), int(N)}
	} else {
		// 4D: [batch, heads, M, K] @ [batch, heads, K, N]
		batch = uint32(shapeA[0] * shapeA[1]) //nolint:gosec // G115: safe, product of small tensor dims
		M = uint32(shapeA[2])                 //nolint:gosec // G115: safe, tensor dims are small positive ints
		K = uint32(shapeA[3])                 //nolint:gosec // G115: safe, tensor dims are small positive ints
		N = uint32(shapeB[3])                 //nolint:gosec // G115: safe, tensor dims are small positive ints
		resultShape = tensor.Shape{shapeA[0], shapeA[1], int(M), int(N)}
	}

	shader := b.compileShader("batchMatMul", batchMatMulShader)
	entry := b.getOrCreatePipeline("batchMatMul", shader, bglBinary)

	// Create GPU buffers (support lazy chaining)
	bufferA := b.createBufferFromTensor(a)
	defer bufferA.Release()

	bufferB := b.createBufferFromTensor(other)
	defer bufferB.Release()

	resultSize := uint64(batch) * uint64(M) * uint64(N) * 4 // float32 = 4 bytes

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runBatchMatMulLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runBatchMatMulLazy: %w", err)
	}

	// Create uniform buffer for params
	params := make([]byte, 16)
	putUint32LE(params[0:4], batch)
	putUint32LE(params[4:8], M)
	putUint32LE(params[8:12], K)
	putUint32LE(params[12:16], N)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	sizeA := uint64(a.ByteSize())     //nolint:gosec // G115: integer overflow conversion int -> uint64
	sizeB := uint64(other.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferA, sizeA),
		bufBinding(bufferB, sizeB),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runBatchMatMulLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	// Dispatch: (N+7)/8 x (M+7)/8 x batch
	workgroupsX := (N + 7) / 8
	workgroupsY := (M + 7) / 8
	computePass.Dispatch(workgroupsX, workgroupsY, batch)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, resultShape, tensor.Float32, "runBatchMatMulLazy")
}

// runTransposeLazy executes 2D matrix transpose with lazy result.
func (b *Backend) runTransposeLazy(input *tensor.RawTensor) (*tensor.RawTensor, error) {
	if input.DType() != tensor.Float32 {
		return nil, &lazyError{msg: "transpose: only float32 is supported"}
	}
	if len(input.Shape()) != 2 {
		return nil, &lazyError{msg: "transpose: requires 2D tensor"}
	}

	rows := uint32(input.Shape()[0]) //nolint:gosec // G115: safe, tensor dims are small positive ints
	cols := uint32(input.Shape()[1]) //nolint:gosec // G115: safe, tensor dims are small positive ints

	shader := b.compileShader("transpose", transposeShader)
	entry := b.getOrCreatePipeline("transpose", shader, bglUnary)

	// Create input buffer
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultShape := tensor.Shape{int(cols), int(rows)}
	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runTransposeLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runTransposeLazy: %w", err)
	}

	// Create params buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], rows)
	putUint32LE(params[4:8], cols)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runTransposeLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroupsX := (cols + 15) / 16
	workgroupsY := (rows + 15) / 16
	computePass.Dispatch(workgroupsX, workgroupsY, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, resultShape, tensor.Float32, "runTransposeLazy")
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

	batchSize := uint32(input.Shape()[0])  //nolint:gosec // G115: safe, tensor dims are small positive ints
	numClasses := uint32(input.Shape()[1]) //nolint:gosec // G115: safe, tensor dims are small positive ints

	shader := b.compileShader("softmax", softmaxShader)
	entry := b.getOrCreatePipeline("softmax", shader, bglUnary)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if err != nil {
		return nil, fmt.Errorf("runSoftmaxLazy: create result buffer: %w", err)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, err := b.createStagingBuffer(resultSize)
	if err != nil {
		return nil, fmt.Errorf("runSoftmaxLazy: %w", err)
	}

	// Create uniform buffer for params
	params := make([]byte, 16)
	putUint32LE(params[0:4], batchSize)
	putUint32LE(params[4:8], numClasses)
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runSoftmaxLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	// Each workgroup handles one row (batch sample)
	workgroups := (batchSize + workgroupSize - 1) / workgroupSize
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, input.Shape(), tensor.Float32, "runSoftmaxLazy")
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
	entry := b.getOrCreatePipeline(shaderName, shader, bglUnary)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	resultSize := uint64(input.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, bufErr := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if bufErr != nil {
		return nil, fmt.Errorf("runTransposeNDLazy: create result buffer: %w", bufErr)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, stagingErr := b.createStagingBuffer(resultSize)
	if stagingErr != nil {
		return nil, fmt.Errorf("runTransposeNDLazy: %w", stagingErr)
	}

	// Create uniform buffer for params
	// Layout: ndim, total_elements, shapes[6], input_strides[6], output_strides[6], axes[6]
	params := make([]byte, 4*26) // 26 u32 values * 4 bytes
	inputStrides := shape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	putUint32LE(params[0:4], uint32(ndim))
	putUint32LE(params[4:8], uint32(shape.NumElements())) //nolint:gosec // G115: integer overflow conversion int -> uint32

	// Pack input shape (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(shape) {
			putUint32LE(params[8+i*4:12+i*4], uint32(shape[i])) //nolint:gosec // G115: safe, tensor dims are small positive ints
		} else {
			putUint32LE(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			putUint32LE(params[32+i*4:36+i*4], uint32(inputStrides[i])) //nolint:gosec // G115: safe, strides derived from tensor dims
		} else {
			putUint32LE(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			putUint32LE(params[56+i*4:60+i*4], uint32(outputStrides[i])) //nolint:gosec // G115: safe, strides derived from tensor dims
		} else {
			putUint32LE(params[56+i*4:60+i*4], 1)
		}
	}

	// Pack axes (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(axes) {
			putUint32LE(params[80+i*4:84+i*4], uint32(axes[i])) //nolint:gosec // G115: safe, axis indices are small non-negative ints
		} else {
			putUint32LE(params[80+i*4:84+i*4], 0)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	paramsSize := uint64(len(params))
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, paramsSize),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runTransposeNDLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	// Calculate workgroup count (1D workgroups, 256 threads each)
	numElements := uint32(shape.NumElements()) //nolint:gosec // G115: integer overflow conversion int -> uint32
	workgroups := (numElements + 255) / 256
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, newShape, input.DType(), "runTransposeNDLazy")
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
	entry := b.getOrCreatePipeline(shaderName, shader, bglUnary)

	// Create input buffer (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	// Calculate result size
	resultNumElements := newShape.NumElements()
	elementSize := uint64(input.DType().Size())           //nolint:gosec // G115: integer overflow conversion int -> uint64
	resultSize := uint64(resultNumElements) * elementSize //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, bufErr := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if bufErr != nil {
		return nil, fmt.Errorf("runExpandLazy: create result buffer: %w", bufErr)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, stagingErr := b.createStagingBuffer(resultSize)
	if stagingErr != nil {
		return nil, fmt.Errorf("runExpandLazy: %w", stagingErr)
	}

	// Create uniform buffer for params
	params := make([]byte, 4*20) // 20 u32 values * 4 bytes
	inputStrides := paddedShape.ComputeStrides()
	outputStrides := newShape.ComputeStrides()

	putUint32LE(params[0:4], uint32(len(newShape)))     //nolint:gosec // G115: integer overflow conversion int -> uint32
	putUint32LE(params[4:8], uint32(resultNumElements)) //nolint:gosec // G115: integer overflow conversion int -> uint32

	// Pack input shape (6 slots) - use paddedShape
	for i := 0; i < 6; i++ {
		if i < len(paddedShape) {
			putUint32LE(params[8+i*4:12+i*4], uint32(paddedShape[i])) //nolint:gosec // G115: safe, tensor dims are small positive ints
		} else {
			putUint32LE(params[8+i*4:12+i*4], 1)
		}
	}

	// Pack input strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(inputStrides) {
			putUint32LE(params[32+i*4:36+i*4], uint32(inputStrides[i])) //nolint:gosec // G115: safe, strides derived from tensor dims
		} else {
			putUint32LE(params[32+i*4:36+i*4], 1)
		}
	}

	// Pack output strides (6 slots)
	for i := 0; i < 6; i++ {
		if i < len(outputStrides) {
			putUint32LE(params[56+i*4:60+i*4], uint32(outputStrides[i])) //nolint:gosec // G115: safe, strides derived from tensor dims
		} else {
			putUint32LE(params[56+i*4:60+i*4], 1)
		}
	}

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	inputSize := uint64(input.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	paramsSize := uint64(len(params))
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, inputSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, paramsSize),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runExpandLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroups := uint32((resultNumElements + 255) / 256) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, newShape, input.DType(), "runExpandLazy")
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
	entry := b.getOrCreatePipeline("gather", shader, bglBinary)

	// Create buffers (support lazy chaining)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	bufferIndices := b.createBufferFromTensor(indices)
	defer bufferIndices.Release()

	gatherResultSize := uint64(gatherBatchSize) * uint64(outputK) * 4 //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, bufErr := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  gatherResultSize,
	})
	if bufErr != nil {
		return nil, fmt.Errorf("runGatherLazy: create result buffer: %w", bufErr)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, stagingErr := b.createStagingBuffer(gatherResultSize)
	if stagingErr != nil {
		return nil, fmt.Errorf("runGatherLazy: %w", stagingErr)
	}

	// Create uniform buffer
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(gatherBatchSize))
	putUint32LE(params[4:8], uint32(inputDim)) //nolint:gosec // G115: integer overflow conversion int -> uint32
	putUint32LE(params[8:12], uint32(outputK)) //nolint:gosec // G115: integer overflow conversion int -> uint32
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	sizeInput := uint64(input.ByteSize())     //nolint:gosec // G115: integer overflow conversion int -> uint64
	sizeIndices := uint64(indices.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, sizeInput),
		bufBinding(bufferIndices, sizeIndices),
		bufBinding(bufferResult, gatherResultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runGatherLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	totalOutput := gatherBatchSize * outputK
	workgroups := uint32((totalOutput + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, gatherResultSize, gatherResultShape, tensor.Float32, "runGatherLazy")
}

// runWhereLazy executes conditional selection on GPU and returns a LAZY tensor.
// result[i] = condition[i] != 0 ? x[i] : y[i].
// The result stays on GPU until Data() is called.
//
//nolint:gocyclo,cyclop,funlen,gocognit // Conditional selection with broadcasting has inherent complexity
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
	entry := b.getOrCreatePipeline(shaderName, shader, bglWhere)

	// Create buffers (from lazy tensors if needed)
	bufferCondition := b.createBufferFromTensor(condFloat32)
	defer bufferCondition.Release()

	bufferX := b.createBufferFromTensor(x)
	defer bufferX.Release()

	bufferY := b.createBufferFromTensor(y)
	defer bufferY.Release()

	resultSize := uint64(x.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64

	// Intermediate Storage buffer: written by compute shader, source for CopyBufferToBuffer.
	bufferResult, bufErr := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
		Size:  resultSize,
	})
	if bufErr != nil {
		return nil, fmt.Errorf("runWhereLazy: create result buffer: %w", bufErr)
	}
	defer bufferResult.Release()

	// Staging buffer (MapRead | CopyDst): ownership transfers to lazy tensor.
	stagingBuf, stagingErr := b.createStagingBuffer(resultSize)
	if stagingErr != nil {
		return nil, fmt.Errorf("runWhereLazy: %w", stagingErr)
	}

	// Create uniform buffer
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: integer overflow conversion int -> uint32
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	condSize := uint64(condFloat32.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferCondition, condSize),
		bufBinding(bufferX, resultSize),
		bufBinding(bufferY, resultSize),
		bufBinding(bufferResult, resultSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	encoder, encErr := b.device.CreateCommandEncoder(nil)
	if encErr != nil {
		stagingBuf.Release()
		return nil, fmt.Errorf("runWhereLazy: create encoder: %w", encErr)
	}
	computePass, cpErr := encoder.BeginComputePass(nil)
	if cpErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: BeginComputePass error: %v", cpErr))
	}
	computePass.SetPipeline(entry.pipeline)
	computePass.SetBindGroup(0, bg, nil)
	workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	computePass.Dispatch(workgroups, 1, 1)
	if endErr := computePass.End(); endErr != nil {
		stagingBuf.Release()
		panic(fmt.Sprintf("webgpu: compute pass end error: %v", endErr))
	}
	return b.finishAndSubmitLazy(encoder, bufferResult, stagingBuf, resultSize, outShape, dtype, "runWhereLazy")
}

// runSumLazy executes sum reduction and returns a LAZY tensor.
// For Sum, the result is scalar (4 bytes), so lazy mode has minimal benefit.
// However, this avoids blocking the GPU pipeline during chained operations.
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
	entry := b.getOrCreatePipeline(shaderName, shader, bglUnary)

	// Create input buffer (from lazy tensor if needed)
	bufferInput := b.createBufferFromTensor(input)
	defer bufferInput.Release()

	// Calculate number of workgroups needed
	numWorkgroups := uint32((numElements + workgroupSize - 1) / workgroupSize) //nolint:gosec // G115: integer overflow conversion int -> uint32
	partialSumsSize := uint64(numWorkgroups) * 4

	bufferPartialSums, bufErr := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  partialSumsSize,
	})
	if bufErr != nil {
		return nil, fmt.Errorf("runSumLazy: create partial sums buffer: %w", bufErr)
	}
	defer bufferPartialSums.Release()

	// Create uniform buffer for params
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:4], uint32(numElements)) //nolint:gosec // G115: integer overflow conversion int -> uint32
	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	inputSize := uint64(input.ByteSize()) //nolint:gosec // G115: integer overflow conversion int -> uint64
	bg := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(bufferInput, inputSize),
		bufBinding(bufferPartialSums, partialSumsSize),
		bufBinding(bufferParams, 16),
	})
	defer bg.Release()

	// Sum needs immediate readback to aggregate partial results on CPU.
	// Use unified encoder: compute + copy to staging in one submission.
	partialData := b.execComputeAndRead(entry.pipeline, bg, numWorkgroups, 1, 1, bufferPartialSums, partialSumsSize)

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
			sum += int32(binary.LittleEndian.Uint32(partialData[i*4 : i*4+4])) //nolint:gosec // G115: integer overflow conversion uint32 -> int32
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
