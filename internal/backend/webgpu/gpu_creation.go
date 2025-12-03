//go:build windows

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
)

// FromRawTensor uploads a CPU tensor to GPU memory.
// This creates a new GPUTensor with data copied from the RawTensor.
func (b *Backend) FromRawTensor(t *tensor.RawTensor) *GPUTensor {
	if t == nil {
		panic("webgpu: FromRawTensor: input tensor is nil")
	}

	// Ensure buffer size is at least 4 bytes and aligned to COPY_BUFFER_ALIGNMENT (4 bytes)
	//nolint:gosec // G115: ByteSize() returns non-negative value, safe to convert to uint64
	byteSize := uint64(t.ByteSize())
	if byteSize < 4 {
		byteSize = 4
	}
	// Align to 4-byte boundary
	alignedSize := (byteSize + 3) &^ 3

	// Prepare aligned data
	alignedData := make([]byte, alignedSize)
	copy(alignedData, t.Data())

	// Create GPU buffer with aligned data
	buffer := b.createBuffer(
		alignedData,
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst,
	)

	// Track buffer allocation
	b.trackBufferAllocation(alignedSize)

	return &GPUTensor{
		buffer:     buffer,
		shape:      t.Shape(),
		dtype:      t.DType(),
		strides:    t.Strides(),
		backend:    b,
		computed:   true, // Data is already computed on CPU
		bufferSize: alignedSize,
	}
}

// ZerosGPU creates a zero-filled GPU tensor.
// Data is initialized to zeros on GPU.
func (b *Backend) ZerosGPU(shape tensor.Shape, dtype tensor.DataType) *GPUTensor {
	if err := shape.Validate(); err != nil {
		panic(fmt.Sprintf("webgpu: ZerosGPU: invalid shape: %v", err))
	}

	// Create zero-filled data with alignment
	numElements := shape.NumElements()
	//nolint:gosec // G115: numElements and dtype.Size() are non-negative, safe conversion
	byteSize := uint64(numElements * dtype.Size())
	if byteSize < 4 {
		byteSize = 4
	}
	// Align to 4-byte boundary
	alignedSize := (byteSize + 3) &^ 3

	data := make([]byte, alignedSize)
	// data is already zero-filled by make()

	// Create GPU buffer
	buffer := b.createBuffer(
		data,
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst,
	)

	// Track buffer allocation
	b.trackBufferAllocation(alignedSize)

	return &GPUTensor{
		buffer:     buffer,
		shape:      shape,
		dtype:      dtype,
		strides:    shape.ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: alignedSize,
	}
}

// OnesGPU creates a GPU tensor filled with ones.
// Data is initialized to ones on CPU then uploaded to GPU.
func (b *Backend) OnesGPU(shape tensor.Shape, dtype tensor.DataType) *GPUTensor {
	if err := shape.Validate(); err != nil {
		panic(fmt.Sprintf("webgpu: OnesGPU: invalid shape: %v", err))
	}

	numElements := shape.NumElements()
	//nolint:gosec // G115: numElements and dtype.Size() are non-negative, safe conversion
	byteSize := uint64(numElements * dtype.Size())
	if byteSize < 4 {
		byteSize = 4
	}
	// Align to 4-byte boundary
	alignedSize := (byteSize + 3) &^ 3

	data := make([]byte, alignedSize)

	// Fill with ones based on dtype
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numElements; i++ {
			binary.LittleEndian.PutUint32(data[i*4:(i+1)*4], 0x3f800000) // 1.0 in float32
		}
	case tensor.Float64:
		for i := 0; i < numElements; i++ {
			binary.LittleEndian.PutUint64(data[i*8:(i+1)*8], 0x3ff0000000000000) // 1.0 in float64
		}
	case tensor.Int32:
		for i := 0; i < numElements; i++ {
			binary.LittleEndian.PutUint32(data[i*4:(i+1)*4], 1)
		}
	case tensor.Int64:
		for i := 0; i < numElements; i++ {
			binary.LittleEndian.PutUint64(data[i*8:(i+1)*8], 1)
		}
	case tensor.Uint8:
		for i := 0; i < numElements; i++ {
			data[i] = 1
		}
	default:
		panic(fmt.Sprintf("webgpu: OnesGPU: unsupported dtype: %v", dtype))
	}

	// Create GPU buffer
	buffer := b.createBuffer(
		data,
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst,
	)

	// Track buffer allocation
	b.trackBufferAllocation(alignedSize)

	return &GPUTensor{
		buffer:     buffer,
		shape:      shape,
		dtype:      dtype,
		strides:    shape.ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: alignedSize,
	}
}

// RandGPU creates a random GPU tensor with uniform distribution [0, 1).
// Data is generated on CPU using math/rand then uploaded to GPU.
func (b *Backend) RandGPU(shape tensor.Shape, dtype tensor.DataType) *GPUTensor {
	if err := shape.Validate(); err != nil {
		panic(fmt.Sprintf("webgpu: RandGPU: invalid shape: %v", err))
	}

	numElements := shape.NumElements()
	//nolint:gosec // G115: numElements and dtype.Size() are non-negative, safe conversion
	byteSize := uint64(numElements * dtype.Size())
	if byteSize < 4 {
		byteSize = 4
	}
	// Align to 4-byte boundary
	alignedSize := (byteSize + 3) &^ 3

	data := make([]byte, alignedSize)

	// Generate random data based on dtype
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numElements; i++ {
			//nolint:gosec // math/rand is appropriate for ML/statistical purposes
			val := rand.Float32()
			//nolint:gosec // unsafe.Pointer for float32 to uint32 reinterpretation
			binary.LittleEndian.PutUint32(data[i*4:(i+1)*4], *(*uint32)(unsafe.Pointer(&val)))
		}
	case tensor.Float64:
		for i := 0; i < numElements; i++ {
			//nolint:gosec // math/rand is appropriate for ML/statistical purposes
			val := rand.Float64()
			//nolint:gosec // unsafe.Pointer for float64 to uint64 reinterpretation
			binary.LittleEndian.PutUint64(data[i*8:(i+1)*8], *(*uint64)(unsafe.Pointer(&val)))
		}
	case tensor.Int32:
		for i := 0; i < numElements; i++ {
			//nolint:gosec // math/rand is appropriate for ML/statistical purposes
			val := rand.Int31()
			//nolint:gosec // G115: rand.Int31() returns non-negative int32, safe conversion
			binary.LittleEndian.PutUint32(data[i*4:(i+1)*4], uint32(val))
		}
	case tensor.Int64:
		for i := 0; i < numElements; i++ {
			//nolint:gosec // math/rand is appropriate for ML/statistical purposes
			val := rand.Int63()
			//nolint:gosec // G115: rand.Int63() returns non-negative int64, safe conversion
			binary.LittleEndian.PutUint64(data[i*8:(i+1)*8], uint64(val))
		}
	case tensor.Uint8:
		for i := 0; i < numElements; i++ {
			//nolint:gosec // math/rand is appropriate for ML/statistical purposes
			data[i] = uint8(rand.Intn(256))
		}
	default:
		panic(fmt.Sprintf("webgpu: RandGPU: unsupported dtype: %v", dtype))
	}

	// Create GPU buffer
	buffer := b.createBuffer(
		data,
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst,
	)

	// Track buffer allocation
	b.trackBufferAllocation(alignedSize)

	return &GPUTensor{
		buffer:     buffer,
		shape:      shape,
		dtype:      dtype,
		strides:    shape.ComputeStrides(),
		backend:    b,
		computed:   true,
		bufferSize: alignedSize,
	}
}
