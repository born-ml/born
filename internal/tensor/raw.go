package tensor

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Device represents the compute device for tensor operations.
type Device int

// Supported compute devices.
const (
	CPU Device = iota
	CUDA
	Vulkan
	Metal
	WebGPU
)

// String returns a human-readable device name.
func (d Device) String() string {
	switch d {
	case CPU:
		return "CPU"
	case CUDA:
		return "CUDA"
	case Vulkan:
		return "Vulkan"
	case Metal:
		return "Metal"
	case WebGPU:
		return "WebGPU"
	default:
		return "Unknown"
	}
}

// tensorBuffer is a reference-counted shared buffer for Copy-on-Write semantics.
// This enables cheap cloning and inplace optimizations when refCount == 1.
type tensorBuffer struct {
	data     []byte
	refCount atomic.Int32
	mu       sync.Mutex // For safe deallocation
}

// newTensorBuffer creates a new reference-counted buffer with refCount = 1.
func newTensorBuffer(size int) *tensorBuffer {
	buf := &tensorBuffer{
		data: make([]byte, size),
	}
	buf.refCount.Store(1)
	return buf
}

// addRef increments the reference count (for Clone operations).
func (tb *tensorBuffer) addRef() {
	tb.refCount.Add(1)
}

// release decrements the reference count and deallocates if it reaches 0.
func (tb *tensorBuffer) release() {
	if tb.refCount.Add(-1) == 0 {
		tb.mu.Lock()
		defer tb.mu.Unlock()
		tb.data = nil
	}
}

// isUnique returns true if this buffer has only one reference (enables inplace ops).
func (tb *tensorBuffer) isUnique() bool {
	return tb.refCount.Load() == 1
}

// RawTensor is the low-level tensor representation.
// It uses reference-counted shared buffers for Copy-on-Write semantics.
// Supports lazy GPU evaluation: data is transferred from GPU only when Data() is called.
type RawTensor struct {
	buffer  *tensorBuffer // Shared reference-counted buffer
	shape   Shape         // Tensor dimensions
	stride  []int         // Memory strides (row-major)
	dtype   DataType      // Runtime type information
	device  Device        // Compute device
	offset  int           // Offset for slicing/views
	gpuData *LazyGPUData  // Lazy GPU data (nil for CPU tensors)
}

// NewRaw creates a new RawTensor with the given shape and type.
// Memory is allocated but not initialized (contains zeros).
func NewRaw(shape Shape, dtype DataType, device Device) (*RawTensor, error) {
	if err := shape.Validate(); err != nil {
		return nil, fmt.Errorf("invalid shape: %w", err)
	}

	numElements := shape.NumElements()
	byteSize := numElements * dtype.Size()

	return &RawTensor{
		buffer: newTensorBuffer(byteSize),
		shape:  shape.Clone(),
		stride: shape.ComputeStrides(),
		dtype:  dtype,
		device: device,
		offset: 0,
	}, nil
}

// Shape returns the tensor's shape.
func (r *RawTensor) Shape() Shape {
	return r.shape
}

// Strides returns the tensor's memory strides.
func (r *RawTensor) Strides() []int {
	return r.stride
}

// DType returns the tensor's data type.
func (r *RawTensor) DType() DataType {
	return r.dtype
}

// Device returns the tensor's compute device.
func (r *RawTensor) Device() Device {
	return r.device
}

// NumElements returns the total number of elements.
func (r *RawTensor) NumElements() int {
	return r.shape.NumElements()
}

// ByteSize returns the total memory size in bytes.
func (r *RawTensor) ByteSize() int {
	return r.NumElements() * r.dtype.Size()
}

// Data returns the raw byte slice.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU (expensive!).
// WARNING: Direct access to underlying memory. Use with caution.
func (r *RawTensor) Data() []byte {
	// Lazy GPU realization: transfer data from GPU if not already done
	if r.gpuData != nil && !r.gpuData.IsRealized() {
		data, err := r.gpuData.Realize()
		if err != nil {
			panic("tensor: failed to realize GPU data: " + err.Error())
		}
		if data != nil {
			copy(r.buffer.data[r.offset:], data)
		}
	}
	return r.buffer.data[r.offset:]
}

// IsLazy returns true if this tensor has unrealized GPU data.
// Use this to check if Data() will trigger an expensive GPU→CPU transfer.
func (r *RawTensor) IsLazy() bool {
	return r.gpuData != nil && !r.gpuData.IsRealized()
}

// GPUData returns the lazy GPU data reference, if any.
// Returns nil for CPU-only tensors.
func (r *RawTensor) GPUData() *LazyGPUData {
	return r.gpuData
}

// SetGPUData sets the lazy GPU data reference.
// This is used by GPU backends to create lazy tensors.
func (r *RawTensor) SetGPUData(gpuData *LazyGPUData) {
	r.gpuData = gpuData
}

// AsFloat32 interprets the data as []float32.
// Panics if the tensor's dtype is not Float32.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsFloat32() []float32 {
	if r.dtype != Float32 {
		panic(fmt.Sprintf("tensor dtype is %s, not float32", r.dtype))
	}
	// Trigger lazy realization if needed
	data := r.Data()
	//nolint:gosec // unsafe.Slice for zero-copy performance, bounds checked by NumElements()
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), r.NumElements())
}

// AsFloat64 interprets the data as []float64.
// Panics if the tensor's dtype is not Float64.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsFloat64() []float64 {
	if r.dtype != Float64 {
		panic(fmt.Sprintf("tensor dtype is %s, not float64", r.dtype))
	}
	// Trigger lazy realization if needed
	data := r.Data()
	//nolint:gosec // unsafe.Slice for zero-copy performance, bounds checked by NumElements()
	return unsafe.Slice((*float64)(unsafe.Pointer(&data[0])), r.NumElements())
}

// AsInt32 interprets the data as []int32.
// Panics if the tensor's dtype is not Int32.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsInt32() []int32 {
	if r.dtype != Int32 {
		panic(fmt.Sprintf("tensor dtype is %s, not int32", r.dtype))
	}
	// Trigger lazy realization if needed
	data := r.Data()
	//nolint:gosec // unsafe.Slice for zero-copy performance, bounds checked by NumElements()
	return unsafe.Slice((*int32)(unsafe.Pointer(&data[0])), r.NumElements())
}

// AsInt64 interprets the data as []int64.
// Panics if the tensor's dtype is not Int64.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsInt64() []int64 {
	if r.dtype != Int64 {
		panic(fmt.Sprintf("tensor dtype is %s, not int64", r.dtype))
	}
	// Trigger lazy realization if needed
	data := r.Data()
	//nolint:gosec // unsafe.Slice for zero-copy performance, bounds checked by NumElements()
	return unsafe.Slice((*int64)(unsafe.Pointer(&data[0])), r.NumElements())
}

// AsUint8 interprets the data as []uint8.
// Panics if the tensor's dtype is not Uint8.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsUint8() []uint8 {
	if r.dtype != Uint8 {
		panic(fmt.Sprintf("tensor dtype is %s, not uint8", r.dtype))
	}
	// Trigger lazy realization if needed (Data() returns []byte = []uint8)
	return r.Data()
}

// AsBool interprets the data as []bool.
// Panics if the tensor's dtype is not Bool.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU.
func (r *RawTensor) AsBool() []bool {
	if r.dtype != Bool {
		panic(fmt.Sprintf("tensor dtype is %s, not bool", r.dtype))
	}
	// Trigger lazy realization if needed
	data := r.Data()
	//nolint:gosec // unsafe.Slice for zero-copy performance, bounds checked by NumElements()
	return unsafe.Slice((*bool)(unsafe.Pointer(&data[0])), r.NumElements())
}

// Clone creates a shallow copy of the RawTensor (shares buffer with reference counting).
// The buffer is reference-counted and will be copied only when modified (copy-on-write).
// This enables cheap cloning and inplace optimizations when refCount == 1.
// Note: GPU lazy data is shared (same underlying GPU buffer).
//
// Example:
//
//	a := tensor.Ones[float32](Shape{1000, 1000}, backend)
//	b := a.Clone()  // Shares buffer with a (just increments refCount)
//	c := a.Add(b)   // May use inplace if refCount allows
func (r *RawTensor) Clone() *RawTensor {
	r.buffer.addRef() // Increment reference count
	return &RawTensor{
		buffer:  r.buffer, // Share the same buffer
		shape:   r.shape.Clone(),
		stride:  append([]int(nil), r.stride...), // Copy strides
		dtype:   r.dtype,
		device:  r.device,
		offset:  r.offset,
		gpuData: r.gpuData, // Share GPU data reference
	}
}

// Release decrements the reference count and deallocates if it reaches 0.
// This is called automatically when a tensor is no longer needed (e.g., by GC finalizer).
func (r *RawTensor) Release() {
	r.buffer.release()
}

// IsUnique returns true if this tensor is the only reference to the buffer.
// When true, backends can perform inplace operations for better performance.
func (r *RawTensor) IsUnique() bool {
	return r.buffer.isUnique()
}

// ForceNonUnique temporarily increases refCount to prevent inplace modifications.
// Returns a cleanup function that MUST be called to restore refCount (use defer).
//
// This is used by autodiff backend to preserve original input values:
// inplace optimizations would corrupt the computational graph.
//
// Example:
//
//	defer tensor.ForceNonUnique()()
//	result := backend.Mul(tensor, other)  // No inplace modification!
func (r *RawTensor) ForceNonUnique() func() {
	r.buffer.addRef() // Increment refcount
	return func() {
		r.buffer.release() // Decrement refcount
	}
}
