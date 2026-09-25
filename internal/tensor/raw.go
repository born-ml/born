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

// Materializer converts backend-specific tensor data to CPU-resident bytes.
// Implemented by GPU backends to abstract the GPU→CPU readback path.
// The returned slice must have length shape.NumElements() * dtype.Size().
type Materializer interface {
	Materialize(t *RawTensor) ([]byte, error)
}

// RawTensor is the low-level tensor representation.
// It uses reference-counted shared buffers for Copy-on-Write semantics.
// Supports lazy GPU evaluation via the Materializer interface: a GPU backend
// sets a Materializer and opaque backendData; Data() calls Materialize to
// trigger the GPU→CPU readback on first access.
type RawTensor struct {
	buffer       *tensorBuffer // Shared reference-counted buffer
	shape        Shape         // Tensor dimensions
	stride       []int         // Memory strides (row-major)
	dtype        DataType      // Runtime type information
	device       Device        // Compute device
	offset       int           // Offset for slicing/views
	backendData  any           // Opaque backend-owned data (managed by backend, not RawTensor).
	materializer Materializer  // Backend that can materialize this tensor (GPU→CPU readback).
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

// copyReadbackData writes GPU readback bytes into the tensor's CPU buffer,
// allocating the buffer if necessary. data may be nil for empty results.
func (r *RawTensor) copyReadbackData(data []byte) {
	if data == nil {
		return
	}
	if r.buffer == nil {
		r.buffer = newTensorBuffer(r.NumElements() * r.dtype.Size())
	}
	copy(r.buffer.data[r.offset:], data)
}

// ensureBuffer allocates a zero-filled CPU buffer if one does not exist yet.
func (r *RawTensor) ensureBuffer() {
	if r.buffer == nil {
		r.buffer = newTensorBuffer(r.NumElements() * r.dtype.Size())
	}
}

// Data returns the raw byte slice.
// For lazy GPU tensors, this triggers data transfer from GPU to CPU (expensive!).
// The materializer (set by GPU backends) is called once on first access; subsequent
// calls return the cached CPU buffer directly.
// WARNING: Direct access to underlying memory. Use with caution.
func (r *RawTensor) Data() []byte {
	// Materializer path: GPU backends set a Materializer and backendData (ADR-019).
	// On first call, Materialize triggers the GPU→CPU readback and caches the result.
	// materializer and backendData are cleared after realization to prevent double-realization.
	if r.materializer != nil && r.device != CPU {
		data, err := r.materializer.Materialize(r)
		if err != nil {
			panic("tensor: materializer readback failed: " + err.Error())
		}
		r.copyReadbackData(data)
		r.materializer = nil
		r.backendData = nil
		r.ensureBuffer()
		return r.buffer.data[r.offset:]
	}

	// Lazy CPU buffer allocation for tensors with nil buffer (e.g. zero-filled creation).
	r.ensureBuffer()
	return r.buffer.data[r.offset:]
}

// BackendData returns the opaque backend-specific data, or nil.
// The returned value is backend-owned; callers must not modify it.
func (r *RawTensor) BackendData() any {
	return r.backendData
}

// SetBackendData sets the opaque backend-specific data.
// The backend is responsible for lifecycle management of the stored value.
func (r *RawTensor) SetBackendData(data any) {
	r.backendData = data
}

// SetMaterializer registers the backend that can materialize this tensor.
// Called by GPU backends when creating lazy tensors (Phase 2 of ADR-019).
// After Data() calls Materialize, the materializer is cleared to avoid double-realization.
func (r *RawTensor) SetMaterializer(m Materializer) {
	r.materializer = m
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
//
// When backendData implements BackendDataRefCounter, Clone calls AddRef so that
// the backend's own reference count stays accurate. Without this call, two clones
// would share the same GPU data with refcount=1; releasing the first clone would
// destroy the GPU buffer while the second clone still holds a pointer to it
// (use-after-free).
//
// Example:
//
//	a := tensor.Ones[float32](Shape{1000, 1000}, backend)
//	b := a.Clone()  // Shares buffer with a (just increments refCount)
//	c := a.Add(b)   // May use inplace if refCount allows
func (r *RawTensor) Clone() *RawTensor {
	if r.buffer != nil {
		r.buffer.addRef()
	}
	// Notify the backend that backendData now has one more owner. GPU backends
	// implement BackendDataRefCounter so their internal refcount matches the
	// number of live RawTensor aliases pointing at the same GPU buffer.
	if rc, ok := r.backendData.(BackendDataRefCounter); ok {
		rc.AddRef()
	}
	return &RawTensor{
		buffer:       r.buffer,
		shape:        r.shape.Clone(),
		stride:       append([]int(nil), r.stride...),
		dtype:        r.dtype,
		device:       r.device,
		offset:       r.offset,
		backendData:  r.backendData,  // Shared reference; backend manages lifecycle.
		materializer: r.materializer, // Shared reference; backend is stateless for this call.
	}
}

// Release decrements the reference count and deallocates if it reaches 0.
func (r *RawTensor) Release() {
	if r.buffer != nil {
		r.buffer.release()
	}
}

// IsUnique returns true if this tensor is the only reference to the buffer.
// GPU-only tensors with nil CPU buffer are always "unique" (no shared CPU data).
func (r *RawTensor) IsUnique() bool {
	if r.buffer == nil {
		return true
	}
	return r.buffer.isUnique()
}

// ForceNonUnique temporarily increases refCount to prevent inplace modifications.
func (r *RawTensor) ForceNonUnique() func() {
	if r.buffer == nil {
		return func() {}
	}
	r.buffer.addRef()
	return func() {
		r.buffer.release()
	}
}
