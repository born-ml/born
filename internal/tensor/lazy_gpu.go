//go:build !wasm

// Package tensor provides tensor data structures for the Born ML framework.
package tensor

import (
	"runtime"
	"sync"
	"unsafe"
)

// LazyBackend is an interface for backends that support lazy GPU evaluation.
// The backend must implement ReadGPUBuffer to transfer data from GPU to CPU.
type LazyBackend interface {
	// ReadGPUBuffer reads data from a GPU buffer to CPU memory.
	// bufferPtr is unsafe.Pointer to *wgpu.Buffer (or similar GPU buffer type).
	// size is the number of bytes to read.
	// Returns the CPU data or an error.
	ReadGPUBuffer(bufferPtr unsafe.Pointer, size uint64) ([]byte, error)

	// ReleaseGPUBuffer releases the GPU buffer when no longer needed.
	ReleaseGPUBuffer(bufferPtr unsafe.Pointer)
}

// LazyGPUData holds a reference to GPU-resident data for lazy evaluation.
// When Data() is called on a RawTensor with LazyGPUData, the data is
// transferred from GPU to CPU only at that point (lazy realization).
type LazyGPUData struct {
	bufferPtr unsafe.Pointer // Pointer to GPU buffer (*wgpu.Buffer)
	size      uint64         // Buffer size in bytes
	backend   LazyBackend    // Backend for reading/releasing buffer
	realized  bool           // Whether data has been transferred to CPU
	mu        sync.Mutex     // Protects realized flag and transfer
}

// NewLazyGPUData creates a new LazyGPUData referencing a GPU buffer.
// The GPU buffer will be automatically released when garbage collected.
func NewLazyGPUData(bufferPtr unsafe.Pointer, size uint64, backend LazyBackend) *LazyGPUData {
	l := &LazyGPUData{
		bufferPtr: bufferPtr,
		size:      size,
		backend:   backend,
		realized:  false,
	}

	// Release GPU buffer when garbage collected to prevent memory leaks
	runtime.SetFinalizer(l, func(lg *LazyGPUData) {
		lg.Release()
	})

	return l
}

// IsRealized returns whether the GPU data has been transferred to CPU.
func (l *LazyGPUData) IsRealized() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.realized
}

// MarkRealized marks the GPU data as realized (transferred to CPU).
func (l *LazyGPUData) MarkRealized() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.realized = true
}

// Realize transfers data from GPU to CPU and returns it.
// This is called lazily when Data() is accessed.
// Thread-safe: multiple goroutines can safely call this.
// After realization, the GPU buffer is released to free GPU memory.
func (l *LazyGPUData) Realize() ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Already realized - this shouldn't happen but handle gracefully
	if l.realized {
		return nil, nil
	}

	// Read data from GPU
	data, err := l.backend.ReadGPUBuffer(l.bufferPtr, l.size)
	if err != nil {
		return nil, err
	}

	l.realized = true

	// Release GPU buffer after copying to CPU - we don't need it anymore
	if l.bufferPtr != nil && l.backend != nil {
		l.backend.ReleaseGPUBuffer(l.bufferPtr)
		l.bufferPtr = nil
	}

	return data, nil
}

// Release releases the GPU buffer.
// Called when the tensor is no longer needed.
func (l *LazyGPUData) Release() {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.bufferPtr != nil && l.backend != nil {
		l.backend.ReleaseGPUBuffer(l.bufferPtr)
		l.bufferPtr = nil
	}
}

// BufferPtr returns the underlying GPU buffer pointer.
// This is used by backend operations that need to chain GPU operations.
func (l *LazyGPUData) BufferPtr() unsafe.Pointer {
	return l.bufferPtr
}

// Size returns the buffer size in bytes.
func (l *LazyGPUData) Size() uint64 {
	return l.size
}

// NewLazyRaw creates a new RawTensor with lazy GPU data.
// The data is not transferred from GPU until Data() is called.
func NewLazyRaw(shape Shape, dtype DataType, device Device, gpuData *LazyGPUData) (*RawTensor, error) {
	if err := shape.Validate(); err != nil {
		return nil, err
	}

	numElements := shape.NumElements()
	byteSize := numElements * dtype.Size()

	// Create buffer but don't allocate CPU memory yet - it will be filled lazily
	return &RawTensor{
		buffer:  newTensorBuffer(byteSize),
		shape:   shape.Clone(),
		stride:  shape.ComputeStrides(),
		dtype:   dtype,
		device:  device,
		offset:  0,
		gpuData: gpuData,
	}, nil
}
