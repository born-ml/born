//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
// Uses go-webgpu (github.com/go-webgpu/webgpu) for zero-CGO WebGPU bindings.
package webgpu

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
	"github.com/gogpu/gputypes"
)

// Backend implements tensor operations on GPU using WebGPU.
type Backend struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue

	// Shader and pipeline cache
	shaders   map[string]*wgpu.ShaderModule
	pipelines map[string]*wgpu.ComputePipeline
	mu        sync.RWMutex

	// Device info
	adapterInfo *wgpu.AdapterInfoGo

	// Buffer pool for memory management
	bufferPool *BufferPool

	// Lazy mode: when true, operations return lazy tensors that keep data on GPU
	// until Data() is explicitly called. This is the key optimization for
	// Phase 3 Integration - eliminates readBuffer() bottleneck.
	// Default: true for optimal performance.
	LazyMode bool

	// Memory tracking
	memoryStats struct {
		totalAllocatedBytes uint64
		peakMemoryBytes     uint64
		activeBuffers       int64
		mu                  sync.RWMutex
	}

	// Command batching for lazy mode performance optimization.
	// Commands are accumulated and submitted together to reduce GPU sync overhead.
	pendingCommands []*wgpu.CommandBuffer
	pendingMu       sync.Mutex
	maxBatchSize    int // Maximum commands before auto-flush (0 = no limit)
}

// New creates a new WebGPU backend.
// Returns an error if WebGPU is not available or initialization fails.
func New() (backend *Backend, err error) {
	// Recover from panic if wgpu_native library is not found.
	defer func() {
		if r := recover(); r != nil {
			backend = nil
			err = fmt.Errorf("webgpu: native library not available: %v", r)
		}
	}()

	// Create WebGPU instance
	instance, createErr := wgpu.CreateInstance(nil)
	if createErr != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", createErr)
	}

	// Request adapter (GPU)
	adapter, adapterErr := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: gputypes.PowerPreferenceHighPerformance,
	})
	if adapterErr != nil {
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request adapter: %w", adapterErr)
	}

	// Get adapter info (optional - don't fail if unavailable)
	adapterInfo, _ := adapter.GetInfo()
	// Note: adapterInfo may be nil if GetInfo fails, which is OK

	// Request device
	device, deviceErr := adapter.RequestDevice(nil)
	if deviceErr != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request device: %w", deviceErr)
	}

	// Get default queue
	queue := device.GetQueue()
	if queue == nil {
		device.Release()
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to get queue")
	}

	b := &Backend{
		instance:    instance,
		adapter:     adapter,
		device:      device,
		queue:       queue,
		shaders:     make(map[string]*wgpu.ShaderModule),
		pipelines:   make(map[string]*wgpu.ComputePipeline),
		adapterInfo: adapterInfo,
		bufferPool:  NewBufferPool(device),
		LazyMode:    true, // Default: lazy mode enabled for optimal performance
	}

	return b, nil
}

// SetLazyMode enables or disables lazy evaluation mode.
// When enabled (default), operations return lazy tensors that keep data on GPU
// until Data() is explicitly called. This dramatically improves performance
// by eliminating unnecessary GPU→CPU transfers.
// When disabled, operations immediately transfer results to CPU (slower).
func (b *Backend) SetLazyMode(enabled bool) {
	b.LazyMode = enabled
}

// queueCommand adds a command buffer to the pending queue for batch submission.
// This reduces GPU sync overhead by submitting multiple commands at once.
// Commands are automatically flushed when reading data or when batch size limit is reached.
func (b *Backend) queueCommand(cmdBuffer *wgpu.CommandBuffer) {
	b.pendingMu.Lock()
	defer b.pendingMu.Unlock()

	b.pendingCommands = append(b.pendingCommands, cmdBuffer)

	// Auto-flush if batch size limit is reached (0 = no limit)
	if b.maxBatchSize > 0 && len(b.pendingCommands) >= b.maxBatchSize {
		b.flushCommandsLocked()
	}
}

// flushCommands submits all pending command buffers to the GPU queue.
// This is called automatically before reading data from GPU.
func (b *Backend) flushCommands() {
	b.pendingMu.Lock()
	defer b.pendingMu.Unlock()
	b.flushCommandsLocked()
}

// flushCommandsLocked submits all pending command buffers (must hold pendingMu lock).
func (b *Backend) flushCommandsLocked() {
	if len(b.pendingCommands) == 0 {
		return
	}
	b.queue.Submit(b.pendingCommands...)
	b.pendingCommands = b.pendingCommands[:0]
}

// FlushCommands submits all pending command buffers to the GPU queue.
// Call this when you need to ensure all queued operations are executed.
// Note: This is called automatically before reading data from GPU buffers.
func (b *Backend) FlushCommands() {
	b.flushCommands()
}

// SetMaxBatchSize sets the maximum number of commands to accumulate before auto-flush.
// Set to 0 (default) to disable auto-flush limit.
// Typical values: 32-128 for balanced latency/throughput.
func (b *Backend) SetMaxBatchSize(size int) {
	b.pendingMu.Lock()
	defer b.pendingMu.Unlock()
	b.maxBatchSize = size
}

// Release releases all WebGPU resources.
// Must be called when the backend is no longer needed.
func (b *Backend) Release() {
	// Flush any pending commands before releasing resources
	b.flushCommands()

	b.mu.Lock()
	defer b.mu.Unlock()

	// Release buffer pool
	if b.bufferPool != nil {
		b.bufferPool.Clear()
		b.bufferPool = nil
	}

	// Release pipelines
	for _, p := range b.pipelines {
		p.Release()
	}
	b.pipelines = nil

	// Release shaders
	for _, s := range b.shaders {
		s.Release()
	}
	b.shaders = nil

	// Release WebGPU objects
	if b.queue != nil {
		b.queue.Release()
		b.queue = nil
	}
	if b.device != nil {
		b.device.Release()
		b.device = nil
	}
	if b.adapter != nil {
		b.adapter.Release()
		b.adapter = nil
	}
	if b.instance != nil {
		b.instance.Release()
		b.instance = nil
	}
}

// Name returns the backend name.
func (b *Backend) Name() string {
	if b.adapterInfo != nil {
		return fmt.Sprintf("WebGPU (%s)", b.adapterInfo.Device)
	}
	return "WebGPU"
}

// Device returns the compute device.
func (b *Backend) Device() tensor.Device {
	return tensor.WebGPU
}

// AdapterInfo returns information about the GPU adapter.
func (b *Backend) AdapterInfo() *wgpu.AdapterInfoGo {
	return b.adapterInfo
}

// IsAvailable checks if WebGPU is available on this system.
func IsAvailable() (available bool) {
	// Recover from panic if wgpu_native library is not found.
	defer func() {
		if r := recover(); r != nil {
			available = false
		}
	}()

	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return false
	}
	defer instance.Release()

	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		return false
	}
	adapter.Release()

	return true
}

// ListAdapters returns information about all available GPU adapters.
func ListAdapters() (adapters []*wgpu.AdapterInfoGo, err error) {
	// Recover from panic if wgpu_native library is not found.
	defer func() {
		if r := recover(); r != nil {
			adapters = nil
			err = fmt.Errorf("webgpu: native library not available: %v", r)
		}
	}()

	instance, createErr := wgpu.CreateInstance(nil)
	if createErr != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", createErr)
	}
	defer instance.Release()

	// For now, just return the default adapter
	// WebGPU spec doesn't have a way to enumerate all adapters
	adapter, adapterErr := instance.RequestAdapter(nil)
	if adapterErr != nil {
		return nil, fmt.Errorf("webgpu: no adapters available: %w", adapterErr)
	}
	defer adapter.Release()

	info, infoErr := adapter.GetInfo()
	if infoErr != nil {
		return nil, fmt.Errorf("webgpu: failed to get adapter info: %w", infoErr)
	}

	return []*wgpu.AdapterInfoGo{info}, nil
}

// MemoryStats represents GPU memory usage statistics.
type MemoryStats struct {
	// Total bytes allocated since backend creation
	TotalAllocatedBytes uint64
	// Peak memory usage in bytes
	PeakMemoryBytes uint64
	// Number of currently active buffers
	ActiveBuffers int64
	// Buffer pool statistics
	PoolAllocated uint64
	PoolReleased  uint64
	PoolHits      uint64
	PoolMisses    uint64
	PooledBuffers int
}

// MemoryStats returns current GPU memory usage statistics.
func (b *Backend) MemoryStats() MemoryStats {
	b.memoryStats.mu.RLock()
	totalAllocated := b.memoryStats.totalAllocatedBytes
	peakMemory := b.memoryStats.peakMemoryBytes
	activeBuffers := b.memoryStats.activeBuffers
	b.memoryStats.mu.RUnlock()

	// Get buffer pool stats
	allocated, released, hits, misses, pooledCount := b.bufferPool.Stats()

	return MemoryStats{
		TotalAllocatedBytes: totalAllocated,
		PeakMemoryBytes:     peakMemory,
		ActiveBuffers:       activeBuffers,
		PoolAllocated:       allocated,
		PoolReleased:        released,
		PoolHits:            hits,
		PoolMisses:          misses,
		PooledBuffers:       pooledCount,
	}
}

// trackBufferAllocation records a buffer allocation in memory statistics.
func (b *Backend) trackBufferAllocation(size uint64) {
	b.memoryStats.mu.Lock()
	defer b.memoryStats.mu.Unlock()

	b.memoryStats.totalAllocatedBytes += size
	b.memoryStats.activeBuffers++

	// Update peak memory if needed
	currentMemory := b.memoryStats.totalAllocatedBytes
	if currentMemory > b.memoryStats.peakMemoryBytes {
		b.memoryStats.peakMemoryBytes = currentMemory
	}
}

// trackBufferRelease records a buffer release in memory statistics.
func (b *Backend) trackBufferRelease(size uint64) {
	b.memoryStats.mu.Lock()
	defer b.memoryStats.mu.Unlock()

	if b.memoryStats.totalAllocatedBytes >= size {
		b.memoryStats.totalAllocatedBytes -= size
	}
	b.memoryStats.activeBuffers--
}

// Gather selects elements along dim using index tensor on GPU.
func (b *Backend) Gather(input *tensor.RawTensor, dim int, indices *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runGatherLazy(input, dim, indices)
	} else {
		result, err = b.runGather(input, dim, indices)
	}
	if err != nil {
		panic("webgpu: Gather: " + err.Error())
	}
	return result
}

// Where performs conditional element selection on GPU.
// result[i] = condition[i] != 0 ? x[i] : y[i].
func (b *Backend) Where(condition, x, y *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runWhereLazy(condition, x, y)
	} else {
		result, err = b.runWhere(condition, x, y)
	}
	if err != nil {
		panic("webgpu: Where: " + err.Error())
	}
	return result
}

// Embedding performs embedding lookup on GPU.
// weight: [num_embeddings, embedding_dim], indices: int32 tensor.
// Returns: [...indices_shape, embedding_dim].
func (b *Backend) Embedding(weight, indices *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runEmbedding(weight, indices)
	if err != nil {
		panic("webgpu: Embedding: " + err.Error())
	}
	return result
}

// ReadGPUBuffer implements tensor.LazyBackend interface.
// Reads data from a GPU buffer to CPU memory.
// bufferPtr must be *wgpu.Buffer.
func (b *Backend) ReadGPUBuffer(bufferPtr unsafe.Pointer, size uint64) ([]byte, error) {
	buffer := (*wgpu.Buffer)(bufferPtr)
	return b.readBuffer(buffer, size)
}

// ReleaseGPUBuffer implements tensor.LazyBackend interface.
// Releases a GPU buffer when no longer needed.
// bufferPtr must be *wgpu.Buffer.
func (b *Backend) ReleaseGPUBuffer(bufferPtr unsafe.Pointer) {
	buffer := (*wgpu.Buffer)(bufferPtr)
	if buffer != nil {
		buffer.Release()
	}
}

// Conv2DInputBackward computes gradient with respect to input for Conv2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) Conv2DInputBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	panic("webgpu: Conv2DInputBackward not implemented")
}

// Conv2DKernelBackward computes gradient with respect to kernel for Conv2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) Conv2DKernelBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	panic("webgpu: Conv2DKernelBackward not implemented")
}

// MaxPool2DBackward computes gradient with respect to input for MaxPool2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) MaxPool2DBackward(input, grad *tensor.RawTensor, maxIndices []int, kernelSize, stride int) *tensor.RawTensor {
	panic("webgpu: MaxPool2DBackward not implemented")
}
