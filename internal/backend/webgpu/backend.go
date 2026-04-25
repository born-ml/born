//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
// Uses gogpu/wgpu (github.com/gogpu/wgpu) for pure Go, zero-CGO WebGPU bindings.
package webgpu

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// pipelineEntry caches a compute pipeline together with its bind group layout.
// gogpu/wgpu does not expose GetBindGroupLayout on a pipeline,
// so we store the layout alongside the pipeline at creation time.
type pipelineEntry struct {
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout
}

// Backend implements tensor operations on GPU using WebGPU.
type Backend struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue

	// Shader and pipeline cache
	shaders   map[string]*wgpu.ShaderModule
	pipelines map[string]pipelineEntry
	mu        sync.RWMutex

	// Device info
	adapterInfo *wgpu.AdapterInfo

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
func New() (*Backend, error) {
	// Create WebGPU instance.
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", err)
	}

	// Request adapter (GPU).
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: gputypes.PowerPreferenceHighPerformance,
	})
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request adapter: %w", err)
	}

	// Get adapter info. In gogpu/wgpu, Info() returns AdapterInfo by value.
	info := adapter.Info()

	// Request device.
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request device: %w", err)
	}

	// Get default queue. In gogpu/wgpu the queue is accessed via device.Queue().
	queue := device.Queue()
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
		pipelines:   make(map[string]pipelineEntry),
		adapterInfo: &info,
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
	// Submit returns (submissionIndex, error); errors are non-fatal for flush.
	_, _ = b.queue.Submit(b.pendingCommands...)
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

	// Release pipelines and their associated bind group layouts.
	for _, entry := range b.pipelines {
		entry.pipeline.Release()
		entry.layout.Release()
	}
	b.pipelines = nil

	// Release shaders
	for _, s := range b.shaders {
		s.Release()
	}
	b.shaders = nil

	// Release WebGPU objects.
	// Note: Queue is owned by Device in gogpu/wgpu and released via device.Release().
	b.queue = nil
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
		return fmt.Sprintf("WebGPU (%s)", b.adapterInfo.Name)
	}
	return "WebGPU"
}

// Device returns the compute device.
func (b *Backend) Device() tensor.Device {
	return tensor.WebGPU
}

// AdapterInfo returns information about the GPU adapter.
func (b *Backend) AdapterInfo() *wgpu.AdapterInfo {
	return b.adapterInfo
}

// IsAvailable checks if WebGPU is available on this system.
func IsAvailable() bool {
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
func ListAdapters() ([]*wgpu.AdapterInfo, error) {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", err)
	}
	defer instance.Release()

	// WebGPU spec doesn't expose adapter enumeration; return the default adapter.
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: no adapters available: %w", err)
	}
	defer adapter.Release()

	// In gogpu/wgpu, Info() returns AdapterInfo by value (no error).
	info := adapter.Info()
	return []*wgpu.AdapterInfo{&info}, nil
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
