//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
// Uses gogpu/wgpu (github.com/gogpu/wgpu) for pure Go, zero-CGO WebGPU bindings.
package webgpu

import (
	"context"
	"fmt"
	"sync"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// pipelineEntry caches a compute pipeline together with its layouts.
// The pipeline layout must remain alive because Vulkan references it
// during vkCmdBindDescriptorSets on every SetBindGroup call.
type pipelineEntry struct {
	pipeline       *wgpu.ComputePipeline
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
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
}

// New creates a new WebGPU backend.
// Returns an error if WebGPU is not available or initialization fails.
func New() (*Backend, error) {
	// Create WebGPU instance. Vulkan is the primary compute backend —
	// stable across all platforms and GPU vendors.
	instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{
		Backends: wgpu.BackendsVulkan,
	})
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

// flushCommands is a no-op retained for backwards compatibility.
// All GPU operations now use immediate submit to prevent buffer lifetime issues.
func (b *Backend) flushCommands() {}

// Release releases all WebGPU resources.
// Must be called when the backend is no longer needed.
func (b *Backend) Release() {
	// Ensure GPU is fully idle before destroying resources.
	// Without this, rapid create/destroy cycles (e.g. test suites) can
	// overwhelm the driver on iGPUs with shared memory.
	if b.device != nil {
		b.device.Poll(wgpu.PollWait)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// Release buffer pool
	if b.bufferPool != nil {
		b.bufferPool.Clear()
		b.bufferPool = nil
	}

	// Release pipelines and their associated layouts.
	for _, entry := range b.pipelines {
		entry.pipeline.Release()
		entry.layout.Release()
		if entry.pipelineLayout != nil {
			entry.pipelineLayout.Release()
		}
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

// IsAvailable checks if WebGPU with compute shader support is available.
// Returns false on software renderers that don't support compute pipelines.
func IsAvailable() bool {
	backend, err := New()
	if err != nil {
		return false
	}
	defer backend.Release()

	// Verify compute shaders actually work by creating a minimal pipeline.
	// Software renderers pass adapter/device creation but fail here.
	shader, err := backend.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "availability-check",
		WGSL:  "@compute @workgroup_size(1) fn main() {}",
	})
	if err != nil {
		return false
	}
	defer shader.Release()

	bgl, err := backend.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{})
	if err != nil {
		return false
	}
	defer bgl.Release()

	pl, err := backend.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return false
	}
	defer pl.Release()

	pipeline, err := backend.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "availability-check", Layout: pl, Module: shader, EntryPoint: "main",
	})
	if err != nil {
		return false
	}
	pipeline.Release()

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
// Reads data from a GPU staging buffer (MapRead | CopyDst) to CPU memory.
// bufferPtr must point to a *wgpu.Buffer created with BufferUsageMapRead.
//
// The lazy path (runBinaryOpLazy, runUnaryOpLazy, etc.) creates a staging buffer
// in the same encoder as the compute pass (unified encoder pattern). The staging
// buffer already has the computed data after the batch submit completes.
//
// Sequence:
//  1. flushCommands() — submit all batched compute+copy commands.
//  2. Poll(PollWait) — block until the GPU completes the submitted commands.
//     This is required because Map()'s internal Poll(PollPoll) may resolve
//     immediately using a prior submission's fence when a backend has received
//     multiple submits (DX12/Vulkan fence ordering artifact with batched commands).
//  3. Map the staging buffer (blocks until the GPU fence resolves).
//  4. Copy data to CPU slice, Unmap.
func (b *Backend) ReadGPUBuffer(bufferPtr unsafe.Pointer, size uint64) ([]byte, error) {
	// Flush all pending batched commands (compute + CopyBufferToBuffer to staging).
	b.flushCommands()

	// Wait for ALL pending GPU work to complete before mapping.
	// Without this Poll, Map()'s internal Poll(PollPoll) can return "done"
	// prematurely on backends (DX12) that signal fences conservatively,
	// causing the staging buffer to be read before the compute pass has
	// finished writing to it — resulting in zeros.
	// This matches the pattern used by readBuffer() for the split-encoder path.
	b.device.Poll(wgpu.PollWait)

	buffer := (*wgpu.Buffer)(bufferPtr)

	// Map the staging buffer. After Poll(PollWait) the GPU is idle, so
	// Map() returns immediately without blocking on the fence.
	ctx := context.Background()
	if err := buffer.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: failed to map staging buffer: %w", err)
	}
	defer func() { _ = buffer.Unmap() }()

	mappedRange, err := buffer.MappedRange(0, size)
	if err != nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: failed to get mapped range: %w", err)
	}
	defer mappedRange.Release()

	data := mappedRange.Bytes()
	if data == nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: MappedRange.Bytes() returned nil (buffer may have been unmapped or released)")
	}
	if uint64(len(data)) < size {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: MappedRange.Bytes() returned %d bytes but need %d (buffer size mismatch)", len(data), size)
	}
	result := make([]byte, size)
	copy(result, data)
	return result, nil
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
