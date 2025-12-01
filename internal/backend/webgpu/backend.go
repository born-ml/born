//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
// Uses go-webgpu (github.com/go-webgpu/webgpu) for zero-CGO WebGPU bindings.
package webgpu

import (
	"fmt"
	"sync"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
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
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
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
	}

	return b, nil
}

// Release releases all WebGPU resources.
// Must be called when the backend is no longer needed.
func (b *Backend) Release() {
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

// Gather selects elements along dim using index tensor (not implemented yet).
func (b *Backend) Gather(_ *tensor.RawTensor, _ int, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Gather not implemented yet - TODO(TASK-016)")
}

// Where performs conditional element selection (not implemented yet).
func (b *Backend) Where(_, _, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Where not implemented yet - TODO(TASK-016)")
}

// Embedding performs embedding lookup (not implemented yet).
func (b *Backend) Embedding(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Embedding not implemented yet")
}
