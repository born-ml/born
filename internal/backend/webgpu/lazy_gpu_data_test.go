//go:build windows || linux

package webgpu

import (
	"testing"
	"unsafe"

	"github.com/gogpu/gputypes"
)

// TestForceRelease_NilBuffer verifies ForceRelease is safe on a LazyGPUData
// whose GPU buffer has already been released (bufferPtr == nil). This models
// the double-release scenario: once by Realize, once by ReclaimMemory.
func TestForceRelease_NilBuffer(t *testing.T) {
	l := &LazyGPUData{
		bufferPtr: nil,
		refCount:  1,
	}
	// Must not panic even though bufferPtr is nil and backend is nil.
	l.ForceRelease()

	if l.RefCount() != 0 {
		t.Errorf("ForceRelease: refCount = %d, want 0", l.RefCount())
	}
}

// TestForceRelease_SetsRefCountZero verifies that ForceRelease resets the
// reference count to zero, covering the Clone/AddRef path where refCount > 1.
// Without ForceRelease the old ReclaimMemory guard (RefCount() > 1) would skip
// cloned tensors entirely and leak GPU memory.
func TestForceRelease_SetsRefCountZero(t *testing.T) {
	l := &LazyGPUData{
		bufferPtr: nil,
		refCount:  3, // simulates two Clone() calls after original allocation
	}
	l.ForceRelease()

	if l.RefCount() != 0 {
		t.Errorf("ForceRelease with refCount=3: refCount = %d, want 0", l.RefCount())
	}
}

// TestForceRelease_IdempotentOnNilPtr verifies that calling ForceRelease twice
// does not panic — the second call sees bufferPtr == nil and is a no-op.
func TestForceRelease_IdempotentOnNilPtr(_ *testing.T) {
	l := &LazyGPUData{
		bufferPtr: nil,
		refCount:  1,
	}
	l.ForceRelease()
	l.ForceRelease() // second call — must not panic
}

// TestForceRelease_WithGPU verifies ForceRelease on a real GPU-backed LazyGPUData:
// after the call bufferPtr is nil and refCount is zero regardless of how many
// times AddRef was called before. Skipped when WebGPU is unavailable.
func TestForceRelease_WithGPU(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	usage := gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc
	buf := backend.bufferPool.Acquire(256, usage)
	if buf == nil {
		t.Skip("buffer pool returned nil — GPU may be unavailable")
	}

	l := NewLazyGPUData(unsafe.Pointer(buf), buf, 256, backend)
	if l.RefCount() != 1 {
		t.Fatalf("initial refCount = %d, want 1", l.RefCount())
	}

	// Simulate two Clone() calls: each increments refCount via AddRef.
	l.AddRef()
	l.AddRef()
	if l.RefCount() != 3 {
		t.Fatalf("after 2 AddRef, refCount = %d, want 3", l.RefCount())
	}

	// ForceRelease must free the GPU buffer and zero the refcount.
	l.ForceRelease()

	if l.BufferPtr() != nil {
		t.Error("ForceRelease: bufferPtr should be nil after force release")
	}
	if l.RefCount() != 0 {
		t.Errorf("ForceRelease: refCount = %d, want 0", l.RefCount())
	}
}

// TestReclaimMemory_ReleasesClonedTensors is the end-to-end regression test for
// the use-after-free bug: a GPU tensor is cloned (AddRef increments refCount to 2),
// then ReclaimMemory is called. The old code skipped tensors with RefCount() > 1,
// keeping the GPU buffer alive and leaking memory. The fix uses ForceRelease which
// frees unconditionally regardless of refCount.
func TestReclaimMemory_ReleasesClonedTensors(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}

	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	usage := gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc
	buf := backend.bufferPool.Acquire(256, usage)
	if buf == nil {
		t.Skip("buffer pool returned nil — GPU may be unavailable")
	}

	l := NewLazyGPUData(unsafe.Pointer(buf), buf, 256, backend)
	// Simulate one Clone(): refCount becomes 2.
	l.AddRef()

	if l.RefCount() != 2 {
		t.Fatalf("after AddRef, refCount = %d, want 2", l.RefCount())
	}

	// ReclaimMemory must release this tensor even though refCount == 2.
	backend.ReclaimMemory()

	if l.BufferPtr() != nil {
		t.Error("ReclaimMemory: bufferPtr should be nil after reclaim (cloned tensors must not survive)")
	}
	if l.RefCount() != 0 {
		t.Errorf("ReclaimMemory: refCount = %d, want 0", l.RefCount())
	}
}
