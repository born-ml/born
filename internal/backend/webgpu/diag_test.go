//go:build windows

package webgpu

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
)

// TestDiagLazyAdd isolates each step of the lazy Add path to find where zeros are introduced.
//
//nolint:gocognit // Diagnostic test with multiple sequential subtests — splitting would lose isolation context
func TestDiagLazyAdd(t *testing.T) {
	if !IsAvailable() {
		t.Skip("WebGPU not available")
	}

	backend, err := New()
	if err != nil {
		t.Fatalf("failed to create backend: %v", err)
	}
	defer backend.Release()

	a := createTensor(t, tensor.Shape{4}, []float32{1, 2, 3, 4})
	b2 := createTensor(t, tensor.Shape{4}, []float32{5, 6, 7, 8})

	// Step 1: Check createBuffer uploads data correctly
	t.Run("step1_createBuffer", func(t *testing.T) {
		bufA := backend.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
		defer bufA.Release()
		data := diagReadBufferDirect(t, backend, bufA, uint64(a.ByteSize()))
		vals := diagAsFloat32(data)
		t.Logf("Input buffer A data: %v", vals)
		if vals[0] != 1.0 {
			t.Errorf("Input buffer not uploaded correctly: expected 1.0, got %v", vals[0])
		} else {
			t.Logf("Input buffer upload: PASS")
		}
	})

	// Step 2: Manual unified pattern with direct submit
	t.Run("step2_manual_unified_direct_submit", func(t *testing.T) {
		numElements := a.NumElements()
		resultSize := uint64(a.ByteSize())

		shaderName := "add"
		shader := backend.compileShader(shaderName, addShader)
		entry := backend.getOrCreatePipeline(shaderName, shader, bglBinary)

		bufferA := backend.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
		defer bufferA.Release()
		bufferB := backend.createBuffer(b2.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
		defer bufferB.Release()

		bufferResult, createErr := backend.device.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
			Size:  resultSize,
		})
		if createErr != nil {
			t.Fatalf("create result buffer: %v", createErr)
		}
		defer bufferResult.Release()

		stagingBuf, createErr := backend.device.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst,
			Size:  resultSize,
		})
		if createErr != nil {
			t.Fatalf("create staging buffer: %v", createErr)
		}
		defer stagingBuf.Release()

		params := backend.createParamsBuffer(numElements)
		defer params.Release()

		bg := backend.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
			bufBinding(bufferA, resultSize),
			bufBinding(bufferB, resultSize),
			bufBinding(bufferResult, resultSize),
			bufBinding(params, 16),
		})
		defer bg.Release()

		encoder, encErr := backend.device.CreateCommandEncoder(nil)
		if encErr != nil {
			t.Fatalf("create encoder: %v", encErr)
		}
		cp, cpErr := encoder.BeginComputePass(nil)
		if cpErr != nil {
			t.Fatalf("BeginComputePass: %v", cpErr)
		}
		cp.SetPipeline(entry.pipeline)
		cp.SetBindGroup(0, bg, nil)
		workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
		cp.Dispatch(workgroups, 1, 1)
		if endErr := cp.End(); endErr != nil {
			t.Fatalf("ComputePass.End: %v", endErr)
		}
		encoder.CopyBufferToBuffer(bufferResult, 0, stagingBuf, 0, resultSize)
		cmdBuf, finErr := encoder.Finish()
		if finErr != nil {
			t.Fatalf("encoder.Finish: %v", finErr)
		}

		if _, submitErr := backend.queue.Submit(cmdBuf); submitErr != nil {
			t.Fatalf("Submit error: %v", submitErr)
		}

		if mapErr := stagingBuf.Map(context.Background(), wgpu.MapModeRead, 0, resultSize); mapErr != nil {
			t.Fatalf("Map error: %v", mapErr)
		}
		defer func() { _ = stagingBuf.Unmap() }()

		rng, rngErr := stagingBuf.MappedRange(0, resultSize)
		if rngErr != nil {
			t.Fatalf("MappedRange error: %v", rngErr)
		}
		defer rng.Release()

		result := diagAsFloat32(rng.Bytes())
		t.Logf("Manual unified (direct submit) result: %v", result)
		if result[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — MANUAL UNIFIED FAILS", result[0])
		} else {
			t.Logf("Manual unified (direct submit): PASS")
		}
	})

	// Step 3: Batched submit (queueCommand + flushCommands) + manual Map
	t.Run("step3_batched_submit_manual_map", func(t *testing.T) {
		numElements := a.NumElements()
		resultSize := uint64(a.ByteSize())

		shaderName := "add"
		shader := backend.compileShader(shaderName, addShader)
		entry := backend.getOrCreatePipeline(shaderName, shader, bglBinary)

		bufferA := backend.createBuffer(a.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
		defer bufferA.Release()
		bufferB := backend.createBuffer(b2.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
		defer bufferB.Release()

		bufferResult, createErr := backend.device.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc,
			Size:  resultSize,
		})
		if createErr != nil {
			t.Fatalf("create result buffer: %v", createErr)
		}
		defer bufferResult.Release()

		stagingBuf, createErr := backend.device.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst,
			Size:  resultSize,
		})
		if createErr != nil {
			t.Fatalf("create staging buffer: %v", createErr)
		}
		defer stagingBuf.Release()

		params := backend.createParamsBuffer(numElements)
		defer params.Release()

		bg := backend.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
			bufBinding(bufferA, resultSize),
			bufBinding(bufferB, resultSize),
			bufBinding(bufferResult, resultSize),
			bufBinding(params, 16),
		})
		defer bg.Release()

		encoder, encErr := backend.device.CreateCommandEncoder(nil)
		if encErr != nil {
			t.Fatalf("create encoder: %v", encErr)
		}
		cp, cpErr := encoder.BeginComputePass(nil)
		if cpErr != nil {
			t.Fatalf("BeginComputePass: %v", cpErr)
		}
		cp.SetPipeline(entry.pipeline)
		cp.SetBindGroup(0, bg, nil)
		workgroups := uint32((numElements + workgroupSize - 1) / workgroupSize)
		cp.Dispatch(workgroups, 1, 1)
		if endErr := cp.End(); endErr != nil {
			t.Fatalf("ComputePass.End: %v", endErr)
		}
		encoder.CopyBufferToBuffer(bufferResult, 0, stagingBuf, 0, resultSize)
		cmdBuf, finErr := encoder.Finish()
		if finErr != nil {
			t.Fatalf("encoder.Finish: %v", finErr)
		}

		backend.queueCommand(cmdBuf)
		backend.flushCommands()
		t.Logf("Batched submit flushed")

		if mapErr := stagingBuf.Map(context.Background(), wgpu.MapModeRead, 0, resultSize); mapErr != nil {
			t.Fatalf("Map error after batch flush: %v", mapErr)
		}
		defer func() { _ = stagingBuf.Unmap() }()

		rng, rngErr := stagingBuf.MappedRange(0, resultSize)
		if rngErr != nil {
			t.Fatalf("MappedRange error: %v", rngErr)
		}
		defer rng.Release()

		result := diagAsFloat32(rng.Bytes())
		t.Logf("Batched submit + manual Map result: %v", result)
		if result[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — BATCH SUBMIT FAILS", result[0])
		} else {
			t.Logf("Batched submit + manual Map: PASS")
		}
	})

		// Step 4: Full lazy Add via Born's runBinaryOpLazy, then direct Map of staging buf
	t.Run("step4_lazy_add_direct_map", func(t *testing.T) {
		result, runErr := backend.runBinaryOpLazy(a, b2, "add", addShader)
		if runErr != nil {
			t.Fatalf("runBinaryOpLazy error: %v", runErr)
		}

		gpuData := result.GPUData()
		if gpuData == nil {
			t.Fatal("No GPU data on lazy result")
		}

		resultSize := gpuData.Size()
		stagingBuf := (*wgpu.Buffer)(gpuData.BufferPtr())
		t.Logf("Staging buf ptr: %p, size: %d", unsafe.Pointer(stagingBuf), resultSize)

		// Flush then Map directly
		backend.flushCommands()
		t.Logf("Commands flushed")

		if mapErr := stagingBuf.Map(context.Background(), wgpu.MapModeRead, 0, resultSize); mapErr != nil {
			t.Fatalf("Direct Map of staging buf error: %v", mapErr)
		}
		defer func() { _ = stagingBuf.Unmap() }()

		rng, rngErr := stagingBuf.MappedRange(0, resultSize)
		if rngErr != nil {
			t.Fatalf("MappedRange error: %v", rngErr)
		}
		defer rng.Release()

		vals := diagAsFloat32(rng.Bytes())
		t.Logf("Direct Map of lazy staging buf: %v", vals)
		if vals[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — LAZY STAGING BUF HAS WRONG DATA", vals[0])
		} else {
			t.Logf("Direct Map of lazy staging buf: PASS")
		}
	})

	// Step 5: Full lazy Add via runBinaryOpLazy, then call ReadGPUBuffer directly
	t.Run("step5_lazy_add_ReadGPUBuffer", func(t *testing.T) {
		result, runErr := backend.runBinaryOpLazy(a, b2, "add", addShader)
		if runErr != nil {
			t.Fatalf("runBinaryOpLazy error: %v", runErr)
		}

		gpuData := result.GPUData()
		if gpuData == nil {
			t.Fatal("No GPU data on lazy result")
		}

		t.Logf("Before ReadGPUBuffer: pending commands = %d", len(backend.pendingCommands))
		t.Logf("Buffer ptr = %p, size = %d", gpuData.BufferPtr(), gpuData.Size())

		// Check submit directly (drain pending first so ReadGPUBuffer sees empty pending)
		backend.pendingMu.Lock()
		cmds := backend.pendingCommands
		backend.pendingCommands = backend.pendingCommands[:0]
		backend.pendingMu.Unlock()
		if len(cmds) > 0 {
			subIdx, submitErr := backend.queue.Submit(cmds...)
			t.Logf("Direct submit result: subIdx=%d, err=%v", subIdx, submitErr)
		}

		// Poll wait to ensure GPU done
		backend.device.Poll(wgpu.PollWait)
		t.Logf("After Poll(PollWait)")

		// Now try mapping directly (bypass ReadGPUBuffer)
		stagingBuf := (*wgpu.Buffer)(gpuData.BufferPtr())
		t.Logf("Staging buf state before Map: %v", stagingBuf.MapState())
		if mapErr := stagingBuf.Map(context.Background(), wgpu.MapModeRead, 0, gpuData.Size()); mapErr != nil {
			t.Logf("Direct Map AFTER Poll: error = %v", mapErr)
		} else {
			rng, _ := stagingBuf.MappedRange(0, gpuData.Size())
			vals := diagAsFloat32(rng.Bytes())
			t.Logf("Direct Map AFTER Poll: result = %v", vals)
			rng.Release()
			_ = stagingBuf.Unmap()
		}

		data, readErr := backend.ReadGPUBuffer(gpuData.BufferPtr(), gpuData.Size())
		if readErr != nil {
			t.Fatalf("ReadGPUBuffer error: %v", readErr)
		}

		vals := diagAsFloat32(data)
		t.Logf("ReadGPUBuffer result: %v", vals)
		if vals[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — ReadGPUBuffer returns wrong data", vals[0])
		} else {
			t.Logf("ReadGPUBuffer: PASS")
		}
	})

	// Step 5b: Same as step 5 but with fresh backend (no state from earlier steps)
	t.Run("step5b_fresh_backend_ReadGPUBuffer", func(t *testing.T) {
		freshBackend, bErr := New()
		if bErr != nil {
			t.Skipf("could not create fresh backend: %v", bErr)
		}
		defer freshBackend.Release()

		result, runErr := freshBackend.runBinaryOpLazy(a, b2, "add", addShader)
		if runErr != nil {
			t.Fatalf("runBinaryOpLazy error: %v", runErr)
		}

		gpuData := result.GPUData()
		if gpuData == nil {
			t.Fatal("No GPU data on lazy result")
		}

		t.Logf("Before ReadGPUBuffer (fresh): pending commands = %d", len(freshBackend.pendingCommands))

		data, readErr := freshBackend.ReadGPUBuffer(gpuData.BufferPtr(), gpuData.Size())
		if readErr != nil {
			t.Fatalf("ReadGPUBuffer (fresh) error: %v", readErr)
		}

		vals := diagAsFloat32(data)
		t.Logf("ReadGPUBuffer (fresh backend) result: %v", vals)
		if vals[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — ReadGPUBuffer (fresh) returns wrong data", vals[0])
		} else {
			t.Logf("ReadGPUBuffer (fresh backend): PASS")
		}
	})

	// Step 6: Full lazy path through result.Data() (exactly as TestAdd does)
	t.Run("step6_result_Data", func(t *testing.T) {
		result := backend.Add(a, b2)
		if !result.IsLazy() {
			t.Fatal("Result should be lazy")
		}

		// This is exactly what extractData() does in TestAdd
		actual := diagAsFloat32(result.Data())
		t.Logf("result.Data() values: %v", actual)
		if actual[0] != 6.0 {
			t.Errorf("Expected 6.0, got %v — result.Data() fails", actual[0])
		} else {
			t.Logf("result.Data(): PASS")
		}
	})
}

// diagReadBufferDirect reads a GPU storage buffer back to CPU via a temporary staging buffer.
func diagReadBufferDirect(t *testing.T, b *Backend, srcBuf *wgpu.Buffer, size uint64) []byte {
	t.Helper()
	staging, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst,
		Size:  size,
	})
	if err != nil {
		t.Fatalf("diagReadBufferDirect: create staging: %v", err)
	}
	defer staging.Release()

	enc, _ := b.device.CreateCommandEncoder(nil)
	enc.CopyBufferToBuffer(srcBuf, 0, staging, 0, size)
	cmd, _ := enc.Finish()
	if _, submitErr := b.queue.Submit(cmd); submitErr != nil {
		t.Fatalf("diagReadBufferDirect: submit: %v", submitErr)
	}
	if mapErr := staging.Map(context.Background(), wgpu.MapModeRead, 0, size); mapErr != nil {
		t.Fatalf("diagReadBufferDirect: map: %v", mapErr)
	}
	defer func() { _ = staging.Unmap() }()
	rng, _ := staging.MappedRange(0, size)
	defer rng.Release()
	out := make([]byte, size)
	copy(out, rng.Bytes())
	return out
}

// diagAsFloat32 converts raw bytes to []float32.
func diagAsFloat32(data []byte) []float32 {
	n := len(data) / 4
	result := make([]float32, n)
	for i := range result {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result
}
