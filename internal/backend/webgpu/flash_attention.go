//go:build windows

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/born-ml/born/internal/tensor"
	"github.com/go-webgpu/webgpu/wgpu"
)

// FlashAttentionGPU executes Flash Attention 2 on GPU using WebGPU.
//
// This implementation uses tiled computation with online softmax to achieve
// O(N) memory complexity instead of O(N²) for standard attention.
//
// Parameters:
//   - q: Query tensor [batch, seqLen, numHeads, headDim]
//   - k: Key tensor [batch, kvLen, numHeads, headDim]
//   - v: Value tensor [batch, kvLen, numHeads, headDim]
//   - scale: Attention scale factor (typically 1/sqrt(headDim))
//   - causal: Whether to apply causal masking
//   - blockSize: Tile size for blocked computation (64 or 128)
//
// Returns:
//   - *tensor.RawTensor: Output tensor [batch, seqLen, numHeads, headDim]
//
//nolint:gocyclo,cyclop,funlen // High complexity/length inherent to Flash Attention setup with multiple validation checks
func (b *Backend) FlashAttentionGPU(
	q, k, v *tensor.RawTensor,
	scale float32,
	causal bool,
	blockSize int,
) (*tensor.RawTensor, error) {
	// Validate inputs
	if len(q.Shape()) != 4 || len(k.Shape()) != 4 || len(v.Shape()) != 4 {
		return nil, fmt.Errorf("FlashAttentionGPU: Q, K, V must be 4D [batch, seq, numHeads, headDim]")
	}

	if q.DType() != tensor.Float32 || k.DType() != tensor.Float32 || v.DType() != tensor.Float32 {
		return nil, fmt.Errorf("FlashAttentionGPU: only float32 is supported")
	}

	batch := q.Shape()[0]
	seqLen := q.Shape()[1]
	kvLen := k.Shape()[1]
	numHeads := q.Shape()[2]
	headDim := q.Shape()[3]

	// Validate dimensions match
	if k.Shape()[0] != batch || v.Shape()[0] != batch {
		return nil, fmt.Errorf("FlashAttentionGPU: batch size mismatch")
	}
	if k.Shape()[2] != numHeads || v.Shape()[2] != numHeads {
		return nil, fmt.Errorf("FlashAttentionGPU: numHeads mismatch")
	}
	if k.Shape()[3] != headDim || v.Shape()[3] != headDim {
		return nil, fmt.Errorf("FlashAttentionGPU: headDim mismatch")
	}

	// Validate supported head dimensions
	if headDim != 64 && headDim != 96 && headDim != 128 && headDim != 256 {
		return nil, fmt.Errorf("FlashAttentionGPU: headDim must be 64, 96, 128, or 256, got %d", headDim)
	}

	// Validate block size
	if blockSize != 64 && blockSize != 128 {
		return nil, fmt.Errorf("FlashAttentionGPU: blockSize must be 64 or 128, got %d", blockSize)
	}

	// Compile shader
	shader := b.compileShader("flash_attention", flashAttentionShader)
	pipeline := b.getOrCreatePipeline("flash_attention", shader)

	// Create GPU buffers
	bufferQ := b.createBuffer(q.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferQ.Release()

	bufferK := b.createBuffer(k.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferK.Release()

	bufferV := b.createBuffer(v.Data(), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	defer bufferV.Release()

	outputSize := uint64(q.ByteSize()) //nolint:gosec // G115: Buffer size fits in uint64 for GPU operations
	bufferOutput := b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		Size:  outputSize,
	})
	defer bufferOutput.Release()

	// Create uniform buffer for params
	// struct Params { batch, seq_len, kv_len, num_heads, head_dim, block_size, scale (f32), causal }
	params := make([]byte, 32) // 8 u32 fields = 32 bytes
	//nolint:gosec // G115: Dimensions validated above - safe conversions from validated int to uint32
	binary.LittleEndian.PutUint32(params[0:4], uint32(batch))
	//nolint:gosec // G115: Dimensions validated above
	binary.LittleEndian.PutUint32(params[4:8], uint32(seqLen))
	//nolint:gosec // G115: Dimensions validated above
	binary.LittleEndian.PutUint32(params[8:12], uint32(kvLen))
	//nolint:gosec // G115: Dimensions validated above
	binary.LittleEndian.PutUint32(params[12:16], uint32(numHeads))
	binary.LittleEndian.PutUint32(params[16:20], uint32(headDim))
	binary.LittleEndian.PutUint32(params[20:24], uint32(blockSize))
	binary.LittleEndian.PutUint32(params[24:28], math.Float32bits(scale))

	causalU32 := uint32(0)
	if causal {
		causalU32 = 1
	}
	binary.LittleEndian.PutUint32(params[28:32], causalU32)

	bufferParams := b.createUniformBuffer(params)
	defer bufferParams.Release()

	// Create bind group
	bindGroupLayout := pipeline.GetBindGroupLayout(0)
	bindGroup := b.device.CreateBindGroupSimple(bindGroupLayout, []wgpu.BindGroupEntry{
		wgpu.BufferBindingEntry(0, bufferQ, 0, outputSize),
		wgpu.BufferBindingEntry(1, bufferK, 0, uint64(k.ByteSize())), //nolint:gosec // G115: Buffer size fits
		wgpu.BufferBindingEntry(2, bufferV, 0, uint64(v.ByteSize())), //nolint:gosec // G115: Buffer size fits
		wgpu.BufferBindingEntry(3, bufferOutput, 0, outputSize),
		wgpu.BufferBindingEntry(4, bufferParams, 0, 32),
	})
	defer bindGroup.Release()

	// Dispatch compute shader
	encoder := b.device.CreateCommandEncoder(nil)

	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)

	// Workgroup dispatch: (num_q_blocks, num_heads, batch)
	numQBlocks := (seqLen + blockSize - 1) / blockSize
	computePass.DispatchWorkgroups(uint32(numQBlocks), uint32(numHeads), uint32(batch)) //nolint:gosec // G115: Safe cast
	computePass.End()

	cmdBuffer := encoder.Finish(nil)
	b.queue.Submit(cmdBuffer)

	// Read result
	resultData, err := b.readBuffer(bufferOutput, outputSize)
	if err != nil {
		return nil, fmt.Errorf("FlashAttentionGPU: failed to read output: %w", err)
	}

	// Create result tensor
	result, err := tensor.NewRaw(q.Shape(), tensor.Float32, tensor.WebGPU)
	if err != nil {
		return nil, fmt.Errorf("FlashAttentionGPU: failed to create result tensor: %w", err)
	}

	copy(result.Data(), resultData)
	return result, nil
}
