package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// Conv2D performs 2D convolution using im2col algorithm.
//
// Input shape: [batch, in_channels, height, width]
// Kernel shape: [out_channels, in_channels, kernel_h, kernel_w]
// Output shape: [batch, out_channels, out_h, out_w]
//
// Parameters:
//   - input: Input tensor [N, C_in, H, W]
//   - kernel: Convolution kernel [C_out, C_in, K_h, K_w]
//   - stride: Stride for convolution (default: 1)
//   - padding: Padding to apply (default: 0)
//
// Algorithm: Im2col
//  1. Transform input patches into columns (im2col)
//  2. Reshape kernel into matrix
//  3. Perform matrix multiplication
//  4. Reshape output to [N, C_out, H_out, W_out]
//
// Im2col is efficient because:
//   - Converts convolution to matmul (highly optimized)
//   - Cache-friendly memory access
//   - Reuses existing matmul code
//
// Reference: "High Performance Convolutional Neural Networks for Document Processing"
// (Chellapilla et al., 2006).
func (cpu *CPUBackend) Conv2D(input, kernel *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	// Validate input shapes
	inputShape := input.Shape()
	kernelShape := kernel.Shape()

	if len(inputShape) != 4 {
		panic(fmt.Sprintf("conv2d: input must be 4D [N,C,H,W], got %dD", len(inputShape)))
	}
	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("conv2d: kernel must be 4D [C_out,C_in,K_h,K_w], got %dD", len(kernelShape)))
	}

	N := inputShape[0]     // batch size
	CIn := inputShape[1]   // input channels
	H := inputShape[2]     // input height
	W := inputShape[3]     // input width
	COut := kernelShape[0] // output channels
	CInK := kernelShape[1] // kernel input channels (must match CIn)
	KH := kernelShape[2]   // kernel height
	KW := kernelShape[3]   // kernel width

	// Validate channel dimensions
	if CIn != CInK {
		panic(fmt.Sprintf("conv2d: input channels %d != kernel channels %d", CIn, CInK))
	}

	// Compute output dimensions
	// out_h = (H + 2*padding - KH) / stride + 1
	// out_w = (W + 2*padding - KW) / stride + 1
	HOut := (H+2*padding-KH)/stride + 1
	WOut := (W+2*padding-KW)/stride + 1

	if HOut <= 0 || WOut <= 0 {
		panic(fmt.Sprintf("conv2d: invalid output dimensions: out_h=%d, out_w=%d (check stride/padding)", HOut, WOut))
	}

	// Create output tensor [N, C_out, H_out, W_out]
	output, err := tensor.NewRaw(tensor.Shape{N, COut, HOut, WOut}, input.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("conv2d: failed to create output tensor: %v", err))
	}

	// Dispatch to type-specific implementation
	switch input.DType() {
	case tensor.Float32:
		conv2dFloat32(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding)
	case tensor.Float64:
		conv2dFloat64(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding)
	default:
		panic(fmt.Sprintf("conv2d: unsupported dtype %s", input.DType()))
	}

	return output
}

// conv2dFloat32 performs Conv2D for float32 using im2col.
//
// Algorithm:
//  1. Im2col: Transform [N, C, H, W] -> [N * H_out * W_out, C * K_h * K_w]
//  2. Reshape kernel: [C_out, C, K_h, K_w] -> [C_out, C * K_h * K_w]
//  3. MatMul: [C_out, C*K_h*K_w] @ [C*K_h*K_w, N*H_out*W_out] -> [C_out, N*H_out*W_out]
//  4. Reshape: [C_out, N*H_out*W_out] -> [N, C_out, H_out, W_out]
//
// Stride specialization: separate path for stride=1 enables compiler optimizations (SIMD).
func conv2dFloat32(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int) {
	// Dispatch to specialized implementation for common case
	if stride == 1 && padding == 0 {
		conv2dFloat32Stride1NoPad(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut)
		return
	}

	// General case (stride > 1 or padding > 0)
	conv2dFloat32General(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding)
}

// conv2dFloat32Stride1NoPad is optimized for stride=1, padding=0 (most common case).
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
func conv2dFloat32Stride1NoPad(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut int) {
	inputData := input.AsFloat32()
	kernelData := kernel.AsFloat32()
	outputData := output.AsFloat32()

	// Step 1: Im2col with stride=1, padding=0
	colWidth := CIn * KH * KW
	colHeight := N * HOut * WOut
	colBuf := make([]float32, colHeight*colWidth)

	im2colFloat32Stride1NoPad(colBuf, inputData, N, CIn, H, W, KH, KW, HOut, WOut)

	// Step 2: Matrix multiplication
	for i := 0; i < COut; i++ {
		for j := 0; j < colHeight; j++ {
			sum := float32(0.0)
			for k := 0; k < colWidth; k++ {
				sum += kernelData[i*colWidth+k] * colBuf[j*colWidth+k]
			}
			outputData[i*colHeight+j] = sum
		}
	}

	// Step 3: Rearrange from [C_out, N*H_out*W_out] to [N, C_out, H_out, W_out]
	tempBuf := make([]float32, len(outputData))
	copy(tempBuf, outputData)

	for n := 0; n < N; n++ {
		for c := 0; c < COut; c++ {
			for h := 0; h < HOut; h++ {
				for w := 0; w < WOut; w++ {
					srcIdx := c*colHeight + n*HOut*WOut + h*WOut + w
					dstIdx := n*COut*HOut*WOut + c*HOut*WOut + h*WOut + w
					outputData[dstIdx] = tempBuf[srcIdx]
				}
			}
		}
	}
}

// conv2dFloat32General handles arbitrary stride and padding.
func conv2dFloat32General(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int) {
	inputData := input.AsFloat32()
	kernelData := kernel.AsFloat32()
	outputData := output.AsFloat32()

	// Step 1: Im2col transformation
	// colBuf: [N * H_out * W_out, C_in * K_h * K_w]
	colWidth := CIn * KH * KW
	colHeight := N * HOut * WOut
	colBuf := make([]float32, colHeight*colWidth)

	im2colFloat32(colBuf, inputData, N, CIn, H, W, KH, KW, HOut, WOut, stride, padding)

	// Step 2: Reshape kernel
	// kernelData is already in [C_out, C_in * K_h * K_w] layout (row-major)

	// Step 3: Matrix multiplication
	// kernel: [C_out, C_in * K_h * K_w]
	// colBuf: [C_in * K_h * K_w, N * H_out * W_out] (transposed view)
	// result: [C_out, N * H_out * W_out]
	//
	// We want: result[i, j] = sum_k kernel[i, k] * colBuf[j, k]
	// But colBuf is in row-major as [N*H_out*W_out, C*K_h*K_w]
	// So we compute: result[i, j] = sum_k kernel[i, k] * colBuf[j*colWidth + k]

	for i := 0; i < COut; i++ {
		for j := 0; j < colHeight; j++ {
			sum := float32(0.0)
			for k := 0; k < colWidth; k++ {
				sum += kernelData[i*colWidth+k] * colBuf[j*colWidth+k]
			}
			// Temporary storage in row-major: [C_out, N*H_out*W_out]
			// We'll rearrange this into [N, C_out, H_out, W_out] next
			outputData[i*colHeight+j] = sum
		}
	}

	// Step 4: Rearrange from [C_out, N*H_out*W_out] to [N, C_out, H_out, W_out]
	// Current layout: output[c, n*H_out*W_out + h*W_out + w]
	// Desired layout: output[n, c, h, w] = output[n*C_out*H_out*W_out + c*H_out*W_out + h*W_out + w]
	tempBuf := make([]float32, len(outputData))
	copy(tempBuf, outputData)

	for n := 0; n < N; n++ {
		for c := 0; c < COut; c++ {
			for h := 0; h < HOut; h++ {
				for w := 0; w < WOut; w++ {
					// Source index: [c, n*H_out*W_out + h*W_out + w]
					srcIdx := c*colHeight + n*HOut*WOut + h*WOut + w
					// Dest index: [n, c, h, w]
					dstIdx := n*COut*HOut*WOut + c*HOut*WOut + h*WOut + w
					outputData[dstIdx] = tempBuf[srcIdx]
				}
			}
		}
	}
}

// im2colFloat32Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize with hardcoded stride=1 (no bounds checks for padding).
func im2colFloat32Stride1NoPad(colBuf, inputData []float32, N, C, H, W, KH, KW, HOut, WOut int) {
	colWidth := C * KH * KW
	colIdx := 0

	for n := 0; n < N; n++ {
		batchOffset := n * C * H * W
		batchData := inputData[batchOffset : batchOffset+C*H*W]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				// With stride=1, padding=0: hStart = outH, wStart = outW
				rowOffset := colIdx * colWidth
				rowData := colBuf[rowOffset : rowOffset+colWidth]

				bufIdx := 0
				for c := 0; c < C; c++ {
					channelOffset := c * H * W
					channelData := batchData[channelOffset : channelOffset+H*W]

					for kh := 0; kh < KH; kh++ {
						h := outH + kh // stride=1: no multiplication
						for kw := 0; kw < KW; kw++ {
							w := outW + kw // stride=1: no multiplication
							// No padding check needed (padding=0 guaranteed)
							rowData[bufIdx] = channelData[h*W+w]
							bufIdx++
						}
					}
				}

				colIdx++
			}
		}
	}
}

// im2colFloat32 transforms input tensor into column matrix (general case).
//
// Input: [N, C, H, W]
// Output: colBuf [N * H_out * W_out, C * K_h * K_w]
//
// Each row of colBuf corresponds to one output position.
// Each column corresponds to one kernel weight.
//
// For each output position (n, out_h, out_w):
//   - Extract the patch from input
//   - Flatten the patch into a row of colBuf
func im2colFloat32(colBuf, inputData []float32, N, C, H, W, KH, KW, HOut, WOut, stride, padding int) {
	colWidth := C * KH * KW
	colIdx := 0 // Current row in colBuf

	for n := 0; n < N; n++ {
		// Pre-slice batch: eliminates n*C*H*W bounds check
		batchOffset := n * C * H * W
		batchData := inputData[batchOffset : batchOffset+C*H*W]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				// For this output position, extract the input patch
				// Top-left corner in input space
				hStart := outH*stride - padding
				wStart := outW*stride - padding

				// Pre-slice output row: eliminates colIdx*colWidth bounds check
				rowOffset := colIdx * colWidth
				rowData := colBuf[rowOffset : rowOffset+colWidth]

				bufIdx := 0
				for c := 0; c < C; c++ {
					// Pre-slice channel: eliminates c*H*W bounds check
					channelOffset := c * H * W
					channelData := batchData[channelOffset : channelOffset+H*W]

					for kh := 0; kh < KH; kh++ {
						h := hStart + kh
						for kw := 0; kw < KW; kw++ {
							w := wStart + kw

							// Check bounds (padding)
							if h >= 0 && h < H && w >= 0 && w < W {
								// Valid input position: single bounds check via pre-slice
								rowData[bufIdx] = channelData[h*W+w]
							} else {
								// Out of bounds (padding with zero)
								rowData[bufIdx] = 0.0
							}
							bufIdx++
						}
					}
				}

				colIdx++
			}
		}
	}
}

// conv2dFloat64 performs Conv2D for float64 using im2col.
// Stride specialization: separate path for stride=1 enables compiler optimizations (SIMD).
func conv2dFloat64(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int) {
	// Dispatch to specialized implementation for common case
	if stride == 1 && padding == 0 {
		conv2dFloat64Stride1NoPad(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut)
		return
	}

	// General case (stride > 1 or padding > 0)
	conv2dFloat64General(output, input, kernel, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding)
}

// conv2dFloat64Stride1NoPad is optimized for stride=1, padding=0 (most common case).
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
func conv2dFloat64Stride1NoPad(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut int) {
	inputData := input.AsFloat64()
	kernelData := kernel.AsFloat64()
	outputData := output.AsFloat64()

	// Im2col with stride=1, padding=0
	colWidth := CIn * KH * KW
	colHeight := N * HOut * WOut
	colBuf := make([]float64, colHeight*colWidth)
	im2colFloat64Stride1NoPad(colBuf, inputData, N, CIn, H, W, KH, KW, HOut, WOut)

	// MatMul
	for i := 0; i < COut; i++ {
		for j := 0; j < colHeight; j++ {
			sum := float64(0.0)
			for k := 0; k < colWidth; k++ {
				sum += kernelData[i*colWidth+k] * colBuf[j*colWidth+k]
			}
			outputData[i*colHeight+j] = sum
		}
	}

	// Rearrange
	tempBuf := make([]float64, len(outputData))
	copy(tempBuf, outputData)
	for n := 0; n < N; n++ {
		for c := 0; c < COut; c++ {
			for h := 0; h < HOut; h++ {
				for w := 0; w < WOut; w++ {
					srcIdx := c*colHeight + n*HOut*WOut + h*WOut + w
					dstIdx := n*COut*HOut*WOut + c*HOut*WOut + h*WOut + w
					outputData[dstIdx] = tempBuf[srcIdx]
				}
			}
		}
	}
}

// conv2dFloat64General handles arbitrary stride and padding.
func conv2dFloat64General(output, input, kernel *tensor.RawTensor, N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int) {
	inputData := input.AsFloat64()
	kernelData := kernel.AsFloat64()
	outputData := output.AsFloat64()

	// Im2col
	colWidth := CIn * KH * KW
	colHeight := N * HOut * WOut
	colBuf := make([]float64, colHeight*colWidth)
	im2colFloat64(colBuf, inputData, N, CIn, H, W, KH, KW, HOut, WOut, stride, padding)

	// MatMul
	for i := 0; i < COut; i++ {
		for j := 0; j < colHeight; j++ {
			sum := float64(0.0)
			for k := 0; k < colWidth; k++ {
				sum += kernelData[i*colWidth+k] * colBuf[j*colWidth+k]
			}
			outputData[i*colHeight+j] = sum
		}
	}

	// Rearrange
	tempBuf := make([]float64, len(outputData))
	copy(tempBuf, outputData)
	for n := 0; n < N; n++ {
		for c := 0; c < COut; c++ {
			for h := 0; h < HOut; h++ {
				for w := 0; w < WOut; w++ {
					srcIdx := c*colHeight + n*HOut*WOut + h*WOut + w
					dstIdx := n*COut*HOut*WOut + c*HOut*WOut + h*WOut + w
					outputData[dstIdx] = tempBuf[srcIdx]
				}
			}
		}
	}
}

// im2colFloat64Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize with hardcoded stride=1 (no bounds checks for padding).
func im2colFloat64Stride1NoPad(colBuf, inputData []float64, N, C, H, W, KH, KW, HOut, WOut int) {
	colWidth := C * KH * KW
	colIdx := 0

	for n := 0; n < N; n++ {
		batchOffset := n * C * H * W
		batchData := inputData[batchOffset : batchOffset+C*H*W]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				// With stride=1, padding=0: hStart = outH, wStart = outW
				rowOffset := colIdx * colWidth
				rowData := colBuf[rowOffset : rowOffset+colWidth]

				bufIdx := 0
				for c := 0; c < C; c++ {
					channelOffset := c * H * W
					channelData := batchData[channelOffset : channelOffset+H*W]

					for kh := 0; kh < KH; kh++ {
						h := outH + kh // stride=1: no multiplication
						for kw := 0; kw < KW; kw++ {
							w := outW + kw // stride=1: no multiplication
							// No padding check needed (padding=0 guaranteed)
							rowData[bufIdx] = channelData[h*W+w]
							bufIdx++
						}
					}
				}

				colIdx++
			}
		}
	}
}

func im2colFloat64(colBuf, inputData []float64, N, C, H, W, KH, KW, HOut, WOut, stride, padding int) {
	colWidth := C * KH * KW
	colIdx := 0

	for n := 0; n < N; n++ {
		// Pre-slice batch: eliminates n*C*H*W bounds check
		batchOffset := n * C * H * W
		batchData := inputData[batchOffset : batchOffset+C*H*W]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				hStart := outH*stride - padding
				wStart := outW*stride - padding

				// Pre-slice output row: eliminates colIdx*colWidth bounds check
				rowOffset := colIdx * colWidth
				rowData := colBuf[rowOffset : rowOffset+colWidth]

				bufIdx := 0
				for c := 0; c < C; c++ {
					// Pre-slice channel: eliminates c*H*W bounds check
					channelOffset := c * H * W
					channelData := batchData[channelOffset : channelOffset+H*W]

					for kh := 0; kh < KH; kh++ {
						h := hStart + kh
						for kw := 0; kw < KW; kw++ {
							w := wStart + kw

							if h >= 0 && h < H && w >= 0 && w < W {
								// Valid input position: single bounds check via pre-slice
								rowData[bufIdx] = channelData[h*W+w]
							} else {
								rowData[bufIdx] = 0.0
							}
							bufIdx++
						}
					}
				}
				colIdx++
			}
		}
	}
}
