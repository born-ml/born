package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// MaxPool2D performs 2D max pooling.
//
// Max pooling reduces spatial dimensions by taking the maximum value
// in each pooling window. Unlike Conv2D, MaxPool2D has no learnable parameters.
//
// Input shape:  [batch, channels, height, width]
// Output shape: [batch, channels, out_height, out_width]
//
// Where:
//
//	out_height = (height - kernelSize) / stride + 1
//	out_width = (width - kernelSize) / stride + 1
//
// Algorithm:
//  1. For each batch and channel
//  2. Slide kernelSize x kernelSize window with given stride
//  3. Take maximum value in each window
//  4. Output max value
//
// Example (2x2 pool, stride=2):
//
//	Input: [[1,2,3,4],    Output: [[4,6],
//	        [5,6,7,8],             [12,14]]
//	        [9,10,11,12],
//	        [13,14,15,16]]
func (cpu *CPUBackend) MaxPool2D(input *tensor.RawTensor, kernelSize, stride int) *tensor.RawTensor {
	// Validate input
	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("maxpool2d: expected 4D input [N,C,H,W], got %dD", len(inputShape)))
	}

	// Extract dimensions
	N := inputShape[0] // batch size
	C := inputShape[1] // channels
	H := inputShape[2] // height
	W := inputShape[3] // width

	// Validate kernel and stride
	if kernelSize <= 0 {
		panic(fmt.Sprintf("maxpool2d: invalid kernel size %d", kernelSize))
	}
	if stride <= 0 {
		panic(fmt.Sprintf("maxpool2d: invalid stride %d", stride))
	}
	if kernelSize > H || kernelSize > W {
		panic(fmt.Sprintf("maxpool2d: kernel size %d too large for input %dx%d", kernelSize, H, W))
	}

	// Compute output dimensions
	HOut := (H-kernelSize)/stride + 1
	WOut := (W-kernelSize)/stride + 1

	if HOut <= 0 || WOut <= 0 {
		panic(fmt.Sprintf("maxpool2d: invalid output dimensions %dx%d (kernel=%d, stride=%d, input=%dx%d)",
			HOut, WOut, kernelSize, stride, H, W))
	}

	// Create output tensor
	outputShape := tensor.Shape{N, C, HOut, WOut}
	output, err := tensor.NewRaw(outputShape, input.DType(), cpu.Device())
	if err != nil {
		panic(fmt.Sprintf("maxpool2d: failed to create output: %v", err))
	}

	// Dispatch to type-specific implementation
	switch input.DType() {
	case tensor.Float32:
		maxpool2dFloat32(output, input, N, C, H, W, HOut, WOut, kernelSize, stride)
	case tensor.Float64:
		maxpool2dFloat64(output, input, N, C, H, W, HOut, WOut, kernelSize, stride)
	default:
		panic(fmt.Sprintf("maxpool2d: unsupported dtype %v", input.DType()))
	}

	return output
}

// maxpool2dFloat32 performs max pooling for float32 tensors.
func maxpool2dFloat32(output, input *tensor.RawTensor, N, C, H, W, HOut, WOut, kernelSize, stride int) {
	inputData := input.AsFloat32()
	outputData := output.AsFloat32()

	// For each batch
	for n := 0; n < N; n++ {
		// For each channel
		for c := 0; c < C; c++ {
			// Pre-slice channel plane: eliminates (n*C+c)*H*W bounds check
			channelOffset := (n*C + c) * H * W
			channelData := inputData[channelOffset : channelOffset+H*W]

			// For each output position
			for outH := 0; outH < HOut; outH++ {
				// Compute pooling window start positions
				hStart := outH * stride

				for outW := 0; outW < WOut; outW++ {
					wStart := outW * stride

					// Find max value in pooling window
					maxVal := float32(-1e38) // Negative infinity approximation

					for kh := 0; kh < kernelSize; kh++ {
						h := hStart + kh
						// Pre-slice row: eliminates h*W bounds check
						rowStart := h * W
						rowData := channelData[rowStart : rowStart+W]

						for kw := 0; kw < kernelSize; kw++ {
							w := wStart + kw
							// Single bounds check via pre-slice
							val := rowData[w]

							if val > maxVal {
								maxVal = val
							}
						}
					}

					// Store max value
					outputIdx := ((n*C+c)*HOut+outH)*WOut + outW
					outputData[outputIdx] = maxVal
				}
			}
		}
	}
}

// maxpool2dFloat64 performs max pooling for float64 tensors.
func maxpool2dFloat64(output, input *tensor.RawTensor, N, C, H, W, HOut, WOut, kernelSize, stride int) {
	inputData := input.AsFloat64()
	outputData := output.AsFloat64()

	// For each batch
	for n := 0; n < N; n++ {
		// For each channel
		for c := 0; c < C; c++ {
			// Pre-slice channel plane: eliminates (n*C+c)*H*W bounds check
			channelOffset := (n*C + c) * H * W
			channelData := inputData[channelOffset : channelOffset+H*W]

			// For each output position
			for outH := 0; outH < HOut; outH++ {
				// Compute pooling window start positions
				hStart := outH * stride

				for outW := 0; outW < WOut; outW++ {
					wStart := outW * stride

					// Find max value in pooling window
					maxVal := float64(-1e308) // Negative infinity approximation

					for kh := 0; kh < kernelSize; kh++ {
						h := hStart + kh
						// Pre-slice row: eliminates h*W bounds check
						rowStart := h * W
						rowData := channelData[rowStart : rowStart+W]

						for kw := 0; kw < kernelSize; kw++ {
							w := wStart + kw
							// Single bounds check via pre-slice
							val := rowData[w]

							if val > maxVal {
								maxVal = val
							}
						}
					}

					// Store max value
					outputIdx := ((n*C+c)*HOut+outH)*WOut + outW
					outputData[outputIdx] = maxVal
				}
			}
		}
	}
}
