package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// Conv2DInputBackward computes gradient w.r.t. input using transposed convolution.
//
// Algorithm: Transposed convolution (full convolution).
//   - For each input position (n, c_in, h, w):
//   - Sum contributions from all output positions that used this input
//   - Each contribution is: grad[n, c_out, h_out, w_out] * kernel[c_out, c_in, kh, kw]
//
// References:
//   - Burn framework: crates/burn-autodiff/src/ops/module.rs (conv2d_x_backward)
//   - "A guide to convolution arithmetic for deep learning" (Dumoulin & Visin, 2016)
//
//nolint:dupl // Intentional duplication with Conv2DKernelBackward (different operations)
func (cpu *CPUBackend) Conv2DInputBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	// Extract shapes
	inputShape := input.Shape()
	kernelShape := kernel.Shape()
	gradShape := grad.Shape()

	N := inputShape[0]
	CIn := inputShape[1]
	H := inputShape[2]
	W := inputShape[3]
	COut := kernelShape[0]
	KH := kernelShape[2]
	KW := kernelShape[3]
	HOut := gradShape[2]
	WOut := gradShape[3]

	// Create input gradient tensor
	inputGrad, err := tensor.NewRaw(tensor.Shape{N, CIn, H, W}, grad.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("Conv2DInputBackward: failed to create gradient tensor: %v", err))
	}

	// Dispatch by dtype and stride (stride specialization for compiler optimization)
	switch grad.DType() {
	case tensor.Float32:
		if stride == 1 && padding == 0 {
			conv2dInputBackwardFloat32Stride1NoPad(
				inputGrad, grad, kernel,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
			)
		} else {
			conv2dInputBackwardFloat32(
				inputGrad, grad, kernel,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
				stride, padding,
			)
		}
	case tensor.Float64:
		if stride == 1 && padding == 0 {
			conv2dInputBackwardFloat64Stride1NoPad(
				inputGrad, grad, kernel,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
			)
		} else {
			conv2dInputBackwardFloat64(
				inputGrad, grad, kernel,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
				stride, padding,
			)
		}
	default:
		panic("Conv2DInputBackward: unsupported dtype")
	}

	return inputGrad
}

// conv2dInputBackwardFloat32 computes input gradient for float32.
//
//nolint:dupl,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop
func conv2dInputBackwardFloat32(
	inputGrad, grad, kernel *tensor.RawTensor,
	n, cIn, h, w, cOut, kH, kW, hOut, wOut, stride, padding int,
) {
	inputGradData := inputGrad.AsFloat32()
	gradData := grad.AsFloat32()
	kernelData := kernel.AsFloat32()

	// Initialize to zero
	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	// For each batch
	for batch := 0; batch < n; batch++ {
		// Pre-slice batch planes
		inputGradBatchOffset := batch * cIn * h * w
		inputGradBatch := inputGradData[inputGradBatchOffset : inputGradBatchOffset+cIn*h*w]

		gradBatchOffset := batch * cOut * hOut * wOut
		gradBatch := gradData[gradBatchOffset : gradBatchOffset+cOut*hOut*wOut]

		// For each output gradient position
		for outH := 0; outH < hOut; outH++ {
			for outW := 0; outW < wOut; outW++ {
				// For each output channel
				for outChan := 0; outChan < cOut; outChan++ {
					gradIdx := outChan*hOut*wOut + outH*wOut + outW
					gradVal := gradBatch[gradIdx]

					// Pre-slice kernel for this output channel
					kernelCOutOffset := outChan * cIn * kH * kW
					kernelCOut := kernelData[kernelCOutOffset : kernelCOutOffset+cIn*kH*kW]

					// Distribute this gradient to all input positions
					for inChan := 0; inChan < cIn; inChan++ {
						// Pre-slice input gradient channel
						inputGradCInOffset := inChan * h * w
						inputGradCIn := inputGradBatch[inputGradCInOffset : inputGradCInOffset+h*w]

						// Pre-slice kernel for this input channel
						kernelCInOffset := inChan * kH * kW
						kernelCIn := kernelCOut[kernelCInOffset : kernelCInOffset+kH*kW]

						for kh := 0; kh < kH; kh++ {
							for kw := 0; kw < kW; kw++ {
								// Input position
								hPos := outH*stride - padding + kh
								wPos := outW*stride - padding + kw

								// Check bounds
								if hPos >= 0 && hPos < h && wPos >= 0 && wPos < w {
									kernelIdx := kh*kW + kw
									inputGradIdx := hPos*w + wPos

									// Accumulate gradient - single bounds check via pre-slice
									inputGradCIn[inputGradIdx] += gradVal * kernelCIn[kernelIdx]
								}
							}
						}
					}
				}
			}
		}
	}
}

// conv2dInputBackwardFloat64 computes input gradient for float64.
//
//nolint:dupl,gocritic,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop
func conv2dInputBackwardFloat64(
	inputGrad, grad, kernel *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	inputGradData := inputGrad.AsFloat64()
	gradData := grad.AsFloat64()
	kernelData := kernel.AsFloat64()

	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	for n := 0; n < N; n++ {
		// Pre-slice batch planes
		inputGradBatchOffset := n * CIn * H * W
		inputGradBatch := inputGradData[inputGradBatchOffset : inputGradBatchOffset+CIn*H*W]

		gradBatchOffset := n * COut * HOut * WOut
		gradBatch := gradData[gradBatchOffset : gradBatchOffset+COut*HOut*WOut]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				for cOut := 0; cOut < COut; cOut++ {
					gradIdx := cOut*HOut*WOut + outH*WOut + outW
					gradVal := gradBatch[gradIdx]

					// Pre-slice kernel for this output channel
					kernelCOutOffset := cOut * CIn * KH * KW
					kernelCOut := kernelData[kernelCOutOffset : kernelCOutOffset+CIn*KH*KW]

					for cIn := 0; cIn < CIn; cIn++ {
						// Pre-slice input gradient channel
						inputGradCInOffset := cIn * H * W
						inputGradCIn := inputGradBatch[inputGradCInOffset : inputGradCInOffset+H*W]

						// Pre-slice kernel for this input channel
						kernelCInOffset := cIn * KH * KW
						kernelCIn := kernelCOut[kernelCInOffset : kernelCInOffset+KH*KW]

						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								if h >= 0 && h < H && w >= 0 && w < W {
									kernelIdx := kh*KW + kw
									inputGradIdx := h*W + w
									// Single bounds check via pre-slice
									inputGradCIn[inputGradIdx] += gradVal * kernelCIn[kernelIdx]
								}
							}
						}
					}
				}
			}
		}
	}
}

// conv2dInputBackwardFloat32Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
//
//nolint:dupl,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop.
func conv2dInputBackwardFloat32Stride1NoPad(
	inputGrad, grad, kernel *tensor.RawTensor,
	n, cIn, h, w, cOut, kH, kW, hOut, wOut int,
) {
	inputGradData := inputGrad.AsFloat32()
	gradData := grad.AsFloat32()
	kernelData := kernel.AsFloat32()

	// Initialize to zero
	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	// For each batch
	for batch := 0; batch < n; batch++ {
		inputGradBatchOffset := batch * cIn * h * w
		inputGradBatch := inputGradData[inputGradBatchOffset : inputGradBatchOffset+cIn*h*w]

		gradBatchOffset := batch * cOut * hOut * wOut
		gradBatch := gradData[gradBatchOffset : gradBatchOffset+cOut*hOut*wOut]

		// For each output gradient position
		for outH := 0; outH < hOut; outH++ {
			for outW := 0; outW < wOut; outW++ {
				// For each output channel
				for outChan := 0; outChan < cOut; outChan++ {
					gradIdx := outChan*hOut*wOut + outH*wOut + outW
					gradVal := gradBatch[gradIdx]

					kernelCOutOffset := outChan * cIn * kH * kW
					kernelCOut := kernelData[kernelCOutOffset : kernelCOutOffset+cIn*kH*kW]

					// Distribute this gradient to all input positions
					for inChan := 0; inChan < cIn; inChan++ {
						inputGradCInOffset := inChan * h * w
						inputGradCIn := inputGradBatch[inputGradCInOffset : inputGradCInOffset+h*w]

						kernelCInOffset := inChan * kH * kW
						kernelCIn := kernelCOut[kernelCInOffset : kernelCInOffset+kH*kW]

						for kh := 0; kh < kH; kh++ {
							for kw := 0; kw < kW; kw++ {
								// With stride=1, padding=0: hPos = outH + kh
								hPos := outH + kh
								wPos := outW + kw

								// No bounds check needed (padding=0, stride=1)
								kernelIdx := kh*kW + kw
								inputGradIdx := hPos*w + wPos

								inputGradCIn[inputGradIdx] += gradVal * kernelCIn[kernelIdx]
							}
						}
					}
				}
			}
		}
	}
}

// conv2dInputBackwardFloat64Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
//
//nolint:dupl,gocognit,gocritic // Intentional duplication for float32/float64; high complexity inherent to convolution backprop.
func conv2dInputBackwardFloat64Stride1NoPad(
	inputGrad, grad, kernel *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut int,
) {
	inputGradData := inputGrad.AsFloat64()
	gradData := grad.AsFloat64()
	kernelData := kernel.AsFloat64()

	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	for n := 0; n < N; n++ {
		inputGradBatchOffset := n * CIn * H * W
		inputGradBatch := inputGradData[inputGradBatchOffset : inputGradBatchOffset+CIn*H*W]

		gradBatchOffset := n * COut * HOut * WOut
		gradBatch := gradData[gradBatchOffset : gradBatchOffset+COut*HOut*WOut]

		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				for cOut := 0; cOut < COut; cOut++ {
					gradIdx := cOut*HOut*WOut + outH*WOut + outW
					gradVal := gradBatch[gradIdx]

					kernelCOutOffset := cOut * CIn * KH * KW
					kernelCOut := kernelData[kernelCOutOffset : kernelCOutOffset+CIn*KH*KW]

					for cIn := 0; cIn < CIn; cIn++ {
						inputGradCInOffset := cIn * H * W
						inputGradCIn := inputGradBatch[inputGradCInOffset : inputGradCInOffset+H*W]

						kernelCInOffset := cIn * KH * KW
						kernelCIn := kernelCOut[kernelCInOffset : kernelCInOffset+KH*KW]

						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								// With stride=1, padding=0: h = outH + kh
								h := outH + kh
								w := outW + kw

								// No bounds check needed (padding=0, stride=1)
								kernelIdx := kh*KW + kw
								inputGradIdx := h*W + w
								inputGradCIn[inputGradIdx] += gradVal * kernelCIn[kernelIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Conv2DKernelBackward computes gradient w.r.t. kernel.
//
// Algorithm: Convolution of input with grad.
//   - For each kernel position (c_out, c_in, kh, kw):
//   - Sum over all batch samples and output positions
//   - Each contribution is: input[n, c_in, h, w] * grad[n, c_out, h_out, w_out]
//   - Where h = h_out * stride - padding + kh, w = w_out * stride - padding + kw
//
// References:
//   - Burn framework: crates/burn-autodiff/src/ops/module.rs (conv2d_weight_backward)
//
//nolint:dupl // Intentional duplication with Conv2DInputBackward (different operations)
func (cpu *CPUBackend) Conv2DKernelBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	inputShape := input.Shape()
	kernelShape := kernel.Shape()
	gradShape := grad.Shape()

	N := inputShape[0]
	CIn := inputShape[1]
	H := inputShape[2]
	W := inputShape[3]
	COut := kernelShape[0]
	KH := kernelShape[2]
	KW := kernelShape[3]
	HOut := gradShape[2]
	WOut := gradShape[3]

	kernelGrad, err := tensor.NewRaw(tensor.Shape{COut, CIn, KH, KW}, grad.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("Conv2DKernelBackward: failed to create gradient tensor: %v", err))
	}

	// Dispatch by dtype and stride (stride specialization for compiler optimization)
	switch grad.DType() {
	case tensor.Float32:
		if stride == 1 && padding == 0 {
			conv2dKernelBackwardFloat32Stride1NoPad(
				kernelGrad, grad, input,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
			)
		} else {
			conv2dKernelBackwardFloat32(
				kernelGrad, grad, input,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
				stride, padding,
			)
		}
	case tensor.Float64:
		if stride == 1 && padding == 0 {
			conv2dKernelBackwardFloat64Stride1NoPad(
				kernelGrad, grad, input,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
			)
		} else {
			conv2dKernelBackwardFloat64(
				kernelGrad, grad, input,
				N, CIn, H, W, COut, KH, KW, HOut, WOut,
				stride, padding,
			)
		}
	default:
		panic("Conv2DKernelBackward: unsupported dtype")
	}

	return kernelGrad
}

// conv2dKernelBackwardFloat32 computes kernel gradient for float32.
//
//nolint:dupl,gocritic,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop
func conv2dKernelBackwardFloat32(
	kernelGrad, grad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	kernelGradData := kernelGrad.AsFloat32()
	gradData := grad.AsFloat32()
	inputData := input.AsFloat32()

	// Initialize to zero
	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	// For each kernel weight
	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float32(0.0)

					// Accumulate gradient over all batch and output positions
					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								// Input position corresponding to this kernel weight
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								// Check bounds
								if h >= 0 && h < H && w >= 0 && w < W {
									inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
									gradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW

									sum += inputData[inputIdx] * gradData[gradIdx]
								}
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}

// conv2dKernelBackwardFloat64 computes kernel gradient for float64.
//
//nolint:dupl,gocritic,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop
func conv2dKernelBackwardFloat64(
	kernelGrad, grad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	kernelGradData := kernelGrad.AsFloat64()
	gradData := grad.AsFloat64()
	inputData := input.AsFloat64()

	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float64(0.0)

					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								if h >= 0 && h < H && w >= 0 && w < W {
									inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
									gradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
									sum += inputData[inputIdx] * gradData[gradIdx]
								}
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}

// conv2dKernelBackwardFloat32Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
//
//nolint:dupl,gocritic,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop.
func conv2dKernelBackwardFloat32Stride1NoPad(
	kernelGrad, grad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut int,
) {
	kernelGradData := kernelGrad.AsFloat32()
	gradData := grad.AsFloat32()
	inputData := input.AsFloat32()

	// Initialize to zero
	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	// For each kernel weight
	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float32(0.0)

					// Accumulate gradient over all batch and output positions
					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								// With stride=1, padding=0: h = outH + kh
								h := outH + kh
								w := outW + kw

								// No bounds check needed (padding=0, stride=1)
								inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
								gradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW

								sum += inputData[inputIdx] * gradData[gradIdx]
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}

// conv2dKernelBackwardFloat64Stride1NoPad is optimized for stride=1, padding=0.
// Compiler can better optimize this with hardcoded stride=1 (loop unrolling, SIMD).
//
//nolint:dupl,gocritic,gocognit // Intentional duplication for float32/float64; high complexity inherent to convolution backprop.
func conv2dKernelBackwardFloat64Stride1NoPad(
	kernelGrad, grad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut int,
) {
	kernelGradData := kernelGrad.AsFloat64()
	gradData := grad.AsFloat64()
	inputData := input.AsFloat64()

	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float64(0.0)

					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								// With stride=1, padding=0: h = outH + kh
								h := outH + kh
								w := outW + kw

								// No bounds check needed (padding=0, stride=1)
								inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
								gradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
								sum += inputData[inputIdx] * gradData[gradIdx]
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}
