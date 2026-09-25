// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

// Package autodiff provides automatic differentiation capabilities.
//
// This package implements reverse-mode automatic differentiation (backpropagation)
// using a gradient tape. It wraps any backend to add autodiff capabilities.
//
// Example:
//
//	import (
//	    "github.com/born-ml/born/autodiff"
//	    "github.com/born-ml/born/backend/cpu"
//	    "github.com/born-ml/born/tensor"
//	)
//
//	func main() {
//	    // Wrap CPU backend with autodiff
//	    base := cpu.New()
//	    backend := autodiff.New(base)
//
//	    // Use for training
//	    x := tensor.Zeros[float32](tensor.Shape{2, 3}, backend)
//	    y := x.Add(x)  // Operations recorded on tape
//
//	    // Compute gradients
//	    grads := backend.Backward(y.Raw())
//	}
package autodiff

import (
	"github.com/born-ml/born/internal/autodiff"
	internaltensor "github.com/born-ml/born/internal/tensor"
	"github.com/born-ml/born/tensor"
)

// Backend is the autodiff-enabled backend.
type Backend[B tensor.Backend] = autodiff.AutodiffBackend[B]

// New creates a new autodiff backend wrapping the given backend.
//
// Example:
//
//	base := cpu.New()
//	backend := autodiff.New(base)
func New[B tensor.Backend](backend B) *Backend[B] {
	return autodiff.New(backend)
}

// GradientTape records operations for automatic differentiation.
type GradientTape = autodiff.GradientTape

// NewGradientTape creates a new gradient tape.
func NewGradientTape() *GradientTape {
	return autodiff.NewGradientTape()
}

// BackwardCapable interface for backends that support backpropagation.
type BackwardCapable = autodiff.BackwardCapable

// Backward computes gradients via backpropagation.
func Backward[T tensor.DType, B BackwardCapable](t *tensor.Tensor[T, B], backend B) map[*tensor.RawTensor]*tensor.RawTensor {
	return autodiff.Backward(t, backend)
}

// ReleaseGradients releases GPU buffers for all gradient tensors in the map.
// Call after optimizer.Step(grads) to free GPU memory immediately.
// Pass the backend so the releaser can schedule GPU buffer deferred destruction.
// Without a backend the function is a no-op (backwards-compatible default).
func ReleaseGradients(grads map[*tensor.RawTensor]*tensor.RawTensor, backend ...tensor.Backend) {
	// The public tensor.Backend and internal/tensor.Backend are structurally
	// identical but are separate interface types, so we cannot forward the
	// variadic slice directly. Implement the release here using the internal
	// BackendReleaser interface.
	var br internaltensor.BackendReleaser
	if len(backend) > 0 && backend[0] != nil {
		br, _ = any(backend[0]).(internaltensor.BackendReleaser)
	}
	for _, grad := range grads {
		if br != nil {
			br.ReleaseBackendData(grad.BackendData())
			grad.SetBackendData(nil)
		}
	}
}

// NoGrad disables gradient recording for operations within the function.
// Use this for inference or operations that shouldn't be differentiated.
//
// Example:
//
//	backend := autodiff.New(cpu.New())
//	autodiff.NoGrad(backend, func() {
//	    // Operations here won't be recorded for gradients
//	    output := model.Forward(input)
//	})
func NoGrad[B tensor.Backend](backend *Backend[B], fn func()) {
	backend.NoGrad(fn)
}
