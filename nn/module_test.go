// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package nn_test

import (
	"testing"

	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/tensor"
	"github.com/born-ml/born/nn"
)

// TestModuleInterface verifies that concrete types implement Module interface.
func TestModuleInterface(t *testing.T) {
	backend := cpu.New()

	tests := []struct {
		name   string
		module nn.Module[*cpu.CPUBackend]
	}{
		{
			name:   "Linear",
			module: nn.NewLinear(10, 5, backend),
		},
		{
			name: "Sequential",
			module: nn.NewSequential[*cpu.CPUBackend](
				nn.NewLinear(10, 5, backend),
				nn.NewReLU[*cpu.CPUBackend](),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify Forward works
			input := tensor.Randn[float32](tensor.Shape{2, 10}, backend)
			_ = tt.module.Forward(input)

			// Verify Parameters works
			params := tt.module.Parameters()
			if params == nil {
				t.Error("Parameters() returned nil, expected non-nil slice")
			}

			// Verify StateDict works
			stateDict := tt.module.StateDict()
			if stateDict == nil {
				t.Error("StateDict() returned nil, expected non-nil map")
			}
		})
	}
}

// TestParameterInterface verifies that concrete Parameter implements interface.
func TestParameterInterface(t *testing.T) {
	backend := cpu.New()
	tensorData := tensor.Randn[float32](tensor.Shape{3, 3}, backend)

	param := nn.NewParameter("test.weight", tensorData)

	// Verify interface methods
	if name := param.Name(); name != "test.weight" {
		t.Errorf("Name() = %q, want %q", name, "test.weight")
	}

	if got := param.Tensor(); got != tensorData {
		t.Error("Tensor() returned different tensor than provided")
	}

	if grad := param.Grad(); grad != nil {
		t.Error("Grad() should be nil before backward pass")
	}

	// Test SetGrad
	gradTensor := tensor.Zeros[float32](tensor.Shape{3, 3}, backend)
	param.SetGrad(gradTensor)
	if got := param.Grad(); got != gradTensor {
		t.Error("Grad() returned different tensor after SetGrad")
	}

	// Test ZeroGrad
	param.ZeroGrad()
	if grad := param.Grad(); grad != nil {
		t.Error("Grad() should be nil after ZeroGrad()")
	}
}

// TestModuleComposition verifies modules can be composed.
func TestModuleComposition(t *testing.T) {
	backend := cpu.New()

	// Create a sequential model
	model := nn.NewSequential[*cpu.CPUBackend](
		nn.NewLinear(784, 128, backend),
		nn.NewReLU[*cpu.CPUBackend](),
		nn.NewLinear(128, 10, backend),
	)

	// Verify it implements Module
	var _ nn.Module[*cpu.CPUBackend] = model

	// Test forward pass
	input := tensor.Randn[float32](tensor.Shape{2, 784}, backend)
	output := model.Forward(input)

	expectedShape := tensor.Shape{2, 10}
	if !output.Shape().Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", output.Shape(), expectedShape)
	}

	// Verify parameters from nested modules
	params := model.Parameters()
	// 2 Linear layers: weights + biases = 4 parameters
	if len(params) != 4 {
		t.Errorf("Parameters() returned %d params, want 4", len(params))
	}
}

// TestNewParameter verifies parameter creation.
func TestNewParameter(t *testing.T) {
	backend := cpu.New()

	tests := []struct {
		name        string
		paramName   string
		tensorShape tensor.Shape
	}{
		{
			name:        "weight parameter",
			paramName:   "layer1.weight",
			tensorShape: tensor.Shape{128, 784},
		},
		{
			name:        "bias parameter",
			paramName:   "layer1.bias",
			tensorShape: tensor.Shape{128},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensorData := tensor.Randn[float32](tt.tensorShape, backend)
			param := nn.NewParameter(tt.paramName, tensorData)

			if got := param.Name(); got != tt.paramName {
				t.Errorf("Name() = %q, want %q", got, tt.paramName)
			}

			if got := param.Tensor(); got != tensorData {
				t.Error("Tensor() returned different tensor")
			}
		})
	}
}
