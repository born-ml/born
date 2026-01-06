// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/born-ml/born/internal/serialization"
	"github.com/born-ml/born/tensor"
)

// Module is the base interface for all neural network components.
//
// Every NN module must implement:
//   - Forward: Compute output from input
//   - Parameters: Return all trainable parameters
//   - StateDict: Export parameters for serialization
//   - LoadStateDict: Import parameters from serialization
//
// Modules can be composed to build complex architectures:
//
//	model := nn.NewSequential(
//	    nn.NewLinear(784, 128, backend),
//	    nn.NewReLU[Backend](),
//	    nn.NewLinear(128, 10, backend),
//	)
//
// Type parameter B must satisfy the tensor.Backend interface.
type Module[B tensor.Backend] interface {
	// Forward computes the output of the module given an input tensor.
	//
	// The input tensor should have the appropriate shape for this module.
	// For example, Linear expects [batch_size, in_features].
	//
	// Returns the output tensor with shape determined by the module type.
	Forward(input *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B]

	// Parameters returns all trainable parameters of this module.
	//
	// This includes weights, biases, and any nested module parameters.
	// Returns an empty slice for modules without trainable parameters
	// (e.g., activation functions).
	Parameters() []*Parameter[B]

	// StateDict returns a map of parameter names to raw tensors.
	//
	// This is used for serialization. The returned map contains all
	// trainable parameters with their names as keys.
	StateDict() map[string]*tensor.RawTensor

	// LoadStateDict loads parameters from a state dictionary.
	//
	// This is used for deserialization. The state dictionary should
	// contain parameter names as keys and RawTensors as values.
	//
	// Returns an error if a required parameter is missing or has wrong shape.
	LoadStateDict(stateDict map[string]*tensor.RawTensor) error
}

// Note: Internal implementations of Module automatically satisfy this interface
// because they have the same method signatures.

// Save saves a module to a .born file.
//
// This is a convenience function that exports the module's state dictionary
// and writes it to a file using the Born native format.
//
// Parameters:
//   - module: The module to save
//   - path: File path to write to
//   - modelType: Type name of the model (e.g., "Sequential", "Linear")
//   - metadata: Optional metadata (can be nil)
//
// Returns an error if saving fails.
//
// Example:
//
//	backend := cpu.New()
//	model := nn.NewLinear(784, 10, backend)
//	err := nn.Save(model, "model.born", "Linear", nil)
func Save[B tensor.Backend](module Module[B], path, modelType string, metadata map[string]string) error {
	// Get state dictionary
	stateDict := module.StateDict()

	// Create writer
	writer, err := serialization.NewBornWriter(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = writer.Close()
	}()

	// Write state dictionary
	return writer.WriteStateDict(stateDict, modelType, metadata)
}

// Load loads a module from a .born file.
//
// This is a convenience function that reads a state dictionary from a file
// and loads it into the provided module.
//
// Parameters:
//   - path: File path to read from
//   - backend: Backend to use for tensors
//   - module: The module to load into (will be modified)
//
// Returns the header and an error if loading fails.
//
// Example:
//
//	backend := cpu.New()
//	model := nn.NewLinear(784, 10, backend)
//	header, err := nn.Load("model.born", backend, model)
func Load[B tensor.Backend](path string, backend B, module Module[B]) (serialization.Header, error) {
	// Create reader
	reader, err := serialization.NewBornReader(path)
	if err != nil {
		return serialization.Header{}, err
	}
	defer func() {
		_ = reader.Close()
	}()

	// Read state dictionary
	stateDict, err := reader.ReadStateDict(backend)
	if err != nil {
		return serialization.Header{}, err
	}

	// Load into module
	if err := module.LoadStateDict(stateDict); err != nil {
		return serialization.Header{}, err
	}

	return reader.Header(), nil
}
