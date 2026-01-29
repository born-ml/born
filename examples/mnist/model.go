package main

import (
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// MNISTNet is a simple fully-connected neural network for MNIST classification.
//
// Architecture:
//   - Input: 784 neurons (28×28 flattened image)
//   - Hidden: 128 neurons with ReLU activation
//   - Output: 10 neurons (logits for 10 digit classes)
//
// This matches the standard simple baseline for MNIST (97-98% accuracy).
type MNISTNet[B tensor.Backend] struct {
	fc1  *nn.Linear[B] // 784 → 128
	relu *nn.ReLU[B]   // ReLU activation
	fc2  *nn.Linear[B] // 128 → 10
}

// NewMNISTNet creates a new MNIST classification network.
//
// The network uses Xavier/Glorot initialization for weights,
// which helps achieve good training dynamics with sigmoid/tanh activations
// and is also effective with ReLU.
func NewMNISTNet[B tensor.Backend](backend B) *MNISTNet[B] {
	return &MNISTNet[B]{
		fc1:  nn.NewLinear[B](784, 128, backend),
		relu: nn.NewReLU[B](),
		fc2:  nn.NewLinear[B](128, 10, backend),
	}
}

// Forward performs forward pass through the network.
//
// Parameters:
//   - input: Batch of flattened images with shape [batch_size, 784]
//
// Returns:
//   - logits: Unnormalized scores for each class with shape [batch_size, 10]
//
// Note: Returns raw logits (no softmax). CrossEntropyLoss will handle softmax internally.
func (m *MNISTNet[B]) Forward(input *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B] {
	// Ensure input is flattened to [batch_size, 784]
	inputShape := input.Shape()
	if len(inputShape) == 1 {
		// Single sample: reshape to [1, 784]
		input = input.Reshape(1, 784)
	} else if len(inputShape) != 2 || inputShape[1] != 784 {
		panic("MNISTNet: input must have shape [batch_size, 784] or [784]")
	}

	// Layer 1: Linear (784 → 128)
	x := m.fc1.Forward(input)

	// ReLU activation
	x = m.relu.Forward(x)

	// Layer 2: Linear (128 → 10)
	logits := m.fc2.Forward(x)

	return logits
}

// Parameters returns all trainable parameters of the network.
//
// This is used by optimizers to update weights and biases during training.
func (m *MNISTNet[B]) Parameters() []*nn.Parameter[B] {
	// Preallocate: 2 layers × 2 params (weight + bias) = 4 params.
	params := make([]*nn.Parameter[B], 0, 4)
	params = append(params, m.fc1.Parameters()...)
	params = append(params, m.fc2.Parameters()...)
	return params
}
