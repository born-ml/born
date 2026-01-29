package main

import (
	"fmt"

	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// MNISTNetCNN is a LeNet-5 style convolutional neural network for MNIST classification.
//
// Architecture:
//
//	Input: [batch, 1, 28, 28] (grayscale MNIST images)
//	Conv1: 1 → 6 channels, 5x5 kernel -> [batch, 6, 24, 24]
//	ReLU
//	MaxPool: 2x2 -> [batch, 6, 12, 12]
//	Conv2: 6 → 16 channels, 5x5 kernel -> [batch, 16, 8, 8]
//	ReLU
//	MaxPool: 2x2 -> [batch, 16, 4, 4]
//	Flatten -> [batch, 256]
//	FC1: 256 → 120
//	ReLU
//	FC2: 120 → 84
//	ReLU
//	FC3: 84 → 10 (class scores)
//
// This architecture is inspired by LeNet-5 (LeCun et al., 1998)
// adapted for 28x28 MNIST images.
type MNISTNetCNN[B tensor.Backend] struct {
	conv1 *nn.Conv2D[B]    // First convolutional layer
	relu1 *nn.ReLU[B]      // Activation after conv1
	pool1 *nn.MaxPool2D[B] // Pooling after conv1
	conv2 *nn.Conv2D[B]    // Second convolutional layer
	relu2 *nn.ReLU[B]      // Activation after conv2
	pool2 *nn.MaxPool2D[B] // Pooling after conv2
	fc1   *nn.Linear[B]    // First fully connected layer
	relu3 *nn.ReLU[B]      // Activation after fc1
	fc2   *nn.Linear[B]    // Second fully connected layer
	relu4 *nn.ReLU[B]      // Activation after fc2
	fc3   *nn.Linear[B]    // Output layer
}

// NewMNISTNetCNN creates a new convolutional MNIST classification network.
//
// The network uses:
//   - Xavier initialization for Conv2D and Linear layers
//   - ReLU activations throughout
//   - Max pooling for spatial downsampling
//
// Total parameters: ~61,706 (much fewer than MLP due to weight sharing in convolutions)
func NewMNISTNetCNN[B tensor.Backend](backend B) *MNISTNetCNN[B] {
	return &MNISTNetCNN[B]{
		// Convolutional layers
		conv1: nn.NewConv2D(1, 6, 5, 5, 1, 0, true, backend), // 1->6, 5x5, stride=1, no padding
		relu1: nn.NewReLU[B](),
		pool1: nn.NewMaxPool2D(2, 2, backend),                 // 2x2 pooling
		conv2: nn.NewConv2D(6, 16, 5, 5, 1, 0, true, backend), // 6->16, 5x5
		relu2: nn.NewReLU[B](),
		pool2: nn.NewMaxPool2D(2, 2, backend), // 2x2 pooling

		// Fully connected layers
		fc1:   nn.NewLinear[B](16*4*4, 120, backend), // 256 -> 120
		relu3: nn.NewReLU[B](),
		fc2:   nn.NewLinear[B](120, 84, backend), // 120 -> 84
		relu4: nn.NewReLU[B](),
		fc3:   nn.NewLinear[B](84, 10, backend), // 84 -> 10
	}
}

// Forward performs forward pass through the CNN.
//
// Parameters:
//   - input: Batch of images with shape [batch_size, 1, 28, 28]
//
// Returns:
//   - logits: Unnormalized scores for each class with shape [batch_size, 10]
//
// Note: Returns raw logits (no softmax). CrossEntropyLoss will handle softmax internally.
func (m *MNISTNetCNN[B]) Forward(input *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B] {
	// Ensure input is 4D [batch, 1, 28, 28]
	inputShape := input.Shape()
	if len(inputShape) == 2 {
		// Reshape from [batch, 784] to [batch, 1, 28, 28]
		batchSize := inputShape[0]
		input = input.Reshape(batchSize, 1, 28, 28)
	} else if len(inputShape) != 4 {
		panic(fmt.Sprintf("Expected 2D [batch, 784] or 4D [batch, 1, 28, 28] input, got %dD", len(inputShape)))
	}

	// Convolutional block 1
	x := m.conv1.Forward(input) // [batch, 6, 24, 24]
	x = m.relu1.Forward(x)
	x = m.pool1.Forward(x) // [batch, 6, 12, 12]

	// Convolutional block 2
	x = m.conv2.Forward(x) // [batch, 16, 8, 8]
	x = m.relu2.Forward(x)
	x = m.pool2.Forward(x) // [batch, 16, 4, 4]

	// Flatten for fully connected layers
	batchSize := x.Shape()[0]
	x = x.Reshape(batchSize, 16*4*4) // [batch, 256]

	// Fully connected block
	x = m.fc1.Forward(x) // [batch, 120]
	x = m.relu3.Forward(x)
	x = m.fc2.Forward(x) // [batch, 84]
	x = m.relu4.Forward(x)
	x = m.fc3.Forward(x) // [batch, 10]

	return x
}

// Parameters returns all trainable parameters.
func (m *MNISTNetCNN[B]) Parameters() []*nn.Parameter[B] {
	// Preallocate: 5 layers × 2 params (weight + bias) = 10 params.
	params := make([]*nn.Parameter[B], 0, 10)
	params = append(params, m.conv1.Parameters()...)
	params = append(params, m.conv2.Parameters()...)
	params = append(params, m.fc1.Parameters()...)
	params = append(params, m.fc2.Parameters()...)
	params = append(params, m.fc3.Parameters()...)
	return params
}

// String returns a string representation of the model architecture.
func (m *MNISTNetCNN[B]) String() string {
	return fmt.Sprintf(`MNISTNetCNN(
  %s
  %s
  %s
  %s
  %s
  %s
  Linear(in=256, out=120)
  ReLU()
  Linear(in=120, out=84)
  ReLU()
  Linear(in=84, out=10)
)`,
		m.conv1.String(),
		"ReLU()",
		m.pool1.String(),
		m.conv2.String(),
		"ReLU()",
		m.pool2.String(),
	)
}
