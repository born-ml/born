package nn

import (
	"github.com/born-ml/born/internal/tensor"
)

// MSELoss computes Mean Squared Error loss.
//
// Loss = mean((predictions - targets)²)
//
// MSE is commonly used for regression tasks where the goal is to predict
// continuous values.
//
// Example:
//
//	mse := nn.NewMSELoss[Backend]()
//	predictions := model.Forward(input)
//	loss := mse.Forward(predictions, targets)
type MSELoss[B tensor.Backend] struct {
	backend B
}

// NewMSELoss creates a new MSE loss function.
func NewMSELoss[B tensor.Backend](backend B) *MSELoss[B] {
	return &MSELoss[B]{
		backend: backend,
	}
}

// Forward computes the MSE loss.
//
// Loss = mean((predictions - targets)²) = sum((predictions - targets)²) / N
//
// All operations go through the backend so they are recorded on the autodiff
// tape and gradients flow back to predictions: d(MSE)/dx = 2*(x-t)/N.
//
// Parameters:
//   - predictions: Model predictions with shape [batch_size, ...]
//   - targets: Ground truth targets with same shape as predictions
//
// Returns a scalar loss value (shape [1]).
func (m *MSELoss[B]) Forward(predictions, targets *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B] {
	// Validate shapes match
	if !predictions.Shape().Equal(targets.Shape()) {
		panic("MSELoss: predictions and targets must have the same shape")
	}

	// diff = predictions - targets
	diff := predictions.Sub(targets)

	// squared = diff²
	squared := diff.Mul(diff)

	// sumRaw = sum of all squared elements (scalar, recorded on tape)
	sumRaw := m.backend.Sum(squared.Raw())

	// mean = sum / N — DivScalar is also a backend op, recorded on tape.
	// This ensures the full gradient chain: d(mean)/d(squared_i) = 1/N flows back.
	n := float32(predictions.Shape().NumElements())
	meanRaw := m.backend.DivScalar(sumRaw, n)

	return tensor.New[float32, B](meanRaw, m.backend)
}

// Parameters returns an empty slice (loss functions have no trainable parameters).
func (m *MSELoss[B]) Parameters() []*Parameter[B] {
	return nil
}

// NOTE: CrossEntropyLoss has been moved to cross_entropy.go
// See internal/nn/cross_entropy.go for the full implementation with numerical stability.
