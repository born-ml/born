// Package autodiff implements automatic differentiation using the decorator pattern.
//
// AutodiffBackend wraps any Backend implementation (CPU, GPU, etc.) and adds
// gradient tracking capabilities through a GradientTape.
//
// Architecture:
//   - Decorator pattern: AutodiffBackend[B] wraps any Backend implementation
//   - GradientTape: Records operations during forward pass
//   - Operation interface: Each op (Add, Mul, MatMul) implements backward pass
//   - Reverse-mode AD: Computes gradients efficiently using chain rule
//
// Usage:
//
//	// Wrap any backend with autodiff
//	cpuBackend := cpu.New()
//	autodiffBackend := autodiff.New(cpuBackend)
//
//	// Use with tensors
//	x := tensor.FromSlice([]float32{2.0}, tensor.Shape{1}, autodiffBackend)
//	y := x.Mul(x) // y = x²
//
//	// Compute gradients
//	y.Backward()
//	fmt.Println(x.Grad()) // dy/dx = 2x = 4.0
package autodiff

import (
	"math"

	"github.com/born-ml/born/internal/autodiff/ops"
	"github.com/born-ml/born/internal/tensor"
)

// AutodiffBackend wraps a Backend and adds automatic differentiation.
// It implements the tensor.Backend interface and records operations in a GradientTape.
//
// Type parameter B must satisfy the tensor.Backend interface.
type AutodiffBackend[B tensor.Backend] struct {
	inner B             // Wrapped backend (CPU, GPU, etc.)
	tape  *GradientTape // Records operations for backpropagation
}

// New creates a new AutodiffBackend wrapping the given backend.
func New[B tensor.Backend](backend B) *AutodiffBackend[B] {
	return &AutodiffBackend[B]{
		inner: backend,
		tape:  NewGradientTape(),
	}
}

// Tape returns the gradient tape for manual control.
// Useful for:
//   - Starting/stopping recording
//   - Clearing tape between iterations
//   - Inspecting recorded operations
func (b *AutodiffBackend[B]) Tape() *GradientTape {
	return b.tape
}

// Inner returns the wrapped backend for direct access.
func (b *AutodiffBackend[B]) Inner() B {
	return b.inner
}

// Name returns the backend name.
func (b *AutodiffBackend[B]) Name() string {
	return "Autodiff(" + b.inner.Name() + ")"
}

// Device returns the compute device.
func (b *AutodiffBackend[B]) Device() tensor.Device {
	return b.inner.Device()
}

// Add performs element-wise addition and records the operation.
func (b *AutodiffBackend[B]) Add(a, c *tensor.RawTensor) *tensor.RawTensor {
	// CRITICAL: Prevent inplace modification that would corrupt autodiff graph.
	// Temporarily increase refCount so IsUnique() returns false.
	// This forces CPU backend to allocate new result instead of inplace modification.
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Add(a, c)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewAddOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Sub performs element-wise subtraction and records the operation.
func (b *AutodiffBackend[B]) Sub(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Sub(a, c)

	if b.tape.IsRecording() {
		op := ops.NewSubOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Mul performs element-wise multiplication and records the operation.
func (b *AutodiffBackend[B]) Mul(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Mul(a, c)

	if b.tape.IsRecording() {
		op := ops.NewMulOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Div performs element-wise division and records the operation.
func (b *AutodiffBackend[B]) Div(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Div(a, c)

	if b.tape.IsRecording() {
		op := ops.NewDivOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// MatMul performs matrix multiplication and records the operation.
func (b *AutodiffBackend[B]) MatMul(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.MatMul(a, c)

	if b.tape.IsRecording() {
		op := ops.NewMatMulOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// BatchMatMul performs batched matrix multiplication and records the operation.
func (b *AutodiffBackend[B]) BatchMatMul(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.BatchMatMul(a, c)

	if b.tape.IsRecording() {
		op := ops.NewBatchMatMulOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Reshape reshapes a tensor and records the operation.
//
// CRITICAL: Like Transpose, Reshape must be recorded on tape!
// Without recording, gradients won't flow back to reshaped parameters.
//
// Example: Conv2D bias
//   - bias parameter: [out_channels]
//   - reshaped for broadcasting: [1, out_channels, 1, 1]
//   - Without ReshapeOp: gradient computed for reshaped tensor only
//   - With ReshapeOp: gradient propagates back to original bias parameter
func (b *AutodiffBackend[B]) Reshape(t *tensor.RawTensor, newShape tensor.Shape) *tensor.RawTensor {
	defer t.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Reshape(t, newShape)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewReshapeOp(t, result)
		b.tape.Record(op)
	}

	return result
}

// Transpose transposes a tensor and records the operation.
//
// CRITICAL: Even though conceptually transpose is a "view", the underlying
// backend may create a new tensor (e.g., CPU backend copies data).
// We MUST record this operation so gradients flow back correctly.
//
// For example, in Linear layer:
//
//	w = weight parameter
//	wT = w.Transpose()  // Creates NEW tensor!
//	output = input @ wT  // MatMul records operation with wT
//
// Without recording Transpose:
//   - Backward computes grad for wT (new tensor)
//   - Optimizer looks for grad of w (original parameter)
//   - NO GRADIENT FOUND! Parameters don't update!
//
// With TransposeOp:
//   - Backward computes grad for wT
//   - TransposeOp.Backward propagates grad back to w
//   - Optimizer finds grad for w ✓
func (b *AutodiffBackend[B]) Transpose(t *tensor.RawTensor, axes ...int) *tensor.RawTensor {
	defer t.ForceNonUnique()()

	// Handle default axes (reverse all dimensions)
	ndim := len(t.Shape())
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = ndim - 1 - i
		}
	}

	// Forward pass using wrapped backend
	result := b.inner.Transpose(t, axes...)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewTransposeOp(t, result, axes)
		b.tape.Record(op)
	}

	return result
}

// Conv2D performs 2D convolution and records the operation.
//
// CRITICAL: Conv2D must be recorded on tape for gradient flow!
// Just like Transpose, Conv2D creates new tensors and without recording,
// gradients won't flow back to the kernel/input parameters.
func (b *AutodiffBackend[B]) Conv2D(input, kernel *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	defer input.ForceNonUnique()()
	defer kernel.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Conv2D(input, kernel, stride, padding)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewConv2DOp(input, kernel, result, stride, padding)
		b.tape.Record(op)
	}

	return result
}

// MaxPool2D performs 2D max pooling and records the operation.
//
// CRITICAL: MaxPool2D must be recorded on tape for gradient flow!
// During backward pass, gradients only flow to positions that had max values.
// MaxPool2DOp stores max indices during forward pass for correct gradient routing.
func (b *AutodiffBackend[B]) MaxPool2D(input *tensor.RawTensor, kernelSize, stride int) *tensor.RawTensor {
	defer input.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.MaxPool2D(input, kernelSize, stride)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewMaxPool2DOp(input, result, kernelSize, stride)
		b.tape.Record(op)
	}

	return result
}

// ReLU applies ReLU activation and records the operation.
func (b *AutodiffBackend[B]) ReLU(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: max(0, x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply ReLU based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			if val > 0 {
				resData[i] = val
			} else {
				resData[i] = 0
			}
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			if val > 0 {
				resData[i] = val
			} else {
				resData[i] = 0
			}
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewReLUOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Sigmoid applies sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
func (b *AutodiffBackend[B]) Sigmoid(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: σ(x) = 1 / (1 + exp(-x))
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Sigmoid based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			// σ(x) = 1 / (1 + exp(-x))
			resData[i] = float32(1.0 / (1.0 + math.Exp(float64(-val))))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			// σ(x) = 1 / (1 + exp(-x))
			resData[i] = 1.0 / (1.0 + math.Exp(-val))
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewSigmoidOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Tanh applies hyperbolic tangent activation.
func (b *AutodiffBackend[B]) Tanh(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: tanh(x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Tanh based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			resData[i] = float32(math.Tanh(float64(val)))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			resData[i] = math.Tanh(val)
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewTanhOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// SiLU applies SiLU (Swish) activation: f(x) = x * sigmoid(x).
//
// SiLU (Sigmoid Linear Unit), also known as Swish, is widely used in
// modern transformer architectures (LLaMA, Mistral, GPT-Neo).
//
// Forward:
//
//	output = x * sigmoid(x)
//
// Backward:
//
//	dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
func (b *AutodiffBackend[B]) SiLU(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	// Forward pass: y = x * sigmoid(x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply SiLU based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			// sigmoid(x) = 1 / (1 + exp(-x))
			sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-val))))
			// y = x * sigmoid(x)
			resData[i] = val * sigmoid
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			// sigmoid(x) = 1 / (1 + exp(-x))
			sigmoid := 1.0 / (1.0 + math.Exp(-val))
			// y = x * sigmoid(x)
			resData[i] = val * sigmoid
		}

	default:
		panic("SiLU: only supports float32 and float64")
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewSiLUOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Log computes element-wise natural logarithm.
//
// Forward:
//
//	output = log(input)
//
// Backward:
//
//	∂L/∂input = ∂L/∂output * (1 / input)
//
// Note: Input values must be positive. For numerical stability with values
// close to zero, consider using LogWithEpsilon operation instead.
func (b *AutodiffBackend[B]) Log(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	// Forward pass: log(x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Log based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			resData[i] = float32(math.Log(float64(val)))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			resData[i] = math.Log(val)
		}

	default:
		panic("Log: only supports float32 and float64")
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewLogOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Softmax applies softmax activation along the specified dimension.
//
// Parameters:
//   - x: Input tensor
//   - dim: Dimension along which to compute softmax (-1 for last dimension)
//
// Forward (for each row):
//
//	softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
//
// The max-shifting ensures numerical stability (prevents overflow).
//
// Backward:
//
//	The Jacobian of softmax is complex, but the gradient simplifies to:
//	∂L/∂x_j = softmax_j * (∂L/∂softmax_j - Σ_i (∂L/∂softmax_i * softmax_i))
func (b *AutodiffBackend[B]) Softmax(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Softmax(x, dim)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewSoftmaxOp(x, result, dim)
		b.tape.Record(op)
	}

	return result
}

// CrossEntropy computes cross-entropy loss for classification.
//
// Forward:
//
//	Loss = mean(-log_softmax(logits)[targets])
//
// Uses the log-sum-exp trick for numerical stability.
//
// Backward:
//
//	∂L/∂logits = (softmax(logits) - y_one_hot) / batch_size
//
// Parameters:
//   - logits: Model predictions [batch_size, num_classes]
//   - targets: Ground truth class indices [batch_size]
//
// Returns:
//   - Scalar loss value (mean over batch)
func (b *AutodiffBackend[B]) CrossEntropy(logits, targets *tensor.RawTensor) *tensor.RawTensor {
	defer logits.ForceNonUnique()()
	// Note: targets doesn't need ForceNonUnique() as it's not differentiated

	// Forward pass using helper function
	result := ops.CrossEntropyForward(logits, targets, b.Device())

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewCrossEntropyOp(logits, targets, result)
		b.tape.Record(op)
	}

	return result
}

// Exp computes element-wise exponential and records the operation.
func (b *AutodiffBackend[B]) Exp(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.Exp(x)

	if b.tape.IsRecording() {
		op := ops.NewExpOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Sqrt computes element-wise square root and records the operation.
func (b *AutodiffBackend[B]) Sqrt(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.Sqrt(x)

	if b.tape.IsRecording() {
		op := ops.NewSqrtOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Rsqrt computes element-wise reciprocal square root and records the operation.
func (b *AutodiffBackend[B]) Rsqrt(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.Rsqrt(x)

	if b.tape.IsRecording() {
		op := ops.NewRsqrtOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Cos computes element-wise cosine and records the operation.
func (b *AutodiffBackend[B]) Cos(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.Cos(x)

	if b.tape.IsRecording() {
		op := ops.NewCosOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Sin computes element-wise sine and records the operation.
func (b *AutodiffBackend[B]) Sin(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.Sin(x)

	if b.tape.IsRecording() {
		op := ops.NewSinOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// SumDim sums tensor along a dimension and records the operation.
func (b *AutodiffBackend[B]) SumDim(x *tensor.RawTensor, dim int, keepDim bool) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.SumDim(x, dim, keepDim)

	if b.tape.IsRecording() {
		op := ops.NewSumDimOp(x, result, dim, keepDim)
		b.tape.Record(op)
	}

	return result
}

// MeanDim computes mean along a dimension and records the operation.
func (b *AutodiffBackend[B]) MeanDim(x *tensor.RawTensor, dim int, keepDim bool) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	result := b.inner.MeanDim(x, dim, keepDim)

	if b.tape.IsRecording() {
		op := ops.NewMeanDimOp(x, result, dim, keepDim)
		b.tape.Record(op)
	}

	return result
}

// NoGrad temporarily disables gradient recording for inference.
//
// This is useful for:
//   - Inference/evaluation (no need to track gradients)
//   - Gradient-free operations (e.g., updating exponential moving averages)
//   - Memory optimization (gradient tape doesn't grow)
//
// The function executes the provided function with gradient recording disabled,
// then restores the previous recording state.
//
// Example:
//
//	// Inference mode
//	backend.NoGrad(func() {
//	    output := model.Forward(input)  // No gradients recorded
//	    predictions := output.ArgMax()
//	})
//
//	// Training continues normally
//	loss := model.Forward(trainInput)
//	loss.Backward()  // Gradients computed
func (b *AutodiffBackend[B]) NoGrad(fn func()) {
	wasRecording := b.tape.IsRecording()
	b.tape.StopRecording()
	defer func() {
		if wasRecording {
			b.tape.StartRecording()
		}
	}()
	fn()
}

// MulScalar multiplies tensor elements by a scalar (autodiff proxy).
func (b *AutodiffBackend[B]) MulScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	return b.inner.MulScalar(x, scalar)
}

// AddScalar adds a scalar to tensor elements (autodiff proxy).
func (b *AutodiffBackend[B]) AddScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	return b.inner.AddScalar(x, scalar)
}

// SubScalar subtracts a scalar from tensor elements (autodiff proxy).
func (b *AutodiffBackend[B]) SubScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	return b.inner.SubScalar(x, scalar)
}

// DivScalar divides tensor elements by a scalar (autodiff proxy).
func (b *AutodiffBackend[B]) DivScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	return b.inner.DivScalar(x, scalar)
}

// Greater performs element-wise greater-than comparison (autodiff proxy).
func (b *AutodiffBackend[B]) Greater(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Greater(a, other)
}

// Lower performs element-wise less-than comparison (autodiff proxy).
func (b *AutodiffBackend[B]) Lower(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Lower(a, other)
}

// GreaterEqual performs element-wise greater-or-equal comparison (autodiff proxy).
func (b *AutodiffBackend[B]) GreaterEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.GreaterEqual(a, other)
}

// LowerEqual performs element-wise less-or-equal comparison (autodiff proxy).
func (b *AutodiffBackend[B]) LowerEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.LowerEqual(a, other)
}

// Equal performs element-wise equality comparison (autodiff proxy).
func (b *AutodiffBackend[B]) Equal(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Equal(a, other)
}

// NotEqual performs element-wise inequality comparison (autodiff proxy).
func (b *AutodiffBackend[B]) NotEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.NotEqual(a, other)
}

// Or performs element-wise logical OR (autodiff proxy).
func (b *AutodiffBackend[B]) Or(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Or(a, other)
}

// And performs element-wise logical AND (autodiff proxy).
func (b *AutodiffBackend[B]) And(a, other *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.And(a, other)
}

// Not performs element-wise logical NOT (autodiff proxy).
func (b *AutodiffBackend[B]) Not(x *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Not(x)
}

// Sum reduces tensor to a single scalar by summing all elements (autodiff proxy).
func (b *AutodiffBackend[B]) Sum(x *tensor.RawTensor) *tensor.RawTensor {
	return b.inner.Sum(x)
}

// Argmax returns indices of maximum values along a dimension (autodiff proxy).
func (b *AutodiffBackend[B]) Argmax(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	return b.inner.Argmax(x, dim)
}

// Expand broadcasts tensor to a larger shape (autodiff proxy).
func (b *AutodiffBackend[B]) Expand(x *tensor.RawTensor, shape tensor.Shape) *tensor.RawTensor {
	return b.inner.Expand(x, shape)
}

// Cast converts tensor to a different data type (autodiff proxy).
func (b *AutodiffBackend[B]) Cast(x *tensor.RawTensor, dtype tensor.DataType) *tensor.RawTensor {
	return b.inner.Cast(x, dtype)
}

// Cat concatenates tensors along a dimension (passthrough - no autodiff yet).
func (b *AutodiffBackend[B]) Cat(tensors []*tensor.RawTensor, dim int) *tensor.RawTensor {
	// Mark all inputs as non-unique for safety
	//nolint:gocritic // defer in loop is intentional for cleanup of all inputs
	for _, t := range tensors {
		defer t.ForceNonUnique()()
	}

	// TODO(TASK-015): Implement CatOp for gradient computation
	// For now, passthrough to inner backend
	return b.inner.Cat(tensors, dim)
}

// Chunk splits tensor into equal parts (passthrough - no autodiff yet).
func (b *AutodiffBackend[B]) Chunk(x *tensor.RawTensor, n, dim int) []*tensor.RawTensor {
	defer x.ForceNonUnique()()

	// TODO(TASK-015): Implement ChunkOp for gradient computation
	// For now, passthrough to inner backend
	return b.inner.Chunk(x, n, dim)
}

// Unsqueeze adds a dimension (recorded via Reshape).
func (b *AutodiffBackend[B]) Unsqueeze(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	// Unsqueeze is just a reshape operation, so use Reshape which is already recorded
	return b.inner.Unsqueeze(x, dim)
}

// Squeeze removes a dimension (recorded via Reshape).
func (b *AutodiffBackend[B]) Squeeze(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	// Squeeze is just a reshape operation, so use Reshape which is already recorded
	return b.inner.Squeeze(x, dim)
}

// Gather selects elements along dim using index tensor (passthrough - no autodiff yet).
func (b *AutodiffBackend[B]) Gather(x *tensor.RawTensor, dim int, index *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()
	defer index.ForceNonUnique()()

	// TODO(TASK-016): Implement GatherOp for gradient computation
	// Backward: scatter_add grad_output to grad_input at index positions
	// For now, passthrough to inner backend
	return b.inner.Gather(x, dim, index)
}

// Where performs conditional element selection (passthrough - no autodiff yet).
func (b *AutodiffBackend[B]) Where(condition, x, y *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()
	defer y.ForceNonUnique()()

	// TODO(TASK-016): Implement WhereOp for gradient computation
	// Backward: grad_x = where(cond, grad_out, 0), grad_y = where(cond, 0, grad_out)
	// For now, passthrough to inner backend
	return b.inner.Where(condition, x, y)
}
