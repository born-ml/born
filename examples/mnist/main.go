package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/born-ml/born/autodiff"
	"github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/nn"
	"github.com/born-ml/born/optim"
	"github.com/born-ml/born/tensor"
)

func main() {
	// Parse command line arguments
	dataDir := flag.String("data", "./data", "Directory containing MNIST data files")
	useTrain := flag.Bool("train", true, "Use training set (60K samples) vs test set (10K samples)")
	maxSamples := flag.Int("samples", 0, "Max samples to load (0 = all)")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch", 32, "Batch size for training")
	lr := flag.Float64("lr", 0.001, "Learning rate for Adam optimizer")
	useSynthetic := flag.Bool("synthetic", false, "Use synthetic data (for testing without MNIST files)")
	outPath := flag.String("out", "", "Output .born file to write serialized model")
	flag.Parse()

	fmt.Println("🚀 Born ML Framework - MNIST Classification")
	fmt.Println("=" + string(make([]byte, 70)))

	// Initialize backend with autodiff
	cpuBackend := cpu.New()
	backend := autodiff.New(cpuBackend)

	// Load MNIST data
	var trainData, valData *MNISTData
	var err error

	if *useSynthetic {
		fmt.Println("\n📊 Using synthetic data (embedded test patterns)...")
		data := CreateEmbeddedMNIST()
		trainData, valData = data.Split(0.2)
		fmt.Printf("   Train: %d samples, Val: %d samples\n", trainData.NumSamples(), valData.NumSamples())
	} else {
		fmt.Printf("\n📊 Loading MNIST data from: %s\n", *dataDir)

		// Load training data
		fmt.Println("   Loading training set...")
		allData, err := LoadMNIST(*dataDir, *useTrain, *maxSamples)
		if err != nil {
			// Try to show helpful error message
			if os.IsNotExist(err) {
				fmt.Println("\n❌ Error: MNIST data files not found!")
				fmt.Println("\nTo download MNIST dataset:")
				fmt.Println("  1. Create a 'data' directory: mkdir data")
				fmt.Println("  2. Download files from: http://yann.lecun.com/exdb/mnist/")
				fmt.Println("     - train-images-idx3-ubyte.gz")
				fmt.Println("     - train-labels-idx1-ubyte.gz")
				fmt.Println("     - t10k-images-idx3-ubyte.gz")
				fmt.Println("     - t10k-labels-idx1-ubyte.gz")
				fmt.Println("  3. Extract (gunzip) into the data directory")
				fmt.Println("\nOr run with -synthetic flag to use embedded test data:")
				fmt.Println("  go run . -synthetic")
				os.Exit(1)
			}
			log.Fatalf("Failed to load MNIST: %v", err)
		}

		// Split into train/val (80/20)
		trainData, valData = allData.Split(0.2)
		fmt.Printf("   Loaded %d total samples\n", allData.NumSamples())
		fmt.Printf("   Train: %d samples, Val: %d samples\n", trainData.NumSamples(), valData.NumSamples())
	}

	// Create model
	fmt.Println("\n🧠 Creating MNISTNet model (784 → 128 → 10)...")
	model := NewMNISTNet(backend)
	numParams := countParameters(model)
	fmt.Printf("   Model has %d trainable parameters\n", numParams)

	// Create optimizer (Adam with default parameters)
	fmt.Printf("\n⚙️  Training Configuration:\n")
	fmt.Printf("   Optimizer: Adam (lr=%.4f, betas=(0.9, 0.999))\n", *lr)
	fmt.Printf("   Loss: CrossEntropyLoss (with autodiff)\n")
	fmt.Printf("   Batch Size: %d\n", *batchSize)
	fmt.Printf("   Epochs: %d\n", *epochs)

	optimizer := optim.NewAdam(
		model.Parameters(),
		optim.AdamConfig{
			LR:    float32(*lr),
			Betas: [2]float32{0.9, 0.999},
			Eps:   1e-8,
		},
		backend,
	)

	// CRITICAL: Enable gradient recording!
	backend.Tape().StartRecording()

	// Create batches
	fmt.Println("\n📦 Creating data batches...")
	trainBatches, err := CreateBatches(trainData, *batchSize, true, backend)
	if err != nil {
		log.Fatalf("Failed to create train batches: %v", err)
	}
	valBatches, err := CreateBatches(valData, 256, false, backend) // Larger batch for validation
	if err != nil {
		log.Fatalf("Failed to create val batches: %v", err)
	}
	fmt.Printf("   Train batches: %d\n", len(trainBatches))
	fmt.Printf("   Val batches: %d\n", len(valBatches))

	// Training loop
	fmt.Println("\n🎓 Starting training...")
	fmt.Println("=" + string(make([]byte, 70)))

	for epoch := 0; epoch < *epochs; epoch++ {
		// Train for one epoch
		avgLoss, trainAcc := trainEpoch(model, trainBatches, optimizer, backend)

		// Validate
		valLoss, valAcc := validate(model, valBatches, backend)

		// Print progress
		fmt.Printf("Epoch %2d/%d: Loss=%.4f, Train Acc=%.2f%%, Val Loss=%.4f, Val Acc=%.2f%%\n",
			epoch+1, *epochs, avgLoss, trainAcc*100, valLoss, valAcc*100)
	}

	fmt.Println("=" + string(make([]byte, 70)))
	fmt.Println("✅ Training complete!")

	// Final evaluation
	finalLoss, finalAcc := validate(model, valBatches, backend)
	fmt.Printf("\n🎯 Final Validation Results:\n")
	fmt.Printf("   Loss: %.4f\n", finalLoss)
	fmt.Printf("   Accuracy: %.2f%%\n", finalAcc*100)

	if finalAcc >= 0.90 {
		fmt.Println("\n🎉 Success! Achieved >90% accuracy target!")
	} else {
		fmt.Printf("\n⚠️  Did not reach 90%% target (got %.2f%%)\n", finalAcc*100)
		fmt.Println("   Try increasing epochs or adjusting learning rate")
	}

	fmt.Println("\n📊 Framework Components Used:")
	fmt.Println("   ✓ Tensor API with type safety")
	fmt.Println("   ✓ CPU Backend with autodiff decorator")
	fmt.Println("   ✓ Automatic differentiation (full gradient tape)")
	fmt.Println("   ✓ NN Modules (Linear, ReLU)")
	fmt.Println("   ✓ CrossEntropyLoss with autodiff integration")
	fmt.Println("   ✓ Adam optimizer with bias correction")
	fmt.Println("   ✓ Real MNIST dataset (60,000 samples)")

	if *outPath != "" {
		fmt.Printf("   ✓ Writing model to %s\n", *outPath)
		err := nn.Save(model, *outPath, "MNISTNet", nil)
		if err != nil {
			log.Printf("Error saving model: %v\n", err)
		}
	}
}

// trainEpoch trains the model for one epoch.
func trainEpoch[B tensor.Backend](
	model *MNISTNet[*autodiff.Backend[B]],
	batches []*Batch[*autodiff.Backend[B]],
	optimizer optim.Optimizer,
	backend *autodiff.Backend[B],
) (avgLoss float32, accuracy float32) {
	totalLoss := float32(0.0)
	totalCorrect := 0
	totalSamples := 0

	for _, batch := range batches {
		// Zero gradients
		optimizer.ZeroGrad()

		// Forward pass
		logits := model.Forward(batch.ImagesTensor)

		// Compute loss using autodiff backend (records on tape)
		lossRaw := backend.CrossEntropy(
			logits.Raw(),
			batch.LabelsTensor.Raw(),
		)

		loss := tensor.New[float32, *autodiff.Backend[B]](lossRaw, backend)
		lossValue := loss.Raw().AsFloat32()[0]

		// Backward pass (automatic differentiation!)
		// Create output gradient of ones for scalar loss
		outputGrad, err := tensor.NewRaw(loss.Shape(), loss.DType(), backend.Device())
		if err != nil {
			panic(err)
		}
		outputGrad.AsFloat32()[0] = 1.0

		grads := backend.Tape().Backward(outputGrad, backend)

		// Update parameters
		optimizer.Step(grads)

		// Track metrics
		totalLoss += lossValue
		acc := nn.Accuracy(logits, batch.LabelsTensor)
		totalCorrect += int(acc * float32(batch.Size))
		totalSamples += batch.Size

		// Clear tape for next iteration
		backend.Tape().Clear()
	}

	avgLoss = totalLoss / float32(len(batches))
	accuracy = float32(totalCorrect) / float32(totalSamples)
	return avgLoss, accuracy
}

// validate evaluates the model on validation data.
func validate[B tensor.Backend](
	model *MNISTNet[*autodiff.Backend[B]],
	batches []*Batch[*autodiff.Backend[B]],
	backend *autodiff.Backend[B],
) (avgLoss float32, accuracy float32) {
	totalLoss := float32(0.0)
	totalCorrect := 0
	totalSamples := 0

	// Disable gradient recording for validation
	wasRecording := backend.Tape().IsRecording()
	backend.Tape().StopRecording()
	defer func() {
		if wasRecording {
			backend.Tape().StartRecording()
		}
	}()

	for _, batch := range batches {
		// Forward pass only (no gradients)
		logits := model.Forward(batch.ImagesTensor)

		// Compute loss
		lossRaw := backend.CrossEntropy(
			logits.Raw(),
			batch.LabelsTensor.Raw(),
		)
		lossValue := lossRaw.AsFloat32()[0]

		// Track metrics
		totalLoss += lossValue
		acc := nn.Accuracy(logits, batch.LabelsTensor)
		totalCorrect += int(acc * float32(batch.Size))
		totalSamples += batch.Size
	}

	avgLoss = totalLoss / float32(len(batches))
	accuracy = float32(totalCorrect) / float32(totalSamples)
	return avgLoss, accuracy
}

// countParameters counts the number of trainable parameters in the model.
func countParameters[B tensor.Backend](model *MNISTNet[*autodiff.Backend[B]]) int {
	total := 0
	for _, param := range model.Parameters() {
		shape := param.Tensor().Shape()
		count := 1
		for _, dim := range shape {
			count *= dim
		}
		total += count
	}
	return total
}
