//go:build windows

// GPU Training Example — demonstrates Born GPU training loop.
//
// A 3-layer MLP trained on synthetic classification data.
// Validates that GPU training works end-to-end: forward → backward → optimizer → repeat.
//
// Usage:
//
//	go run ./examples/gpu-training
//	go run ./examples/gpu-training -steps 50 -batch 128 -hidden 512
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"time"

	"github.com/born-ml/born/autodiff"
	"github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/nn"
	"github.com/born-ml/born/optim"
	"github.com/born-ml/born/tensor"
)

type B = *autodiff.Backend[*webgpu.Backend]

func main() {
	steps := flag.Int("steps", 30, "Number of training steps")
	batchSize := flag.Int("batch", 64, "Batch size")
	inputDim := flag.Int("input", 256, "Input dimension")
	hiddenDim := flag.Int("hidden", 256, "Hidden dimension")
	outputDim := flag.Int("output", 10, "Number of classes")
	lr := flag.Float64("lr", 0.001, "Learning rate")
	flag.Parse()

	fmt.Println("Born ML — GPU Training Example")
	fmt.Println("=" + string(make([]byte, 50)))

	if !webgpu.IsAvailable() {
		fmt.Println("WebGPU not available!")
		os.Exit(1)
	}

	gpu, err := webgpu.New()
	if err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Release()

	backend := autodiff.New(gpu)
	fmt.Printf("Backend: %s\n", backend.Name())
	fmt.Printf("Model: %d → %d → %d → %d\n", *inputDim, *hiddenDim, *hiddenDim, *outputDim)
	fmt.Printf("Batch: %d, Steps: %d, LR: %g\n\n", *batchSize, *steps, *lr)

	linear1 := nn.NewLinear[B](*inputDim, *hiddenDim, backend)
	linear2 := nn.NewLinear[B](*hiddenDim, *hiddenDim, backend)
	linear3 := nn.NewLinear[B](*hiddenDim, *outputDim, backend)

	params := make([]*nn.Parameter[B], 0, len(linear1.Parameters())+len(linear2.Parameters())+len(linear3.Parameters()))
	params = append(params, linear1.Parameters()...)
	params = append(params, linear2.Parameters()...)
	params = append(params, linear3.Parameters()...)

	totalParams := 0
	for _, p := range params {
		n := 1
		for _, d := range p.Tensor().Shape() {
			n *= d
		}
		totalParams += n
	}
	fmt.Printf("Parameters: %d\n\n", totalParams)

	optimizer := optim.NewAdam(params, optim.AdamConfig{
		LR:    float32(*lr),
		Betas: [2]float32{0.9, 0.999},
	}, backend)

	rng := rand.New(rand.NewSource(42))
	start := time.Now()

	for step := 0; step < *steps; step++ {
		inputData := make([]float32, (*batchSize)*(*inputDim))
		for i := range inputData {
			inputData[i] = rng.Float32()*2 - 1
		}
		targetData := make([]int32, *batchSize)
		for i := range targetData {
			targetData[i] = rng.Int31n(int32(*outputDim))
		}

		input, _ := tensor.FromSlice(inputData, tensor.Shape{*batchSize, *inputDim}, backend)
		targets, _ := tensor.FromSlice(targetData, tensor.Shape{*batchSize}, backend)

		backend.Tape().StartRecording()

		h1 := linear1.Forward(input)
		a1 := tensor.New[float32, B](backend.ReLU(h1.Raw()), backend)
		h2 := linear2.Forward(a1)
		a2 := tensor.New[float32, B](backend.ReLU(h2.Raw()), backend)
		logits := linear3.Forward(a2)

		lossRaw := backend.CrossEntropy(logits.Raw(), targets.Raw())
		lossVal := lossRaw.AsFloat32()[0]

		loss := tensor.New[float32, B](lossRaw, backend)

		grads := autodiff.Backward(loss, backend)
		optimizer.Step(grads)
		autodiff.ReleaseGradients(grads)
		backend.ClearTape()

		if step%5 == 0 || step == *steps-1 {
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			elapsed := time.Since(start)
			stepsPerSec := float64(step+1) / elapsed.Seconds()
			fmt.Printf("Step %3d/%d: loss=%.4f  heap=%dMB  %.1f steps/sec\n",
				step, *steps, lossVal, ms.HeapAlloc/1024/1024, stepsPerSec)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\nCompleted %d steps in %v (%.1f steps/sec)\n",
		*steps, elapsed.Round(time.Millisecond), float64(*steps)/elapsed.Seconds())
	fmt.Printf("Expected random loss: %.2f\n", -math.Log(1.0/float64(*outputDim)))
}
