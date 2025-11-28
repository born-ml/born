//go:build windows

// MNIST GPU Inference Benchmark
//
// This example demonstrates CPU vs WebGPU performance comparison for neural network inference.
// It creates a simple MLP model and measures forward pass times on both backends.
//
// Usage:
//
//	go run ./examples/mnist-gpu -batch 256 -iterations 100
//
// Note: This benchmark focuses on the compute-intensive operations (MatMul, ReLU).
// WebGPU shines for large matrix operations, while CPU may be faster for small batches.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/internal/backend/webgpu"
	"github.com/born-ml/born/internal/tensor"
)

func main() {
	// Parse command line arguments
	batchSize := flag.Int("batch", 64, "Batch size for inference")
	iterations := flag.Int("iterations", 50, "Number of iterations to run")
	warmup := flag.Int("warmup", 5, "Number of warmup iterations")
	flag.Parse()

	fmt.Println("Born ML Framework - MNIST GPU Inference Benchmark")
	fmt.Println("=" + string(make([]byte, 70)))
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Batch size: %d\n", *batchSize)
	fmt.Printf("  Iterations: %d\n", *iterations)
	fmt.Printf("  Warmup: %d\n\n", *warmup)

	// Check WebGPU availability
	if !webgpu.IsAvailable() {
		fmt.Println("WebGPU not available on this system!")
		fmt.Println("Ensure wgpu-native is installed and accessible.")
		return
	}

	// Create backends
	cpuBackend := cpu.New()
	gpuBackend, err := webgpu.New()
	if err != nil {
		fmt.Printf("Failed to create WebGPU backend: %v\n", err)
		return
	}
	defer gpuBackend.Release()

	fmt.Printf("GPU Backend: %s\n\n", gpuBackend.Name())

	// Create model weights (shared between CPU and GPU)
	// MLP: 784 -> 256 -> 128 -> 10
	fmt.Println("Creating model weights...")

	// Layer 1: 784 -> 256
	w1Data := randomWeights(784, 256)
	b1Data := randomBias(256)

	// Layer 2: 256 -> 128
	w2Data := randomWeights(256, 128)
	b2Data := randomBias(128)

	// Layer 3: 128 -> 10
	w3Data := randomWeights(128, 10)
	b3Data := randomBias(10)

	// Create input batch (random MNIST-like data)
	inputData := make([]float32, (*batchSize)*784)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	fmt.Println("Model architecture: 784 -> 256 -> 128 -> 10")
	totalParams := 784*256 + 256 + 256*128 + 128 + 128*10 + 10
	fmt.Printf("Total parameters: %d\n\n", totalParams)

	// ==========================================================================
	// CPU Benchmark
	// ==========================================================================
	fmt.Println("Running CPU benchmark...")

	// Create CPU tensors
	cpuInput := createTensor(tensor.Shape{*batchSize, 784}, inputData, tensor.CPU)
	cpuW1 := createTensor(tensor.Shape{784, 256}, w1Data, tensor.CPU)
	cpuB1 := createTensor(tensor.Shape{256}, b1Data, tensor.CPU)
	cpuW2 := createTensor(tensor.Shape{256, 128}, w2Data, tensor.CPU)
	cpuB2 := createTensor(tensor.Shape{128}, b2Data, tensor.CPU)
	cpuW3 := createTensor(tensor.Shape{128, 10}, w3Data, tensor.CPU)
	cpuB3 := createTensor(tensor.Shape{10}, b3Data, tensor.CPU)

	// Warmup
	for i := 0; i < *warmup; i++ {
		_ = forwardCPU(cpuBackend, cpuInput, cpuW1, cpuB1, cpuW2, cpuB2, cpuW3, cpuB3)
	}

	// Timed iterations
	cpuTimes := make([]time.Duration, *iterations)
	for i := 0; i < *iterations; i++ {
		start := time.Now()
		_ = forwardCPU(cpuBackend, cpuInput, cpuW1, cpuB1, cpuW2, cpuB2, cpuW3, cpuB3)
		cpuTimes[i] = time.Since(start)
	}

	cpuAvg := avgDuration(cpuTimes)
	cpuMin := minDuration(cpuTimes)
	cpuMax := maxDuration(cpuTimes)
	fmt.Printf("  CPU: avg=%.2fms, min=%.2fms, max=%.2fms\n",
		float64(cpuAvg.Microseconds())/1000,
		float64(cpuMin.Microseconds())/1000,
		float64(cpuMax.Microseconds())/1000)

	// ==========================================================================
	// GPU Benchmark
	// ==========================================================================
	fmt.Println("Running WebGPU benchmark...")

	// Create GPU tensors
	gpuInput := createTensor(tensor.Shape{*batchSize, 784}, inputData, tensor.WebGPU)
	gpuW1 := createTensor(tensor.Shape{784, 256}, w1Data, tensor.WebGPU)
	gpuB1 := createTensor(tensor.Shape{256}, b1Data, tensor.WebGPU)
	gpuW2 := createTensor(tensor.Shape{256, 128}, w2Data, tensor.WebGPU)
	gpuB2 := createTensor(tensor.Shape{128}, b2Data, tensor.WebGPU)
	gpuW3 := createTensor(tensor.Shape{128, 10}, w3Data, tensor.WebGPU)
	gpuB3 := createTensor(tensor.Shape{10}, b3Data, tensor.WebGPU)

	// Warmup
	for i := 0; i < *warmup; i++ {
		_ = forwardGPU(gpuBackend, gpuInput, gpuW1, gpuB1, gpuW2, gpuB2, gpuW3, gpuB3)
	}

	// Timed iterations
	gpuTimes := make([]time.Duration, *iterations)
	for i := 0; i < *iterations; i++ {
		start := time.Now()
		_ = forwardGPU(gpuBackend, gpuInput, gpuW1, gpuB1, gpuW2, gpuB2, gpuW3, gpuB3)
		gpuTimes[i] = time.Since(start)
	}

	gpuAvg := avgDuration(gpuTimes)
	gpuMin := minDuration(gpuTimes)
	gpuMax := maxDuration(gpuTimes)
	fmt.Printf("  GPU: avg=%.2fms, min=%.2fms, max=%.2fms\n",
		float64(gpuAvg.Microseconds())/1000,
		float64(gpuMin.Microseconds())/1000,
		float64(gpuMax.Microseconds())/1000)

	// ==========================================================================
	// Summary
	// ==========================================================================
	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("RESULTS SUMMARY")
	fmt.Println("=" + string(make([]byte, 70)))

	speedup := float64(cpuAvg) / float64(gpuAvg)
	fmt.Printf("\nBatch size: %d, Input: [%d, 784], Model: 784->256->128->10\n", *batchSize, *batchSize)
	fmt.Printf("\n%-20s %12s %12s %12s\n", "Backend", "Avg (ms)", "Min (ms)", "Max (ms)")
	fmt.Printf("%-20s %12.2f %12.2f %12.2f\n", "CPU",
		float64(cpuAvg.Microseconds())/1000,
		float64(cpuMin.Microseconds())/1000,
		float64(cpuMax.Microseconds())/1000)
	fmt.Printf("%-20s %12.2f %12.2f %12.2f\n", "WebGPU",
		float64(gpuAvg.Microseconds())/1000,
		float64(gpuMin.Microseconds())/1000,
		float64(gpuMax.Microseconds())/1000)

	fmt.Printf("\nSpeedup: %.2fx", speedup)
	if speedup > 1 {
		fmt.Println(" (GPU faster)")
	} else {
		fmt.Println(" (CPU faster)")
	}

	// Throughput calculation
	cpuThroughput := float64(*batchSize) / (float64(cpuAvg.Microseconds()) / 1e6)
	gpuThroughput := float64(*batchSize) / (float64(gpuAvg.Microseconds()) / 1e6)
	fmt.Printf("\nThroughput:\n")
	fmt.Printf("  CPU: %.0f samples/sec\n", cpuThroughput)
	fmt.Printf("  GPU: %.0f samples/sec\n", gpuThroughput)

	// Memory stats
	stats := gpuBackend.MemoryStats()
	fmt.Printf("\nGPU Memory:\n")
	fmt.Printf("  Total allocated: %d bytes\n", stats.TotalAllocatedBytes)
	fmt.Printf("  Peak memory: %d bytes\n", stats.PeakMemoryBytes)
	fmt.Printf("  Active buffers: %d\n", stats.ActiveBuffers)
	fmt.Printf("  Pool hits: %d, misses: %d\n", stats.PoolHits, stats.PoolMisses)

	fmt.Println("\nNote: GPU performance improves significantly with larger batch sizes")
	fmt.Println("      due to better parallelization of matrix operations.")
}

// forwardCPU performs forward pass on CPU backend and returns output tensor.
func forwardCPU(backend *cpu.Backend, input, w1, b1, w2, b2, w3, b3 *tensor.RawTensor) *tensor.RawTensor { //nolint:unparam // Output used in timing loop, not assigned to variable
	// Layer 1: input @ W1 + b1, then ReLU
	h1 := backend.MatMul(input, w1)
	h1 = addBias(backend, h1, b1)
	h1 = backend.ReLU(h1)

	// Layer 2: h1 @ W2 + b2, then ReLU
	h2 := backend.MatMul(h1, w2)
	h2 = addBias(backend, h2, b2)
	h2 = backend.ReLU(h2)

	// Layer 3: h2 @ W3 + b3, then Softmax
	output := backend.MatMul(h2, w3)
	output = addBias(backend, output, b3)
	output = backend.Softmax(output)

	return output
}

// forwardGPU performs forward pass on WebGPU backend and returns output tensor.
func forwardGPU(backend *webgpu.Backend, input, w1, b1, w2, b2, w3, b3 *tensor.RawTensor) *tensor.RawTensor { //nolint:unparam // Output used in timing loop, not assigned to variable
	// Layer 1: input @ W1 + b1, then ReLU
	h1 := backend.MatMul(input, w1)
	h1 = addBiasGPU(backend, h1, b1)
	h1 = backend.ReLU(h1)

	// Layer 2: h1 @ W2 + b2, then ReLU
	h2 := backend.MatMul(h1, w2)
	h2 = addBiasGPU(backend, h2, b2)
	h2 = backend.ReLU(h2)

	// Layer 3: h2 @ W3 + b3, then Softmax
	output := backend.MatMul(h2, w3)
	output = addBiasGPU(backend, output, b3)
	output = backend.Softmax(output)

	return output
}

// addBias broadcasts bias to each row and adds it (CPU version).
func addBias(backend *cpu.Backend, x, bias *tensor.RawTensor) *tensor.RawTensor {
	// x: [batch, features], bias: [features]
	batchSize := x.Shape()[0]
	features := x.Shape()[1]

	// Broadcast bias to [batch, features]
	broadcastBias, _ := tensor.NewRaw(tensor.Shape{batchSize, features}, tensor.Float32, tensor.CPU)
	biasData := bias.AsFloat32()
	broadcastData := broadcastBias.AsFloat32()
	for i := 0; i < batchSize; i++ {
		copy(broadcastData[i*features:(i+1)*features], biasData)
	}

	return backend.Add(x, broadcastBias)
}

// addBiasGPU broadcasts bias to each row and adds it (GPU version).
func addBiasGPU(backend *webgpu.Backend, x, bias *tensor.RawTensor) *tensor.RawTensor {
	// x: [batch, features], bias: [features]
	batchSize := x.Shape()[0]
	features := x.Shape()[1]

	// Broadcast bias to [batch, features]
	broadcastBias, _ := tensor.NewRaw(tensor.Shape{batchSize, features}, tensor.Float32, tensor.WebGPU)
	biasData := bias.AsFloat32()
	broadcastData := broadcastBias.AsFloat32()
	for i := 0; i < batchSize; i++ {
		copy(broadcastData[i*features:(i+1)*features], biasData)
	}

	return backend.Add(x, broadcastBias)
}

// Helper functions

func randomWeights(rows, cols int) []float32 {
	// Xavier initialization
	scale := float32(math.Sqrt(2.0 / float64(rows+cols)))
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = (rand.Float32()*2 - 1) * scale
	}
	return data
}

func randomBias(size int) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = (rand.Float32()*2 - 1) * 0.01
	}
	return data
}

func createTensor(shape tensor.Shape, data []float32, device tensor.Device) *tensor.RawTensor {
	raw, err := tensor.NewRaw(shape, tensor.Float32, device)
	if err != nil {
		panic(err)
	}
	byteData := raw.Data()
	for i, v := range data {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(byteData[i*4:(i+1)*4], bits)
	}
	return raw
}

func avgDuration(times []time.Duration) time.Duration {
	var total time.Duration
	for _, t := range times {
		total += t
	}
	return total / time.Duration(len(times))
}

func minDuration(times []time.Duration) time.Duration {
	min := times[0]
	for _, t := range times[1:] {
		if t < min {
			min = t
		}
	}
	return min
}

func maxDuration(times []time.Duration) time.Duration {
	max := times[0]
	for _, t := range times[1:] {
		if t > max {
			max = t
		}
	}
	return max
}
