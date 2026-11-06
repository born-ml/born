//go:build windows || linux

package webgpu

import (
	"fmt"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

// TestSumAcrossWorkgroups checks Sum at sizes on both sides of the CPU/GPU
// threshold and of the 256-wide workgroup boundary, in both evaluation modes.
// Sizes from gpuSumMinElements up go through the workgroup reduction kernels,
// or through the CPU fallback where the device can't run them correctly
// (gogpu/wgpu's software backend before v0.34.8 summed wrong, wgpu P1-SW-025).
func TestSumAcrossWorkgroups(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	b, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer b.Release()

	sizes := []int{1, 1023, 1024, 1025, 4096, 100000, 100001}
	for _, lazy := range []bool{false, true} {
		b.SetLazyMode(lazy)
		for _, dtype := range []tensor.DataType{tensor.Float32, tensor.Int32} {
			for _, n := range sizes {
				t.Run(fmt.Sprintf("lazy=%v/%s/n=%d", lazy, dtype, n), func(t *testing.T) {
					input, want := sumTestInput(t, dtype, n)
					out := b.Sum(input)
					var got int
					if dtype == tensor.Float32 {
						got = int(out.AsFloat32()[0])
					} else {
						got = int(out.AsInt32()[0])
					}
					if got != want {
						t.Errorf("Sum = %d, want %d", got, want)
					}
				})
			}
		}
	}
	if err := b.gpuSumCheck.err; err != nil {
		t.Logf("adapter %q: GPU sum kernels not used, Sum falls back to the CPU: %v", b.adapterInfo.Name, err)
	}
}

// sumTestInput returns n values in [-3, 3] and their exact sum. Small integers
// keep every partial sum exact in float32, so the test compares exactly.
func sumTestInput(t *testing.T, dtype tensor.DataType, n int) (*tensor.RawTensor, int) {
	t.Helper()
	input, err := tensor.NewRaw(tensor.Shape{n}, dtype, tensor.WebGPU)
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for i := range n {
		v := (i*5)%7 - 3
		want += v
		if dtype == tensor.Float32 {
			input.AsFloat32()[i] = float32(v)
		} else {
			input.AsInt32()[i] = int32(v)
		}
	}
	return input, want
}
