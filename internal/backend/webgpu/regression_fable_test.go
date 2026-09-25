//go:build windows || linux

package webgpu

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/optim"
	"github.com/born-ml/born/internal/tensor"
)

type adB = *autodiff.AutodiffBackend[*Backend]

func (b *Backend) liveCount() int {
	b.liveGPU.mu.Lock()
	defer b.liveGPU.mu.Unlock()
	return len(b.liveGPU.tensors)
}

// RV1: training loop — cache size, live GPU tensors, allocated bytes per step.
func TestRV_TrainingLoopMemory(t *testing.T) {
	gpu, err := New()
	if err != nil {
		t.Skip("no webgpu")
	}
	defer gpu.Release()
	ad := autodiff.New(gpu)

	const batch, in, classes = 8, 16, 10
	lin := nn.NewLinear[adB](in, classes, ad)
	opt := optim.NewAdam(lin.Parameters(), optim.AdamConfig{LR: 0.01, Betas: [2]float32{0.9, 0.999}}, ad)

	x := make([]float32, batch*in)
	y := make([]int32, batch)
	for i := range x {
		x[i] = float32(i%7) * 0.1
	}
	var cache, live []int
	var alloc []uint64
	for step := 0; step < 8; step++ {
		input, _ := tensor.FromSlice(x, tensor.Shape{batch, in}, ad)
		targets, _ := tensor.FromSlice(y, tensor.Shape{batch}, ad)
		ad.Tape().StartRecording()
		logits := lin.Forward(input)
		loss := tensor.New[float32, adB](ad.CrossEntropy(logits.Raw(), targets.Raw()), ad)
		_ = loss.Raw().AsFloat32()[0]
		grads := autodiff.Backward(loss, ad)
		opt.Step(grads)
		autodiff.ReleaseGradients(grads, ad)
		ad.ClearTape()
		cache = append(cache, gpu.inputBufferCacheSize())
		live = append(live, gpu.liveCount())
		alloc = append(alloc, gpu.MemoryStats().TotalAllocatedBytes)
	}
	t.Logf("inputBufferCache per step: %v", cache)
	t.Logf("live LazyGPUData per step:  %v", live)
	t.Logf("TotalAllocatedBytes/step:   %v", alloc)
	if cache[7] != cache[2] {
		t.Errorf("cache still grows: %v", cache)
	}
	if live[7] > live[2] {
		t.Errorf("live GPU tensors grow across steps: %v (leak — likely Detach/SetTensor refcount)", live)
	}
}

// RV2: diamond graph that panicked before F3.
func TestRV_DiamondGraph(t *testing.T) {
	gpu, err := New()
	if err != nil {
		t.Skip("no webgpu")
	}
	defer gpu.Release()
	ad := autodiff.New(gpu)

	a, _ := tensor.FromSlice([]float32{1, 2, 3, 4}, tensor.Shape{4}, ad)
	a.RequireGrad()
	ad.Tape().StartRecording()
	aG := tensor.New[float32, adB](ad.MulScalar(a.Raw(), float32(1)), ad)
	b := tensor.New[float32, adB](ad.Exp(aG.Raw()), ad)
	k := tensor.New[float32, adB](ad.MulScalar(aG.Raw(), float32(2)), ad)
	h := tensor.New[float32, adB](ad.Add(aG.Raw(), b.Raw()), ad)
	s := tensor.New[float32, adB](ad.Add(h.Raw(), k.Raw()), ad)
	loss := tensor.New[float32, adB](ad.SumDim(s.Raw(), 0, false), ad)
	grads := autodiff.Backward(loss, ad)
	got := grads[a.Raw()].AsFloat32()
	for i, v := range []float32{1, 2, 3, 4} {
		want := 3 + float32(math.Exp(float64(v)))
		if math.Abs(float64(got[i]-want)) > 1e-3 {
			t.Fatalf("grad[%d]=%v want %v", i, got[i], want)
		}
	}
	t.Logf("diamond grad OK: %v", got)
}

// RV3: Conv2D training on GPU — F9 CPU fallback. Two conv layers so the
// first conv's output is both a Conv2D input and a ReLU output (shared
// tensor used by two backward ops after materializeForCPU).
func TestRV_Conv2DBackwardGPU(t *testing.T) {
	gpu, err := New()
	if err != nil {
		t.Skip("no webgpu")
	}
	defer gpu.Release()
	ad := autodiff.New(gpu)

	c1 := nn.NewConv2D[adB](1, 2, 3, 3, 1, 1, true, ad)
	c2 := nn.NewConv2D[adB](2, 1, 3, 3, 1, 1, true, ad)
	params := append(c1.Parameters(), c2.Parameters()...)
	opt := optim.NewSGD(params, optim.SGDConfig{LR: 0.01}, ad)

	xd := make([]float32, 1*1*6*6)
	for i := range xd {
		xd[i] = float32(i%5) * 0.2
	}
	for step := 0; step < 3; step++ {
		x, _ := tensor.FromSlice(xd, tensor.Shape{1, 1, 6, 6}, ad)
		ad.Tape().StartRecording()
		h := c1.Forward(x)
		hr := tensor.New[float32, adB](ad.ReLU(h.Raw()), ad)
		out := c2.Forward(hr)
		loss := tensor.New[float32, adB](ad.Sum(out.Raw()), ad)
		grads := autodiff.Backward(loss, ad)
		for _, p := range params {
			g, ok := grads[p.Tensor().Raw()]
			if !ok {
				t.Fatalf("step %d: param %s has no gradient", step, p.Name())
			}
			for _, v := range g.AsFloat32() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("step %d: NaN/Inf gradient in %s", step, p.Name())
				}
			}
		}
		opt.Step(grads)
		ad.ClearTape()
	}
	t.Log("Conv2D→ReLU→Conv2D trains on software adapter without panic")
}

// RV4: inference-style loop — no ClearTape/ReclaimMemory ever called.
func TestRV_InferenceCacheGrowth(t *testing.T) {
	gpu, err := New()
	if err != nil {
		t.Skip("no webgpu")
	}
	defer gpu.Release()
	lin := nn.NewLinear[*Backend](4, 2, gpu)
	var sizes []int
	for i := 0; i < 6; i++ {
		x, _ := tensor.FromSlice([]float32{1, 2, 3, 4}, tensor.Shape{1, 4}, gpu)
		_ = lin.Forward(x).Data()
		sizes = append(sizes, gpu.inputBufferCacheSize())
	}
	t.Logf("inference loop cache entries: %v (grows = still P1 for inference)", sizes)
}
