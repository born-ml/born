//go:build amd64

package tensor

//go:generate sh -c "cd _gen/activation && go run . -out ../../sigmoid_simd_amd64.s -stubs ../../sigmoid_simd_stub_gen_amd64.go -pkg tensor"

import (
	"math"
	"os"

	"golang.org/x/sys/cpu"
)

// init wires the vendored AVX2+FMA sigmoid kernel into the dispatch, but only
// when BORN_EXPERIMENTAL_SIMD=1 and the CPU supports AVX2+FMA. Default builds
// (flag unset) keep the scalar path, so this is opt-in.
func init() {
	if os.Getenv("BORN_EXPERIMENTAL_SIMD") == "1" && cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		sigmoidF32 = sigmoidAVX2
	}
}

// sigmoidAVX2 applies the vendored 8-wide kernel to the bulk of in and finishes
// the sub-8 remainder with the scalar reference, so any length is handled.
func sigmoidAVX2(out, in []float32) {
	n := len(in)
	n8 := n &^ 7
	if n8 > 0 {
		sigmoidF32AVX2(out, in, n8)
	}
	for i := n8; i < n; i++ {
		out[i] = float32(1.0 / (1.0 + math.Exp(float64(-in[i]))))
	}
}
