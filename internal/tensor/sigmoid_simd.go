package tensor

// sigmoidF32 is the optional vendored-SIMD sigmoid: out[i] = 1/(1+exp(-in[i])).
//
// It is nil by default. An arch-specific build wires in a kernel only when BOTH
// the experimental env flag BORN_EXPERIMENTAL_SIMD is set AND the CPU supports
// the required instructions (see sigmoid_simd_amd64.go). When non-nil, Sigmoid
// (and the SiLU fast path) use it instead of the per-element scalar loop. out
// and in must have the same length.
var sigmoidF32 func(out, in []float32)
