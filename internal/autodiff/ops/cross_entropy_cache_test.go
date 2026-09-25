package ops

import (
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

// TestOneHotIdentityCache verifies that getOrBuildOneHotIdentity returns the same
// *RawTensor pointer for repeated calls with the same (n, dtype) key, and a
// different pointer when the key differs.
func TestOneHotIdentityCache(t *testing.T) {
	// Same (n, dtype) must return the same pointer (cache hit).
	a := getOrBuildOneHotIdentity(10, tensor.Float32, tensor.CPU)
	b := getOrBuildOneHotIdentity(10, tensor.Float32, tensor.CPU)

	if a != b {
		t.Error("expected cache hit for same (n, dtype): got different pointers")
	}

	// Different n must return a different pointer (cache miss).
	c := getOrBuildOneHotIdentity(20, tensor.Float32, tensor.CPU)
	if a == c {
		t.Error("expected cache miss for different n: got same pointer")
	}

	// Different dtype must return a different pointer (cache miss).
	d := getOrBuildOneHotIdentity(10, tensor.Float64, tensor.CPU)
	if a == d {
		t.Error("expected cache miss for different dtype: got same pointer")
	}

	// Third call for the first key still hits cache.
	e := getOrBuildOneHotIdentity(10, tensor.Float32, tensor.CPU)
	if a != e {
		t.Error("expected cache hit on third call for same (n, dtype): got different pointer")
	}
}
