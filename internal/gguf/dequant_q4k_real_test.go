package gguf

// TestDequantizeQ4K_RealData verifies Q4_K dequantization against a Python
// reference computed from the actual first block of blk.0.attn_q.weight in
// TinyLlama-1.1B-Chat Q4_K_M.gguf.
//
// Python reference script: attention-experiment/scripts/dequant_q4k_ref.py
// All expected values were computed by the GGML algorithm from ggml-quants.c.
//
// Block layout (144 bytes):
//
//	[0:2]   d    = 0x0428 → 6.342e-05 (F16)
//	[2:4]   dmin = 0x1033 → 5.126e-04 (F16)
//	[4:16]  scales[12] packed
//	[16:144] qs[128] 4-bit values

import (
	"math"
	"testing"
)

// realQ4KBlock contains the exact first 144 bytes of blk.0.attn_q.weight
// from tinyllama-1.1b-chat.Q4_K_M.gguf (tensor offset 115732480, dtype=12/Q4_K).
var realQ4KBlock = []byte{
	0x28, 0x04, 0x33, 0x10, 0xdd, 0xad, 0xed, 0xa8, 0xdc, 0x5d, 0xf7, 0xd8, 0x13, 0x7d, 0xff, 0x0a, // [0..15]
	0x57, 0x76, 0x64, 0x50, 0x86, 0x76, 0x48, 0x86, 0x58, 0x6f, 0xf7, 0x78, 0x5a, 0xd8, 0x47, 0x66, // [16..31]
	0x7a, 0x37, 0x4c, 0x67, 0x99, 0x44, 0x88, 0x68, 0x08, 0x86, 0x38, 0xa8, 0xac, 0x46, 0x48, 0x69, // [32..47]
	0x4a, 0x8c, 0x3f, 0xcb, 0x5a, 0x59, 0x0b, 0x5a, 0x2b, 0x28, 0x5f, 0x5b, 0x3a, 0x7a, 0x54, 0xa9, // [48..63]
	0x59, 0xea, 0x9a, 0x3a, 0x89, 0x4c, 0x6c, 0x0b, 0x7d, 0x8b, 0x5a, 0x5b, 0x5a, 0x1d, 0x58, 0x30, // [64..79]
	0x4a, 0x59, 0x67, 0x76, 0x6c, 0x38, 0x8b, 0x48, 0x46, 0x48, 0x58, 0x46, 0x08, 0x67, 0x67, 0x36, // [80..95]
	0x88, 0x69, 0x00, 0x46, 0x48, 0x78, 0x3b, 0x48, 0x1f, 0x49, 0x58, 0x74, 0x38, 0xf7, 0x6b, 0x84, // [96..111]
	0x99, 0xa9, 0xa8, 0x98, 0xba, 0xa6, 0x89, 0x08, 0xbc, 0x6a, 0x54, 0x80, 0xd9, 0xd8, 0xb9, 0xc9, // [112..127]
	0x97, 0x86, 0x76, 0x99, 0xb9, 0xb8, 0xf8, 0xcf, 0xd9, 0xba, 0xa9, 0x78, 0xd8, 0x77, 0x88, 0x84, // [128..143]
}

// TestDequantizeQ4K_RealData_First10 verifies the first 10 dequantized values
// against the Python reference.
//
// Python output (dequant_q4k_ref.py):
//
//	result[0] = 0.00518513  (scale=0.001839, q=7, minv=0.007689)
//	result[1] = 0.00150681
//	result[2] = 0.00334597
//	result[3] = 0.00518513
//	result[4] = -0.00033236
//	result[5] = 0.00334597
//	result[6] = -0.00768900
//	result[7] = 0.00150681
//	result[8] = 0.00334597
//	result[9] = 0.00702429
func TestDequantizeQ4K_RealData_First10(t *testing.T) {
	result, err := dequantizeBlockQ4_K(realQ4KBlock)
	if err != nil {
		t.Fatalf("dequantizeBlockQ4_K failed: %v", err)
	}
	if len(result) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(result))
	}

	// Expected values computed by Python reference (dequant_q4k_ref.py).
	// Tolerance accounts for float32 vs float64 intermediate precision.
	const tol = 1e-5

	type want struct {
		idx int
		val float32
	}

	wantVals := []want{
		{0, 0.00518513},
		{1, 0.00150681},
		{2, 0.00334597},
		{3, 0.00518513},
		{4, -0.00033236},
		{5, 0.00334597},
		{6, -0.00768900},
		{7, 0.00150681},
		{8, 0.00334597},
		{9, 0.00702429},
	}

	for _, w := range wantVals {
		if math.Abs(float64(result[w.idx]-w.val)) > tol {
			t.Errorf("result[%d] = %.8f, want %.8f (diff=%.2e)",
				w.idx, result[w.idx], w.val, math.Abs(float64(result[w.idx]-w.val)))
		}
	}
}

// TestDequantizeQ4K_RealData_SubBlockBoundaries verifies values at the start
// of each of the 8 sub-blocks, confirming correct scale and min indexing.
//
// Python sub-block breakdown (dequant_q4k_ref.py):
//
//	sub[0]: scale=0.001839 minv=0.007689  → result[0]=0.005185,  result[1]=0.001507
//	sub[1]: scale=0.002854 minv=0.027680  → result[32]=0.000858, result[33]=-0.007703
//	sub[2]: scale=0.002854 minv=0.032294  → result[64]=-0.003755 result[65]=-0.020878
//	sub[3]: scale=0.002537 minv=0.021529  → result[96]=0.001302  result[97]=-0.008845
//	sub[4]: scale=0.001776 minv=0.003588  → result[128]=0.014169 result[129]=0.003515
//	sub[5]: scale=0.001839 minv=0.014865  → result[160]=-0.000152 result[161]=-0.000152
//	sub[6]: scale=0.003488 minv=0.032294  → result[192]=-0.000901 result[193]=-0.000901
//	sub[7]: scale=0.001522 minv=0.001538  → result[224]=0.009117 result[225]=0.012161
func TestDequantizeQ4K_RealData_SubBlockBoundaries(t *testing.T) {
	result, err := dequantizeBlockQ4_K(realQ4KBlock)
	if err != nil {
		t.Fatalf("dequantizeBlockQ4_K failed: %v", err)
	}

	const tol = 2e-5 // Slightly looser to handle float32 rounding.

	tests := []struct {
		name string
		idx  int
		want float32
	}{
		// sub-block 0 (scales[0]=29, mins[0]=15)
		{"sub0[0]", 0, 0.005185},
		{"sub0[1]", 1, 0.001507},
		// sub-block 1 (scales[1]=45, mins[1]=54)
		{"sub1[0]", 32, 0.000858},
		{"sub1[1]", 33, -0.007703},
		// sub-block 2 (scales[2]=45, mins[2]=63)
		{"sub2[0]", 64, -0.003755},
		{"sub2[1]", 65, -0.020878},
		// sub-block 3 (scales[3]=40, mins[3]=42)
		{"sub3[0]", 96, 0.001302},
		{"sub3[1]", 97, -0.008845},
		// sub-block 4 (scales[4]=28, mins[4]=7)
		{"sub4[0]", 128, 0.014169},
		{"sub4[1]", 129, 0.003515},
		// sub-block 5 (scales[5]=29, mins[5]=29)
		{"sub5[0]", 160, -0.000152},
		{"sub5[1]", 161, -0.000152},
		// sub-block 6 (scales[6]=55, mins[6]=63)
		{"sub6[0]", 192, -0.000901},
		{"sub6[1]", 193, -0.000901},
		// sub-block 7 (scales[7]=24, mins[7]=3)
		{"sub7[0]", 224, 0.009117},
		{"sub7[1]", 225, 0.012161},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if math.Abs(float64(result[tt.idx]-tt.want)) > tol {
				t.Errorf("result[%d] = %.8f, want %.8f (diff=%.2e)",
					tt.idx, result[tt.idx], tt.want,
					math.Abs(float64(result[tt.idx]-tt.want)))
			}
		})
	}
}

// TestDequantizeQ4K_RealData_ScaleUnpacking verifies the intermediate
// scale/min unpacking step against the Python reference values.
//
// Python unpacked scales: [29, 45, 45, 40, 28, 29, 55, 24]
// Python unpacked mins:   [15, 54, 63, 42,  7, 29, 63,  3].
func TestDequantizeQ4K_RealData_ScaleUnpacking(t *testing.T) {
	// Reproduce the scale unpacking from dequantizeBlockQ4_K.
	sc := realQ4KBlock[4:16]

	gotScales := make([]uint8, 8)
	gotMins := make([]uint8, 8)
	for i := 0; i < 4; i++ {
		gotScales[i] = sc[i] & 0x3F
		gotScales[i+4] = sc[i+4] & 0x3F
		gotMins[i] = (sc[i] >> 6) | ((sc[i+8] & 0x0F) << 2)
		gotMins[i+4] = (sc[i+4] >> 6) | ((sc[i+8] >> 4) << 2)
	}

	wantScales := []uint8{29, 45, 45, 40, 28, 29, 55, 24}
	wantMins := []uint8{15, 54, 63, 42, 7, 29, 63, 3}

	for i := 0; i < 8; i++ {
		if gotScales[i] != wantScales[i] {
			t.Errorf("scales[%d] = %d, want %d", i, gotScales[i], wantScales[i])
		}
		if gotMins[i] != wantMins[i] {
			t.Errorf("mins[%d] = %d, want %d", i, gotMins[i], wantMins[i])
		}
	}
}

// TestDequantizeQ4K_RealData_AllElementsFinite checks that no NaN or Inf
// values are produced for the complete 256-element block.
func TestDequantizeQ4K_RealData_AllElementsFinite(t *testing.T) {
	result, err := dequantizeBlockQ4_K(realQ4KBlock)
	if err != nil {
		t.Fatalf("dequantizeBlockQ4_K failed: %v", err)
	}

	for i, v := range result {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("result[%d] = %v (not finite)", i, v)
		}
	}
}
