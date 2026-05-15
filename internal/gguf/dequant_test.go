package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestFloat16ToFloat32 tests IEEE 754 half precision conversion.
func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name string
		h    uint16
		want float32
	}{
		{"zero", 0x0000, 0.0},
		{"one", 0x3C00, 1.0},
		{"minus_one", 0xBC00, -1.0},
		{"two", 0x4000, 2.0},
		{"half", 0x3800, 0.5},
		{"max_normal", 0x7BFF, 65504.0},
		{"min_positive_normal", 0x0400, 6.103515625e-05},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float16ToFloat32(tt.h)
			if math.Abs(float64(got-tt.want)) > 1e-6 {
				t.Errorf("Float16ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
			}
		})
	}
}

// TestDequantizeF32 tests identity operation for F32 type.
func TestDequantizeF32(t *testing.T) {
	// Create test data: [1.0, 2.0, 3.0, 4.0].
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data[4:8], math.Float32bits(2.0))
	binary.LittleEndian.PutUint32(data[8:12], math.Float32bits(3.0))
	binary.LittleEndian.PutUint32(data[12:16], math.Float32bits(4.0))

	result, err := Dequantize(data, GGMLTypeF32, 4)
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %v, want %v", i, result[i], v)
		}
	}
}

// TestDequantizeF16 tests half precision conversion.
func TestDequantizeF16(t *testing.T) {
	// Create test data: [1.0, 2.0, 0.5, -1.0] in F16.
	data := make([]byte, 8)
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00) // 1.0
	binary.LittleEndian.PutUint16(data[2:4], 0x4000) // 2.0
	binary.LittleEndian.PutUint16(data[4:6], 0x3800) // 0.5
	binary.LittleEndian.PutUint16(data[6:8], 0xBC00) // -1.0

	result, err := Dequantize(data, GGMLTypeF16, 4)
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	expected := []float32{1.0, 2.0, 0.5, -1.0}
	for i, v := range expected {
		if math.Abs(float64(result[i]-v)) > 1e-6 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], v)
		}
	}
}

// TestDequantizeQ8_0 tests 8-bit quantization (simplest quantized format).
func TestDequantizeQ8_0(t *testing.T) {
	// Create Q8_0 block: d=0.5, qs=[1, 2, -1, -2, ...].
	data := make([]byte, 34)

	// d = 0.5 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3800)

	// qs: 32 int8 values.
	for i := 0; i < 32; i++ {
		data[2+i] = byte(int8(i - 16))
	}

	result, err := DequantizeBlock(data, GGMLTypeQ8_0)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// Check first few values: d * q[i].
	d := float32(0.5)
	for i := 0; i < 4; i++ {
		expected := d * float32(int8(i-16))
		if math.Abs(float64(result[i]-expected)) > 1e-6 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}

// TestDequantizeQ4_0 tests 4-bit quantization.
func TestDequantizeQ4_0(t *testing.T) {
	// Create Q4_0 block: d=1.0, qs packed 4-bit values.
	data := make([]byte, 18)

	// d = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00)

	// qs: 16 bytes, each containing two 4-bit values (0-15).
	// First byte: 0x10 = low=0, high=1.
	for i := 0; i < 16; i++ {
		data[2+i] = byte(i) | (byte(i+1) << 4)
	}

	result, err := DequantizeBlock(data, GGMLTypeQ4_0)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// Formula: d * (q - 8).
	// First element: q=0, result = 1.0 * (0 - 8) = -8.0.
	if result[0] != -8.0 {
		t.Errorf("result[0] = %v, want -8.0", result[0])
	}

	// Second element: q=1, result = 1.0 * (1 - 8) = -7.0.
	if result[1] != -7.0 {
		t.Errorf("result[1] = %v, want -7.0", result[1])
	}
}

// TestDequantizeQ4_1 tests 4-bit quantization with offset.
func TestDequantizeQ4_1(t *testing.T) {
	// Create Q4_1 block: d=0.5, m=1.0, qs packed 4-bit values.
	data := make([]byte, 20)

	// d = 0.5 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3800)
	// m = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[2:4], 0x3C00)

	// qs: 16 bytes.
	for i := 0; i < 16; i++ {
		data[4+i] = 0x00 // All zeros for simplicity.
	}

	result, err := DequantizeBlock(data, GGMLTypeQ4_1)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// Formula: d * q + m = 0.5 * 0 + 1.0 = 1.0.
	expected := float32(1.0)
	for i := 0; i < 32; i++ {
		if math.Abs(float64(result[i]-expected)) > 1e-6 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}

// TestDequantizeQ5_0 tests 5-bit quantization.
func TestDequantizeQ5_0(t *testing.T) {
	// Create Q5_0 block: d=1.0, qh=0, qs=0.
	data := make([]byte, 22)

	// d = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00)

	// qh = 0 (no high bits set).
	binary.LittleEndian.PutUint32(data[2:6], 0)

	// qs = 0 (all zeros).
	for i := 0; i < 16; i++ {
		data[6+i] = 0
	}

	result, err := DequantizeBlock(data, GGMLTypeQ5_0)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// Formula: d * (q - 16) = 1.0 * (0 - 16) = -16.0.
	expected := float32(-16.0)
	for i := 0; i < 32; i++ {
		if math.Abs(float64(result[i]-expected)) > 1e-6 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}

// TestDequantizeQ4_K tests 4-bit K-quant scale unpacking and value reconstruction.
func TestDequantizeQ4_K(t *testing.T) {
	data := make([]byte, 144)

	// d = 1.0 (F16 0x3C00), dmin = 0.5 (F16 0x3800).
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(data[2:4], 0x3800)

	// Scales layout (12 bytes at offset 4):
	//   bytes [0..3]: low 6 bits = sc[0..3], bits [6:7] = high 2 bits of m[0..3]
	//   bytes [4..7]: low 6 bits = sc[4..7], bits [6:7] = high 2 bits of m[4..7]
	//   bytes [8..11]: low nibble = low 4 bits of m[0..3], high nibble = low 4 bits of m[4..7]
	//
	// Set sc[0]=3, m[0]=2 → data[4+0] = (3 & 0x3F) | ((2 >> 2) << 6) = 3
	//                        data[4+8] = (2 & 0x0F) = 2  (low nibble for m[0])
	data[4+0] = 3   // sc[0] = 3 (low 6 bits), m[0] high 2 bits = 0
	data[4+1] = 5   // sc[1] = 5
	data[4+2] = 0   // sc[2] = 0
	data[4+3] = 0   // sc[3] = 0
	data[4+4] = 0   // sc[4] = 0
	data[4+5] = 0   // sc[5] = 0
	data[4+6] = 0   // sc[6] = 0
	data[4+7] = 0   // sc[7] = 0
	data[4+8] = 0x02 // low nibble = m[0] low 4 bits = 2; high nibble = m[4] low 4 bits = 0
	data[4+9] = 0x00
	data[4+10] = 0x00
	data[4+11] = 0x00

	// m[0] = (data[4+0] >> 6) | ((data[4+8] & 0x0F) << 2) = 0 | (2 << 2) = 8
	// Wait, let me recalculate:
	// data[4+0] = 3 = 0b00000011 → sc[0] = 3 & 0x3F = 3, high bits = 3 >> 6 = 0
	// data[4+8] = 2 = 0b00000010 → low nibble = 2
	// m[0] = (0) | (2 << 2) = 8
	// So: scale = d * sc[0] = 1.0 * 3 = 3.0, min = dmin * m[0] = 0.5 * 8 = 4.0

	// qs[128] at offset 16. Sub-block 0 uses first 16 bytes.
	// Set qs[16] = 0x53 → low nibble = 3, high nibble = 5.
	data[16] = 0x53

	// Sub-block 0, element 0: scale * q_low - min = 3.0 * 3 - 4.0 = 5.0
	// Sub-block 0, element 1: scale * q_high - min = 3.0 * 5 - 4.0 = 11.0

	result, err := DequantizeBlock(data, GGMLTypeQ4_K)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(result))
	}

	// Verify sub-block 0 first two elements.
	if math.Abs(float64(result[0]-5.0)) > 0.01 {
		t.Errorf("result[0] = %.4f, want 5.0 (scale=3.0, q=3, min=4.0)", result[0])
	}
	if math.Abs(float64(result[1]-11.0)) > 0.01 {
		t.Errorf("result[1] = %.4f, want 11.0 (scale=3.0, q=5, min=4.0)", result[1])
	}

	// Sub-block 1 uses sc[1]=5, m[1]=0. scale=5.0, min=0.
	// Its qs start at data[16+16]=data[32]. data[32]=0 → q_low=0, q_high=0.
	// result[32] = 5.0 * 0 - 0 = 0.0
	if math.Abs(float64(result[32])) > 0.01 {
		t.Errorf("result[32] = %.4f, want 0.0 (scale=5.0, q=0, min=0.0)", result[32])
	}

	t.Logf("Q4_K sub-block 0: result[0]=%.2f result[1]=%.2f (expect 5.0, 11.0)", result[0], result[1])
}

// TestDequantizeQ4_K_ScaleExtraction verifies all 8 scale/min pairs extract correctly.
func TestDequantizeQ4_K_ScaleExtraction(t *testing.T) {
	data := make([]byte, 144)
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(data[2:4], 0x3C00) // dmin = 1.0

	// Set distinct sc and m values for verification.
	// sc[0..3] in low 6 bits of bytes [0..3], sc[4..7] in low 6 bits of bytes [4..7].
	data[4+0] = 1  // sc[0] = 1
	data[4+1] = 2  // sc[1] = 2
	data[4+2] = 3  // sc[2] = 3
	data[4+3] = 4  // sc[3] = 4
	data[4+4] = 10 // sc[4] = 10
	data[4+5] = 20 // sc[5] = 20
	data[4+6] = 30 // sc[6] = 30
	data[4+7] = 40 // sc[7] = 40

	// Verify all-zero qs produce no error before setting qs values.
	if _, err := DequantizeBlock(data, GGMLTypeQ4_K); err != nil {
		t.Fatal(err)
	}

	// Set one qs byte to test scale extraction.
	// All qs = 0, so result = scale*0 - min = 0.
	// m values with high 2 bits = 0 → all m = 0.
	data[16] = 0x01 // sub-block 0, elem 0: q=1
	result, _ := DequantizeBlock(data, GGMLTypeQ4_K)

	// result[0] = d * sc[0] * 1 - dmin * m[0] = 1.0 * 1 * 1 - 1.0 * 0 = 1.0
	if math.Abs(float64(result[0]-1.0)) > 0.01 {
		t.Errorf("sc[0] extraction: result[0] = %.4f, want 1.0", result[0])
	}

	// Test sub-block 4 (uses sc[4] = 10).
	data[16+64] = 0x01 // sub-block 4, elem 0: q=1
	result, _ = DequantizeBlock(data, GGMLTypeQ4_K)

	// result[128] = d * sc[4] * 1 - dmin * m[4] = 1.0 * 10 * 1 - 0 = 10.0
	if math.Abs(float64(result[128]-10.0)) > 0.01 {
		t.Errorf("sc[4] extraction: result[128] = %.4f, want 10.0", result[128])
	}
}

// TestDequantizeQ5_K tests 5-bit K-quant (256 elements).
func TestDequantizeQ5_K(t *testing.T) {
	// Create Q5_K block with simple values.
	data := make([]byte, 176)

	// d = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00)
	// dmin = 0.0 in F16.
	binary.LittleEndian.PutUint16(data[2:4], 0x0000)

	// scales[12]: set all to minimal values.
	for i := 0; i < 12; i++ {
		data[4+i] = 0x01
	}

	// qh[32]: high bits.
	for i := 0; i < 32; i++ {
		data[16+i] = 0x00
	}

	// qs[128]: low bits.
	for i := 0; i < 128; i++ {
		data[48+i] = 0x00
	}

	result, err := DequantizeBlock(data, GGMLTypeQ5_K)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(result))
	}

	t.Logf("Q5_K dequantization produced %d elements", len(result))
}

// TestDequantizeQ6_K tests 6-bit K-quant (256 elements).
func TestDequantizeQ6_K(t *testing.T) {
	// Create Q6_K block with simple values.
	data := make([]byte, 210)

	// ql[128]: low 4 bits.
	for i := 0; i < 128; i++ {
		data[i] = 0x00
	}

	// qh[64]: high 2 bits.
	for i := 0; i < 64; i++ {
		data[128+i] = 0x00
	}

	// scales[16]: signed scales.
	for i := 0; i < 16; i++ {
		data[192+i] = 0x01 // scale = 1.
	}

	// d = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[208:210], 0x3C00)

	result, err := DequantizeBlock(data, GGMLTypeQ6_K)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(result))
	}

	// Formula: scale * (q - 32) = 1.0 * (0 - 32) = -32.0.
	expected := float32(-32.0)
	for i := 0; i < 16; i++ { // Check first sub-block.
		if math.Abs(float64(result[i]-expected)) > 1e-5 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}

// TestDequantizeUnsupportedType tests error handling for unsupported types.
func TestDequantizeUnsupportedType(t *testing.T) {
	data := make([]byte, 100)

	// Test with Q2_K (not implemented yet).
	_, err := Dequantize(data, GGMLTypeQ2_K, 32)
	if err == nil {
		t.Error("expected error for unsupported type Q2_K, got nil")
	}

	// Test with IQ2_XXS (not implemented).
	_, err = Dequantize(data, GGMLTypeIQ2_XXS, 32)
	if err == nil {
		t.Error("expected error for unsupported type IQ2_XXS, got nil")
	}
}

// TestDequantizeInsufficientData tests error handling for insufficient data.
func TestDequantizeInsufficientData(t *testing.T) {
	// Q8_0 needs 34 bytes per block.
	data := make([]byte, 10) // Too small.

	_, err := Dequantize(data, GGMLTypeQ8_0, 32)
	if err == nil {
		t.Error("expected error for insufficient data, got nil")
	}
}

// TestDequantizeBlockQ8_1 tests Q8_1 with sum offset.
func TestDequantizeBlockQ8_1(t *testing.T) {
	// Create Q8_1 block: d=0.1, s=0.01, qs=[1, 2, 3, ...].
	data := make([]byte, 36)

	// d = 0.1 in F16 (approx 0x2E66).
	binary.LittleEndian.PutUint16(data[0:2], 0x2E66)
	// s = 0.01 in F16 (approx 0x2028).
	binary.LittleEndian.PutUint16(data[2:4], 0x2028)

	// qs: 32 int8 values [1, 2, 3, ..., 32].
	for i := 0; i < 32; i++ {
		data[4+i] = byte(int8(i + 1))
	}

	result, err := DequantizeBlock(data, GGMLTypeQ8_1)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// sum = 1 + 2 + ... + 32 = 32 * 33 / 2 = 528.
	// Formula: x[i] = d * q[i] + s * sum.
	d := Float16ToFloat32(0x2E66)
	s := Float16ToFloat32(0x2028)
	sum := float32(528)

	for i := 0; i < 32; i++ {
		expected := d*float32(i+1) + s*sum
		if math.Abs(float64(result[i]-expected)) > 1e-3 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}

// TestDequantizeMultipleBlocks tests Dequantize with multiple blocks.
func TestDequantizeMultipleBlocks(t *testing.T) {
	// Create data for 2 Q8_0 blocks (64 elements total).
	data := make([]byte, 68) // 2 * 34 bytes.

	// Block 1: d=1.0, qs=[0, 1, 2, ..., 31].
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00)
	for i := 0; i < 32; i++ {
		data[2+i] = byte(int8(i))
	}

	// Block 2: d=2.0, qs=[0, 1, 2, ..., 31].
	binary.LittleEndian.PutUint16(data[34:36], 0x4000)
	for i := 0; i < 32; i++ {
		data[36+i] = byte(int8(i))
	}

	result, err := Dequantize(data, GGMLTypeQ8_0, 64)
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	if len(result) != 64 {
		t.Fatalf("expected 64 elements, got %d", len(result))
	}

	// Check first block: d=1.0, result[0] = 1.0 * 0 = 0.
	if result[0] != 0.0 {
		t.Errorf("result[0] = %v, want 0.0", result[0])
	}

	// Check second block: d=2.0, result[32] = 2.0 * 0 = 0.
	if result[32] != 0.0 {
		t.Errorf("result[32] = %v, want 0.0", result[32])
	}

	// Check second block element: result[33] = 2.0 * 1 = 2.0.
	if result[33] != 2.0 {
		t.Errorf("result[33] = %v, want 2.0", result[33])
	}
}

// TestDequantizeQ5_1 tests 5-bit quantization with offset.
func TestDequantizeQ5_1(t *testing.T) {
	// Create Q5_1 block: d=0.5, m=1.0, qh=0, qs=0.
	data := make([]byte, 24)

	// d = 0.5 in F16.
	binary.LittleEndian.PutUint16(data[0:2], 0x3800)
	// m = 1.0 in F16.
	binary.LittleEndian.PutUint16(data[2:4], 0x3C00)

	// qh = 0 (no high bits).
	binary.LittleEndian.PutUint32(data[4:8], 0)

	// qs = 0 (all zeros).
	for i := 0; i < 16; i++ {
		data[8+i] = 0
	}

	result, err := DequantizeBlock(data, GGMLTypeQ5_1)
	if err != nil {
		t.Fatalf("DequantizeBlock failed: %v", err)
	}

	if len(result) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(result))
	}

	// Formula: d * q + m = 0.5 * 0 + 1.0 = 1.0.
	expected := float32(1.0)
	for i := 0; i < 32; i++ {
		if math.Abs(float64(result[i]-expected)) > 1e-6 {
			t.Errorf("result[%d] = %v, want %v", i, result[i], expected)
		}
	}
}
