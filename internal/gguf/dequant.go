package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Dequantize преобразует quantized данные в float32.
// Поддерживает: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_K, Q5_K, Q6_K.
func Dequantize(data []byte, dtype GGMLType, numElements int) ([]float32, error) {
	trait := dtype.Trait()

	// Validate input.
	if trait.TypeSize == 0 {
		return nil, fmt.Errorf("unsupported type: %s", dtype)
	}

	expectedBytes := dtype.RowSize(numElements)
	if len(data) < expectedBytes {
		return nil, fmt.Errorf("insufficient data: need %d bytes, got %d", expectedBytes, len(data))
	}

	// Handle non-quantized types.
	if !trait.Quantized {
		return dequantizeUnquantized(data, dtype, numElements)
	}

	// Handle quantized types block by block.
	result := make([]float32, numElements)
	numBlocks := (numElements + trait.BlockSize - 1) / trait.BlockSize

	offset := 0
	elemIdx := 0

	for i := 0; i < numBlocks; i++ {
		blockData := data[offset : offset+trait.TypeSize]
		block, err := DequantizeBlock(blockData, dtype)
		if err != nil {
			return nil, fmt.Errorf("dequantize block %d: %w", i, err)
		}

		// Copy block values to result.
		for j := 0; j < len(block) && elemIdx < numElements; j++ {
			result[elemIdx] = block[j]
			elemIdx++
		}

		offset += trait.TypeSize
	}

	return result, nil
}

// DequantizeBlock дequантизирует один блок данных.
// Используется для streaming дequантизации больших тензоров.
func DequantizeBlock(data []byte, dtype GGMLType) ([]float32, error) {
	switch dtype {
	case GGMLTypeQ4_0:
		return dequantizeBlockQ4_0(data)
	case GGMLTypeQ4_1:
		return dequantizeBlockQ4_1(data)
	case GGMLTypeQ5_0:
		return dequantizeBlockQ5_0(data)
	case GGMLTypeQ5_1:
		return dequantizeBlockQ5_1(data)
	case GGMLTypeQ8_0:
		return dequantizeBlockQ8_0(data)
	case GGMLTypeQ8_1:
		return dequantizeBlockQ8_1(data)
	case GGMLTypeQ4_K:
		return dequantizeBlockQ4_K(data)
	case GGMLTypeQ5_K:
		return dequantizeBlockQ5_K(data)
	case GGMLTypeQ6_K:
		return dequantizeBlockQ6_K(data)
	default:
		return nil, fmt.Errorf("unsupported quantized type: %s", dtype)
	}
}

// dequantizeUnquantized handles non-quantized types (F32, F16, etc).
func dequantizeUnquantized(data []byte, dtype GGMLType, numElements int) ([]float32, error) {
	result := make([]float32, numElements)

	switch dtype {
	case GGMLTypeF32:
		for i := 0; i < numElements; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}

	case GGMLTypeF16:
		for i := 0; i < numElements; i++ {
			h := binary.LittleEndian.Uint16(data[i*2:])
			result[i] = Float16ToFloat32(h)
		}

	case GGMLTypeI8:
		for i := 0; i < numElements; i++ {
			result[i] = float32(int8(data[i])) //nolint:gosec // G115: intentional byte-to-signed reinterpretation for GGML I8 format
		}

	case GGMLTypeI16:
		for i := 0; i < numElements; i++ {
			result[i] = float32(int16(binary.LittleEndian.Uint16(data[i*2:]))) //nolint:gosec // G115: integer overflow conversion uint16 -> int16
		}

	case GGMLTypeI32:
		for i := 0; i < numElements; i++ {
			result[i] = float32(int32(binary.LittleEndian.Uint32(data[i*4:]))) //nolint:gosec // G115: integer overflow conversion uint32 -> int32
		}

	default:
		return nil, fmt.Errorf("unsupported unquantized type: %s", dtype)
	}

	return result, nil
}

// Float16ToFloat32 конвертирует half precision (IEEE 754) в float32.
func Float16ToFloat32(h uint16) float32 {
	// Extract sign, exponent, and mantissa.
	sign := (h >> 15) & 0x1
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF

	var result uint32

	switch exp {
	case 0:
		if mant == 0 {
			// Zero.
			result = uint32(sign) << 31
		} else {
			// Subnormal F16: value = (-1)^sign * 2^(-14) * (mant / 1024).
			// Convert directly without normalization to avoid uint16 underflow.
			f := float64(mant) / 1024.0 * math.Pow(2, -14)
			if sign == 1 {
				f = -f
			}
			return float32(f)
		}
	case 0x1F:
		// Inf or NaN.
		result = (uint32(sign) << 31) | 0x7F800000 | (uint32(mant) << 13)
	default:
		// Normal number.
		result = (uint32(sign) << 31) | (uint32(exp+127-15) << 23) | (uint32(mant) << 13)
	}

	return math.Float32frombits(result)
}

// Q4_0: 32 elements per block, 4 bits per element.
// Structure: half d (2 bytes), uint8_t qs[16] (16 bytes).
// Formula: x[i] = d * (q[i] - 8) where q[i] is 4-bit value.
func dequantizeBlockQ4_0(data []byte) ([]float32, error) {
	if len(data) < 18 {
		return nil, fmt.Errorf("insufficient data for Q4_0 block: need 18 bytes, got %d", len(data))
	}

	// Read scale factor (half precision).
	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))

	// Read quantized values (4 bits each, 2 per byte).
	result := make([]float32, 32)
	for i := 0; i < 16; i++ {
		qByte := data[2+i]

		// Low 4 bits.
		q0 := qByte & 0x0F
		result[i*2] = d * (float32(q0) - 8.0)

		// High 4 bits.
		q1 := qByte >> 4
		result[i*2+1] = d * (float32(q1) - 8.0)
	}

	return result, nil
}

// Q4_1: 32 elements per block, 4 bits per element with offset.
// Structure: half d (2 bytes), half m (2 bytes), uint8_t qs[16] (16 bytes).
// Formula: x[i] = d * q[i] + m.
func dequantizeBlockQ4_1(data []byte) ([]float32, error) {
	if len(data) < 20 {
		return nil, fmt.Errorf("insufficient data for Q4_1 block: need 20 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	m := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))

	result := make([]float32, 32)
	for i := 0; i < 16; i++ {
		qByte := data[4+i]

		q0 := qByte & 0x0F
		result[i*2] = d*float32(q0) + m

		q1 := qByte >> 4
		result[i*2+1] = d*float32(q1) + m
	}

	return result, nil
}

// Q5_0: 32 elements per block, 5 bits per element.
// Structure: half d (2 bytes), uint32_t qh (4 bytes), uint8_t qs[16] (16 bytes).
// qh contains high bits, qs contains low 4 bits.
// Formula: x[i] = d * (q[i] - 16) where q[i] is 5-bit value.
func dequantizeBlockQ5_0(data []byte) ([]float32, error) {
	if len(data) < 22 {
		return nil, fmt.Errorf("insufficient data for Q5_0 block: need 22 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	qh := binary.LittleEndian.Uint32(data[2:6])

	result := make([]float32, 32)
	for i := 0; i < 16; i++ {
		qByte := data[6+i]

		// Reconstruct 5-bit values.
		q0Low := qByte & 0x0F
		q0High := (qh >> i) & 0x1
		q0 := q0Low | (uint8(q0High) << 4)
		result[i*2] = d * (float32(q0) - 16.0)

		q1Low := qByte >> 4
		q1High := (qh >> (i + 16)) & 0x1
		q1 := q1Low | (uint8(q1High) << 4)
		result[i*2+1] = d * (float32(q1) - 16.0)
	}

	return result, nil
}

// Q5_1: 32 elements per block, 5 bits per element with offset.
// Structure: half d (2 bytes), half m (2 bytes), uint32_t qh (4 bytes), uint8_t qs[16] (16 bytes).
// Formula: x[i] = d * q[i] + m.
func dequantizeBlockQ5_1(data []byte) ([]float32, error) {
	if len(data) < 24 {
		return nil, fmt.Errorf("insufficient data for Q5_1 block: need 24 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	m := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))
	qh := binary.LittleEndian.Uint32(data[4:8])

	result := make([]float32, 32)
	for i := 0; i < 16; i++ {
		qByte := data[8+i]

		q0Low := qByte & 0x0F
		q0High := (qh >> i) & 0x1
		q0 := q0Low | (uint8(q0High) << 4)
		result[i*2] = d*float32(q0) + m

		q1Low := qByte >> 4
		q1High := (qh >> (i + 16)) & 0x1
		q1 := q1Low | (uint8(q1High) << 4)
		result[i*2+1] = d*float32(q1) + m
	}

	return result, nil
}

// Q8_0: 32 elements per block, 8 bits per element.
// Structure: half d (2 bytes), int8_t qs[32] (32 bytes).
// Formula: x[i] = d * q[i].
func dequantizeBlockQ8_0(data []byte) ([]float32, error) {
	if len(data) < 34 {
		return nil, fmt.Errorf("insufficient data for Q8_0 block: need 34 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))

	result := make([]float32, 32)
	for i := 0; i < 32; i++ {
		q := int8(data[2+i]) //nolint:gosec // G115: intentional byte-to-signed reinterpretation for Q8_0 quantized weights
		result[i] = d * float32(q)
	}

	return result, nil
}

// Q8_1: 32 elements per block, 8 bits per element with offset.
// Structure: half d (2 bytes), half s (2 bytes), int8_t qs[32] (32 bytes).
// Formula: x[i] = d * q[i] + s * sum(q).
func dequantizeBlockQ8_1(data []byte) ([]float32, error) {
	if len(data) < 36 {
		return nil, fmt.Errorf("insufficient data for Q8_1 block: need 36 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	s := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))

	// Calculate sum of quantized values.
	sum := int32(0)
	for i := 0; i < 32; i++ {
		sum += int32(int8(data[4+i])) //nolint:gosec // G115: intentional byte-to-signed reinterpretation for Q8_1 quantized weights
	}

	result := make([]float32, 32)
	for i := 0; i < 32; i++ {
		q := int8(data[4+i]) //nolint:gosec // G115: intentional byte-to-signed reinterpretation for Q8_1 quantized weights
		result[i] = d*float32(q) + s*float32(sum)
	}

	return result, nil
}

// Q4_K: 256 elements per block, 4-bit K-quant with super-block scales.
// Structure:
//
//	half d (2 bytes) - super-block scale
//	half dmin (2 bytes) - super-block minimum
//	uint8_t scales[12] (12 bytes) - packed 6-bit scales
//	uint8_t qs[128] (128 bytes) - 4-bit quantized values
//
// 256 elements = 8 sub-blocks of 32 elements each.
//
//nolint:revive // Function name matches GGML specification (Q4_K format).
func dequantizeBlockQ4_K(data []byte) ([]float32, error) {
	if len(data) < 144 {
		return nil, fmt.Errorf("insufficient data for Q4_K block: need 144 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))

	// Unpack scales and mins from 12 packed bytes (bytes 4..15).
	//
	// GGML Q4_K scale layout (from ggml-quants.c dequantize_row_q4_K):
	//   bytes [0..3]:  low 6 bits = sc[0..3], bits [6:7] = high 2 bits of m[0..3]
	//   bytes [4..7]:  low 6 bits = sc[4..7], bits [6:7] = high 2 bits of m[4..7]
	//   bytes [8..11]: low nibble = low 4 bits of m[0..3], high nibble = low 4 bits of m[4..7]
	scales := make([]uint8, 8)
	mins := make([]uint8, 8)
	sc := data[4:16] // 12 scale bytes

	for i := 0; i < 4; i++ {
		scales[i] = sc[i] & 0x3F
		scales[i+4] = sc[i+4] & 0x3F
		mins[i] = (sc[i] >> 6) | ((sc[i+8] & 0x0F) << 2)
		mins[i+4] = (sc[i+4] >> 6) | ((sc[i+8] >> 4) << 2)
	}

	result := make([]float32, 256)

	// Process 8 sub-blocks of 32 elements each.
	for subBlock := 0; subBlock < 8; subBlock++ {
		scale := d * float32(scales[subBlock])
		minVal := dmin * float32(mins[subBlock])

		offset := 16 + subBlock*16

		for i := 0; i < 16; i++ {
			qByte := data[offset+i]

			// Low 4 bits.
			q0 := qByte & 0x0F
			result[subBlock*32+i*2] = scale*float32(q0) - minVal

			// High 4 bits.
			q1 := qByte >> 4
			result[subBlock*32+i*2+1] = scale*float32(q1) - minVal
		}
	}

	return result, nil
}

// Q5_K: 256 elements per block, 5-bit K-quant.
// Structure:
//
//	half d (2 bytes)
//	half dmin (2 bytes)
//	uint8_t scales[12] (12 bytes)
//	uint8_t qh[32] (32 bytes) - high bits
//	uint8_t qs[128] (128 bytes) - low 4 bits
//
//nolint:revive // Function name matches GGML specification (Q5_K format).
func dequantizeBlockQ5_K(data []byte) ([]float32, error) {
	if len(data) < 176 {
		return nil, fmt.Errorf("insufficient data for Q5_K block: need 176 bytes, got %d", len(data))
	}

	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))

	// Unpack scales and mins — same GGML layout as Q4_K.
	scales := make([]uint8, 8)
	mins := make([]uint8, 8)
	sc := data[4:16]

	for i := 0; i < 4; i++ {
		scales[i] = sc[i] & 0x3F
		scales[i+4] = sc[i+4] & 0x3F
		mins[i] = (sc[i] >> 6) | ((sc[i+8] & 0x0F) << 2)
		mins[i+4] = (sc[i+4] >> 6) | ((sc[i+8] >> 4) << 2)
	}

	result := make([]float32, 256)

	// qh starts at offset 16, qs starts at offset 48.
	for subBlock := 0; subBlock < 8; subBlock++ {
		scale := d * float32(scales[subBlock])
		minVal := dmin * float32(mins[subBlock])

		qhOffset := 16 + subBlock*4
		qsOffset := 48 + subBlock*16

		for i := 0; i < 16; i++ {
			qByte := data[qsOffset+i]
			qhByte := data[qhOffset+i/4]

			// Low element.
			q0Low := qByte & 0x0F
			q0High := (qhByte >> ((i % 4) * 2)) & 0x1
			q0 := q0Low | (q0High << 4)
			result[subBlock*32+i*2] = scale*float32(q0) - minVal

			// High element.
			q1Low := qByte >> 4
			q1High := (qhByte >> ((i%4)*2 + 1)) & 0x1
			q1 := q1Low | (q1High << 4)
			result[subBlock*32+i*2+1] = scale*float32(q1) - minVal
		}
	}

	return result, nil
}

// Q6_K: 256 elements per block, 6-bit K-quant.
// Structure:
//
//	uint8_t ql[128] (128 bytes) - low 4 bits
//	uint8_t qh[64] (64 bytes) - high 2 bits
//	int8_t scales[16] (16 bytes) - signed scales
//	half d (2 bytes)
//
//nolint:revive // Function name matches GGML specification (Q6_K format).
func dequantizeBlockQ6_K(data []byte) ([]float32, error) {
	if len(data) < 210 {
		return nil, fmt.Errorf("insufficient data for Q6_K block: need 210 bytes, got %d", len(data))
	}

	// d is at the end.
	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[208:210]))

	result := make([]float32, 256)

	// 16 sub-blocks of 16 elements each.
	for subBlock := 0; subBlock < 16; subBlock++ {
		scale := d * float32(int8(data[192+subBlock])) //nolint:gosec // G115: integer overflow conversion byte -> int8

		qlOffset := subBlock * 8
		qhOffset := 128 + subBlock*4

		for i := 0; i < 8; i++ {
			qlByte := data[qlOffset+i]
			qhByte := data[qhOffset+i/2]

			// Each qlByte has 2 elements (4 bits each).
			// Each qhByte has 4 elements (2 bits each).

			// Low element.
			q0Low := qlByte & 0x0F
			q0High := (qhByte >> ((i % 2) * 4)) & 0x3
			q0 := q0Low | (q0High << 4)
			result[subBlock*16+i*2] = scale * float32(int8(q0)-32) //nolint:gosec // G115: integer overflow conversion byte -> int8

			// High element.
			q1Low := qlByte >> 4
			q1High := (qhByte >> ((i%2)*4 + 2)) & 0x3
			q1 := q1Low | (q1High << 4)
			result[subBlock*16+i*2+1] = scale * float32(int8(q1)-32) //nolint:gosec // G115: integer overflow conversion byte -> int8
		}
	}

	return result, nil
}
