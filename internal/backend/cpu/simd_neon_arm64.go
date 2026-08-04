//go:build arm64 && !goexperiment.simd

package cpu

import "golang.org/x/sys/cpu"

// init wires the vendored NEON element-wise kernels into the simd dispatch
// function pointers when the CPU exposes ASIMD (ARM64 NEON). ASIMD is mandatory
// on all ARMv8-A processors, so this path is active on essentially every arm64
// device. The function pointers are declared in ops_float32_simd_stub.go and
// dispatched with the simdMinLen guard in ops_float32.go.
func init() {
	if cpu.ARM64.HasASIMD {
		simdAddInplaceFloat32 = neonAddInplaceFloat32
		simdSubInplaceFloat32 = neonSubInplaceFloat32
		simdMulInplaceFloat32 = neonMulInplaceFloat32
		simdDivInplaceFloat32 = neonDivInplaceFloat32

		simdAddVectorizedFloat32 = neonAddVectorizedFloat32
		simdSubVectorizedFloat32 = neonSubVectorizedFloat32
		simdMulVectorizedFloat32 = neonMulVectorizedFloat32
		simdDivVectorizedFloat32 = neonDivVectorizedFloat32
	}
}
