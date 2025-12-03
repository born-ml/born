//go:build windows

// Package webgpu provides embedded WGSL compute shaders for tensor operations.
package webgpu

// WGSL compute shaders for tensor operations.
// Using string constants instead of embed for simplicity.

// workgroupSize is the default number of threads per workgroup.
const workgroupSize = 256

// addShader performs element-wise addition: result = a + b.
const addShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] + b[idx];
    }
}
`

// subShader performs element-wise subtraction: result = a - b.
const subShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] - b[idx];
    }
}
`

// mulShader performs element-wise multiplication: result = a * b.
const mulShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] * b[idx];
    }
}
`

// divShader performs element-wise division: result = a / b.
const divShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] / b[idx];
    }
}
`

// matmulShader performs matrix multiplication: C = A @ B.
// A is [M, K], B is [K, N], C is [M, N].
const matmulShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    M: u32,  // rows of A and C
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B and C
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a_idx = row * params.K + k;
        let b_idx = k * params.N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * params.N + col;
    result[c_idx] = sum;
}
`

// transposeShader transposes a 2D matrix.
const transposeShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    rows: u32,
    cols: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let in_idx = row * params.cols + col;
    let out_idx = col * params.rows + row;
    result[out_idx] = input[in_idx];
}
`

// reluShader applies ReLU activation: result = max(0, x).
const reluShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = max(0.0, input[idx]);
    }
}
`

// sigmoidShader applies sigmoid activation: result = 1 / (1 + exp(-x)).
const sigmoidShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}
`

// tanhShader applies tanh activation.
const tanhShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = tanh(input[idx]);
    }
}
`

// sumShader performs parallel sum reduction.
//
//nolint:unused // Will be used for reduction operations (sum, mean, etc.)
const sumShader = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

struct Params {
    size: u32,
    stride: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let partner = idx + params.stride;

    if (idx < params.stride && partner < params.size) {
        data[idx] = data[idx] + data[partner];
    }
}
`

// negShader performs element-wise negation: result = -x.
//
//nolint:unused // Will be used for negation operation in ops.go
const negShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = -input[idx];
    }
}
`

// expShader performs element-wise exp: result = exp(x).
const expShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = exp(input[idx]);
    }
}
`

// logShader performs element-wise log: result = log(x).
const logShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = log(input[idx]);
    }
}
`

// sqrtShader performs element-wise sqrt: result = sqrt(x).
const sqrtShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = sqrt(input[idx]);
    }
}
`

// scalarMulShader performs scalar multiplication: result = x * scalar.
const scalarMulShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
    scalar: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = input[idx] * params.scalar;
    }
}
`

// scalarAddShader performs scalar addition: result = x + scalar.
const scalarAddShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
    scalar: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = input[idx] + params.scalar;
    }
}
`

// rsqrtShader performs element-wise reciprocal square root: result = 1/sqrt(x).
const rsqrtShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = inverseSqrt(input[idx]);
    }
}
`

// cosShader performs element-wise cosine: result = cos(x).
const cosShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = cos(input[idx]);
    }
}
`

// sinShader performs element-wise sine: result = sin(x).
const sinShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = sin(input[idx]);
    }
}
`

// siluShader performs SiLU activation: result = x * sigmoid(x) = x / (1 + exp(-x)).
//
//nolint:unused // Prepared for SiLU operation (will be used when SiLU is added to backend interface)
const siluShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        let x = input[idx];
        result[idx] = x / (1.0 + exp(-x));
    }
}
`

// softmaxShader applies softmax along rows (last dimension).
// Input shape: [batch_size, num_classes]
// Uses max-shift trick for numerical stability.
const softmaxShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    batch_size: u32,
    num_classes: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.batch_size) {
        return;
    }

    let offset = row * params.num_classes;

    // Find max for numerical stability
    var max_val: f32 = input[offset];
    for (var i: u32 = 1u; i < params.num_classes; i = i + 1u) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.num_classes; i = i + 1u) {
        let exp_val = exp(input[offset + i] - max_val);
        result[offset + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i: u32 = 0u; i < params.num_classes; i = i + 1u) {
        result[offset + i] = result[offset + i] / sum;
    }
}
`

// batchMatMulShader performs batched matrix multiplication: C[b] = A[b] @ B[b].
// A is [batch, M, K], B is [batch, K, N], C is [batch, M, N].
const batchMatMulShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    batch: u32,
    M: u32,
    K: u32,
    N: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    if (batch_idx >= params.batch || row >= params.M || col >= params.N) {
        return;
    }

    let a_batch_offset = batch_idx * params.M * params.K;
    let b_batch_offset = batch_idx * params.K * params.N;
    let c_batch_offset = batch_idx * params.M * params.N;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a_idx = a_batch_offset + row * params.K + k;
        let b_idx = b_batch_offset + k * params.N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = c_batch_offset + row * params.N + col;
    result[c_idx] = sum;
}
`

// greaterShader performs element-wise greater-than comparison: result = a > b ? 1.0 : 0.0.
const greaterShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] > b[idx]);
    }
}
`

// lowerShader performs element-wise less-than comparison: result = a < b ? 1.0 : 0.0.
const lowerShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] < b[idx]);
    }
}
`

// greaterEqualShader performs element-wise greater-or-equal comparison.
const greaterEqualShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] >= b[idx]);
    }
}
`

// lowerEqualShader performs element-wise less-or-equal comparison.
const lowerEqualShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] <= b[idx]);
    }
}
`

// equalShader performs element-wise equality comparison.
const equalShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] == b[idx]);
    }
}
`

// notEqualShader performs element-wise inequality comparison.
const notEqualShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, 1.0, a[idx] != b[idx]);
    }
}
`

// andShader performs element-wise logical AND (non-zero = true).
const andShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        let a_bool = a[idx] != 0.0;
        let b_bool = b[idx] != 0.0;
        result[idx] = select(0.0, 1.0, a_bool && b_bool);
    }
}
`

// orShader performs element-wise logical OR (non-zero = true).
const orShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        let a_bool = a[idx] != 0.0;
        let b_bool = b[idx] != 0.0;
        result[idx] = select(0.0, 1.0, a_bool || b_bool);
    }
}
`

// notShader performs element-wise logical NOT (non-zero = true).
const notShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(1.0, 0.0, input[idx] != 0.0);
    }
}
`

// argmaxShader finds index of maximum value along last dimension.
// Input: [batch, dim], Output: [batch] (int32 stored as f32).
const argmaxShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    batch_size: u32,
    dim_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.dim_size;
    var max_val = input[offset];
    var max_idx: u32 = 0u;

    for (var i: u32 = 1u; i < params.dim_size; i = i + 1u) {
        let val = input[offset + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    result[batch_idx] = f32(max_idx);
}
`

// globalSumShader performs parallel sum reduction.
// Uses workgroup shared memory for efficient reduction.
const globalSumShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        result[workgroup_id.x] = shared_data[0];
    }
}
`

// globalSumShaderInt32 performs parallel sum reduction for int32.
const globalSumShaderInt32 = `
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0;
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        result[workgroup_id.x] = shared_data[0];
    }
}
`

// conv2dShader performs 2D convolution.
// Input shape: [batch, in_channels, height, width].
// Kernel shape: [out_channels, in_channels, kH, kW].
// Output shape: [batch, out_channels, out_height, out_width].
const conv2dShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    batch: u32,
    in_channels: u32,
    in_height: u32,
    in_width: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_width = (params.in_width + 2u * params.padding - params.kernel_w) / params.stride + 1u;
    let out_height = (params.in_height + 2u * params.padding - params.kernel_h) / params.stride + 1u;

    let b = global_id.z / params.out_channels;
    let oc = global_id.z % params.out_channels;
    let oh = global_id.y;
    let ow = global_id.x;

    if (b >= params.batch || oh >= out_height || ow >= out_width) {
        return;
    }

    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
                let ih = oh * params.stride + kh;
                let iw = ow * params.stride + kw;

                // Check padding bounds
                let ih_pad = ih - params.padding;
                let iw_pad = iw - params.padding;

                if (ih_pad < params.in_height && iw_pad < params.in_width) {
                    let in_idx = b * params.in_channels * params.in_height * params.in_width +
                                 ic * params.in_height * params.in_width +
                                 ih_pad * params.in_width +
                                 iw_pad;

                    let k_idx = oc * params.in_channels * params.kernel_h * params.kernel_w +
                                ic * params.kernel_h * params.kernel_w +
                                kh * params.kernel_w +
                                kw;

                    sum = sum + input[in_idx] * kernel[k_idx];
                }
            }
        }
    }

    let out_idx = b * params.out_channels * out_height * out_width +
                  oc * out_height * out_width +
                  oh * out_width +
                  ow;
    output[out_idx] = sum;
}
`

// maxPool2dShader performs 2D max pooling.
// Input shape: [batch, channels, height, width].
// Output shape: [batch, channels, out_height, out_width].
const maxPool2dShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_width = (params.in_width - params.kernel_w) / params.stride + 1u;
    let out_height = (params.in_height - params.kernel_h) / params.stride + 1u;

    let b = global_id.z / params.channels;
    let c = global_id.z % params.channels;
    let oh = global_id.y;
    let ow = global_id.x;

    if (b >= params.batch || oh >= out_height || ow >= out_width) {
        return;
    }

    var max_val: f32 = -3.402823e+38; // -FLT_MAX

    for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let ih = oh * params.stride + kh;
            let iw = ow * params.stride + kw;

            let in_idx = b * params.channels * params.in_height * params.in_width +
                         c * params.in_height * params.in_width +
                         ih * params.in_width +
                         iw;

            max_val = max(max_val, input[in_idx]);
        }
    }

    let out_idx = b * params.channels * out_height * out_width +
                  c * out_height * out_width +
                  oh * out_width +
                  ow;
    output[out_idx] = max_val;
}
`

// whereShader performs conditional selection: result = condition ? x : y.
// condition is interpreted as boolean (non-zero = true).
const whereShader = `
@group(0) @binding(0) var<storage, read> condition: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(y[idx], x[idx], condition[idx] != 0.0);
    }
}
`

// whereShaderInt32 performs conditional selection for int32: result = condition ? x : y.
// condition is interpreted as boolean (non-zero = true).
const whereShaderInt32 = `
@group(0) @binding(0) var<storage, read> condition: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<i32>;
@group(0) @binding(2) var<storage, read> y: array<i32>;
@group(0) @binding(3) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(y[idx], x[idx], condition[idx] != 0.0);
    }
}
`

// embeddingShader performs embedding lookup: output[i] = weight[indices[i], :].
// weight: [num_embeddings, embedding_dim], indices: [...], output: [..., embedding_dim].
const embeddingShader = `
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    num_indices: u32,
    embedding_dim: u32,
    num_embeddings: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.num_indices * params.embedding_dim;
    if (idx >= total_elements) {
        return;
    }

    let batch_idx = idx / params.embedding_dim;
    let dim_idx = idx % params.embedding_dim;
    let embed_idx = u32(indices[batch_idx]);

    if (embed_idx < params.num_embeddings) {
        let src_offset = embed_idx * params.embedding_dim + dim_idx;
        result[idx] = weight[src_offset];
    } else {
        result[idx] = 0.0;
    }
}
`

// Int32 Binary Operations - shaders for integer tensor operations.

// addShaderInt32 performs element-wise addition for int32: result = a + b.
const addShaderInt32 = `
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] + b[idx];
    }
}
`

// subShaderInt32 performs element-wise subtraction for int32: result = a - b.
const subShaderInt32 = `
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] - b[idx];
    }
}
`

// mulShaderInt32 performs element-wise multiplication for int32: result = a * b.
const mulShaderInt32 = `
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] * b[idx];
    }
}
`

// divShaderInt32 performs element-wise division for int32: result = a / b.
const divShaderInt32 = `
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<i32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = a[idx] / b[idx];
    }
}
`

// gatherShader gathers elements along the last dimension using indices.
// Input: [batch, dim], Indices: [batch, k] (int32), Output: [batch, k].
const gatherShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    batch_size: u32,
    input_dim: u32,
    output_k: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_output = params.batch_size * params.output_k;

    if (idx >= total_output) {
        return;
    }

    let batch_idx = idx / params.output_k;
    let k_idx = idx % params.output_k;

    // Get the index to gather (int32)
    let gather_idx = u32(indices[idx]);

    // Bounds check
    if (gather_idx < params.input_dim) {
        let input_offset = batch_idx * params.input_dim + gather_idx;
        result[idx] = input[input_offset];
    } else {
        result[idx] = 0.0;
    }
}
`

// transposeNDShader performs N-dimensional transpose with arbitrary axes permutation.
// Supports up to 6D tensors (covers 99% of ML use cases).
const transposeNDShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    ndim: u32,
    total_elements: u32,
    input_shape_0: u32,
    input_shape_1: u32,
    input_shape_2: u32,
    input_shape_3: u32,
    input_shape_4: u32,
    input_shape_5: u32,
    input_strides_0: u32,
    input_strides_1: u32,
    input_strides_2: u32,
    input_strides_3: u32,
    input_strides_4: u32,
    input_strides_5: u32,
    output_strides_0: u32,
    output_strides_1: u32,
    output_strides_2: u32,
    output_strides_3: u32,
    output_strides_4: u32,
    output_strides_5: u32,
    axes_0: u32,
    axes_1: u32,
    axes_2: u32,
    axes_3: u32,
    axes_4: u32,
    axes_5: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    // Convert output linear index to coordinates
    var coords: array<u32, 6>;
    var temp = idx;

    // Load output strides into array for indexing
    var output_strides: array<u32, 6>;
    output_strides[0] = params.output_strides_0;
    output_strides[1] = params.output_strides_1;
    output_strides[2] = params.output_strides_2;
    output_strides[3] = params.output_strides_3;
    output_strides[4] = params.output_strides_4;
    output_strides[5] = params.output_strides_5;

    // Load input strides
    var input_strides: array<u32, 6>;
    input_strides[0] = params.input_strides_0;
    input_strides[1] = params.input_strides_1;
    input_strides[2] = params.input_strides_2;
    input_strides[3] = params.input_strides_3;
    input_strides[4] = params.input_strides_4;
    input_strides[5] = params.input_strides_5;

    // Load axes permutation
    var axes: array<u32, 6>;
    axes[0] = params.axes_0;
    axes[1] = params.axes_1;
    axes[2] = params.axes_2;
    axes[3] = params.axes_3;
    axes[4] = params.axes_4;
    axes[5] = params.axes_5;

    // Calculate coordinates in output space
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        coords[d] = temp / output_strides[d];
        temp = temp % output_strides[d];
    }

    // Map output coordinates to input index using axes permutation
    var input_idx: u32 = 0u;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        input_idx = input_idx + coords[d] * input_strides[axes[d]];
    }

    result[idx] = input[input_idx];
}
`

// transposeNDShaderInt32 performs N-dimensional transpose for int32 tensors.
const transposeNDShaderInt32 = `
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> result: array<i32>;

struct Params {
    ndim: u32,
    total_elements: u32,
    input_shape_0: u32,
    input_shape_1: u32,
    input_shape_2: u32,
    input_shape_3: u32,
    input_shape_4: u32,
    input_shape_5: u32,
    input_strides_0: u32,
    input_strides_1: u32,
    input_strides_2: u32,
    input_strides_3: u32,
    input_strides_4: u32,
    input_strides_5: u32,
    output_strides_0: u32,
    output_strides_1: u32,
    output_strides_2: u32,
    output_strides_3: u32,
    output_strides_4: u32,
    output_strides_5: u32,
    axes_0: u32,
    axes_1: u32,
    axes_2: u32,
    axes_3: u32,
    axes_4: u32,
    axes_5: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    var coords: array<u32, 6>;
    var temp = idx;

    var output_strides: array<u32, 6>;
    output_strides[0] = params.output_strides_0;
    output_strides[1] = params.output_strides_1;
    output_strides[2] = params.output_strides_2;
    output_strides[3] = params.output_strides_3;
    output_strides[4] = params.output_strides_4;
    output_strides[5] = params.output_strides_5;

    var input_strides: array<u32, 6>;
    input_strides[0] = params.input_strides_0;
    input_strides[1] = params.input_strides_1;
    input_strides[2] = params.input_strides_2;
    input_strides[3] = params.input_strides_3;
    input_strides[4] = params.input_strides_4;
    input_strides[5] = params.input_strides_5;

    var axes: array<u32, 6>;
    axes[0] = params.axes_0;
    axes[1] = params.axes_1;
    axes[2] = params.axes_2;
    axes[3] = params.axes_3;
    axes[4] = params.axes_4;
    axes[5] = params.axes_5;

    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        coords[d] = temp / output_strides[d];
        temp = temp % output_strides[d];
    }

    var input_idx: u32 = 0u;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        input_idx = input_idx + coords[d] * input_strides[axes[d]];
    }

    result[idx] = input[input_idx];
}
`

// expandShader performs NumPy-style broadcasting to expand tensor to new shape.
// Supports up to 6D tensors.
const expandShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    ndim: u32,
    total_elements: u32,
    input_shape_0: u32,
    input_shape_1: u32,
    input_shape_2: u32,
    input_shape_3: u32,
    input_shape_4: u32,
    input_shape_5: u32,
    input_strides_0: u32,
    input_strides_1: u32,
    input_strides_2: u32,
    input_strides_3: u32,
    input_strides_4: u32,
    input_strides_5: u32,
    output_strides_0: u32,
    output_strides_1: u32,
    output_strides_2: u32,
    output_strides_3: u32,
    output_strides_4: u32,
    output_strides_5: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_elements) {
        return;
    }

    // Load shapes and strides
    var input_shape: array<u32, 6>;
    input_shape[0] = params.input_shape_0;
    input_shape[1] = params.input_shape_1;
    input_shape[2] = params.input_shape_2;
    input_shape[3] = params.input_shape_3;
    input_shape[4] = params.input_shape_4;
    input_shape[5] = params.input_shape_5;

    var input_strides: array<u32, 6>;
    input_strides[0] = params.input_strides_0;
    input_strides[1] = params.input_strides_1;
    input_strides[2] = params.input_strides_2;
    input_strides[3] = params.input_strides_3;
    input_strides[4] = params.input_strides_4;
    input_strides[5] = params.input_strides_5;

    var output_strides: array<u32, 6>;
    output_strides[0] = params.output_strides_0;
    output_strides[1] = params.output_strides_1;
    output_strides[2] = params.output_strides_2;
    output_strides[3] = params.output_strides_3;
    output_strides[4] = params.output_strides_4;
    output_strides[5] = params.output_strides_5;

    // Convert output index to coordinates
    var coords: array<u32, 6>;
    var temp = out_idx;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        coords[d] = temp / output_strides[d];
        temp = temp % output_strides[d];
    }

    // Map to input index with broadcasting (dim=1 broadcasts to any size)
    var in_idx: u32 = 0u;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        // If input dim is 1, use coord 0 (broadcast), otherwise use actual coord
        let in_coord = select(coords[d], 0u, input_shape[d] == 1u);
        in_idx = in_idx + in_coord * input_strides[d];
    }

    result[out_idx] = input[in_idx];
}
`

// reluBackwardShader computes ReLU gradient: grad * (input > 0).
// d(ReLU(x))/dx = 1 if x > 0, else 0.
const reluBackwardShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.size) {
        result[idx] = select(0.0, grad[idx], input[idx] > 0.0);
    }
}
`

// sumDimShader performs sum reduction along the last dimension.
// Input: [batch, dim], Output: [batch].
const sumDimShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    batch_size: u32,
    dim_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.dim_size;
    var sum: f32 = 0.0;

    for (var i: u32 = 0u; i < params.dim_size; i = i + 1u) {
        sum = sum + input[offset + i];
    }

    result[batch_idx] = sum;
}
`

// softmaxBackwardShader computes softmax gradient.
// d_input[i] = s[i] * (grad[i] - sum(s * grad))
// where s = softmax output.
const softmaxBackwardShader = `
@group(0) @binding(0) var<storage, read> output: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    batch_size: u32,
    dim_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.dim_size;

    // Compute sum(s * grad)
    var dot_product: f32 = 0.0;
    for (var i: u32 = 0u; i < params.dim_size; i = i + 1u) {
        dot_product = dot_product + output[offset + i] * grad[offset + i];
    }

    // Compute d_input = s * (grad - dot_product)
    for (var i: u32 = 0u; i < params.dim_size; i = i + 1u) {
        let s = output[offset + i];
        let g = grad[offset + i];
        result[offset + i] = s * (g - dot_product);
    }
}
`

// expandShaderInt32 performs broadcasting for int32 tensors.
const expandShaderInt32 = `
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> result: array<i32>;

struct Params {
    ndim: u32,
    total_elements: u32,
    input_shape_0: u32,
    input_shape_1: u32,
    input_shape_2: u32,
    input_shape_3: u32,
    input_shape_4: u32,
    input_shape_5: u32,
    input_strides_0: u32,
    input_strides_1: u32,
    input_strides_2: u32,
    input_strides_3: u32,
    input_strides_4: u32,
    input_strides_5: u32,
    output_strides_0: u32,
    output_strides_1: u32,
    output_strides_2: u32,
    output_strides_3: u32,
    output_strides_4: u32,
    output_strides_5: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_elements) {
        return;
    }

    var input_shape: array<u32, 6>;
    input_shape[0] = params.input_shape_0;
    input_shape[1] = params.input_shape_1;
    input_shape[2] = params.input_shape_2;
    input_shape[3] = params.input_shape_3;
    input_shape[4] = params.input_shape_4;
    input_shape[5] = params.input_shape_5;

    var input_strides: array<u32, 6>;
    input_strides[0] = params.input_strides_0;
    input_strides[1] = params.input_strides_1;
    input_strides[2] = params.input_strides_2;
    input_strides[3] = params.input_strides_3;
    input_strides[4] = params.input_strides_4;
    input_strides[5] = params.input_strides_5;

    var output_strides: array<u32, 6>;
    output_strides[0] = params.output_strides_0;
    output_strides[1] = params.output_strides_1;
    output_strides[2] = params.output_strides_2;
    output_strides[3] = params.output_strides_3;
    output_strides[4] = params.output_strides_4;
    output_strides[5] = params.output_strides_5;

    var coords: array<u32, 6>;
    var temp = out_idx;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        coords[d] = temp / output_strides[d];
        temp = temp % output_strides[d];
    }

    var in_idx: u32 = 0u;
    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        let in_coord = select(coords[d], 0u, input_shape[d] == 1u);
        in_idx = in_idx + in_coord * input_strides[d];
    }

    result[out_idx] = input[in_idx];
}
`
