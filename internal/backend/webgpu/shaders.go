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
//
//nolint:unused // Will be used for exponential operation in ops.go
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
//
//nolint:unused // Will be used for logarithm operation in ops.go
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
//
//nolint:unused // Will be used for square root operation in ops.go
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
//
//nolint:unused // Will be used for scalar multiplication operation in ops.go
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
//
//nolint:unused // Will be used for scalar addition operation in ops.go
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
