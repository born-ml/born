# Contributing to Born

Thank you for your interest in contributing to Born! This document guides you through the contribution process.

Born is a modern ML framework for Go, and we welcome contributions of all kinds - from bug fixes to new features, documentation improvements to performance optimizations.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)
- [Communication](#communication)

---

## Getting Started

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Go** | 1.25+ | [Download](https://go.dev/dl/) |
| **Git** | Latest | Version control |
| **golangci-lint** | Latest | [Install](https://golangci-lint.run/usage/install/) |
| **Make** | Any | Build automation (optional) |

**For WebGPU development** (optional):
- [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) - GPU backend

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/born-ml/born.git
cd born

# Install dependencies
go mod download

# Verify setup
make check    # Runs tests + lint + vet

# Or manually:
go test ./...
golangci-lint run
```

### Project Structure

```
born/
├── internal/           # Private implementation
│   ├── tensor/         # Core tensor types and Backend interface
│   ├── backend/cpu/    # CPU backend implementation
│   ├── backend/webgpu/ # GPU backend (WebGPU)
│   ├── autodiff/       # Automatic differentiation
│   ├── nn/             # Neural network modules
│   └── optim/          # Optimizers
├── tensor/             # Public API
├── nn/                 # Public API
├── optim/              # Public API
├── examples/           # Example applications
└── docs/               # Documentation
```

---

## Development Workflow

### 1. Check Project Status

Before starting:
- Check [GitHub Issues](https://github.com/born-ml/born/issues) for existing work
- Review [ROADMAP.md](ROADMAP.md) for project priorities
- Check [CHANGELOG.md](CHANGELOG.md) for recent changes

### 2. Pick a Task

- Issues labeled `good-first-issue` are great starting points
- Issues labeled `help-wanted` need contributors
- Comment on the issue to claim it before starting

### 3. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feat/your-feature

# Or for fixes
git checkout -b fix/issue-description
```

**Branch naming conventions**:
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code restructuring
- `perf/` - Performance improvements
- `hotfix/` - Urgent production fixes

### 4. Development

- Write code following our [Code Standards](#code-standards)
- Write tests (target: 70%+ coverage)
- Update documentation if needed
- Run checks locally before pushing

### 5. Testing

```bash
# All tests
make test
# Or: go test ./...

# With race detector
make test-race
# Or: go test -race ./...

# Coverage report
make coverage
# Or: go test -cover ./...

# Specific package
go test ./internal/tensor/... -v

# Single test
go test -run TestTensorAdd -v

# Benchmarks
make bench
# Or: go test -bench=. -benchmem ./...
```

### 6. Linting

We use aggressive golangci-lint configuration with 20+ linters:

```bash
# Run linter
make lint
# Or: golangci-lint run

# Auto-fix where possible
golangci-lint run --fix
```

**Key linter requirements**:
- `gofmt` - Code formatting
- `govet` - Go vet checks
- `errcheck` - Error handling
- `staticcheck` - Static analysis
- `gosec` - Security checks
- `godot` - Doc comments must end with period
- `gocyclo` - Cyclomatic complexity < 15
- `funlen` - Function length < 120 lines

If you need to disable a linter for a specific line, always add explanation:
```go
//nolint:gosec // G404: Using math/rand for non-cryptographic shuffle
```

### 7. Submit PR

```bash
# Commit changes
git add .
git commit -m "feat(tensor): add broadcasting support"

# Push branch
git push -u origin feat/your-feature
```

Then create a Pull Request on GitHub.

---

## Code Standards

### Go Style

Follow [Effective Go](https://go.dev/doc/effective_go) and these guidelines:

**Naming**:
| Type | Convention | Example |
|------|------------|---------|
| Exported | PascalCase | `TensorAdd`, `NewBackend` |
| Unexported | camelCase | `addFloat32`, `memoryPool` |
| Constants | PascalCase | `MaxBatchSize` |
| Interfaces | -er suffix when appropriate | `Backend`, `Optimizer` |

**Example**:
```go
// Backend defines the interface for tensor computation backends.
type Backend interface {
    Add(a, b *RawTensor) *RawTensor
    MatMul(a, b *RawTensor) *RawTensor
    Name() string
}

// CPUBackend implements Backend for CPU computation.
type CPUBackend struct {
    memoryPool *MemoryPool
    workers    int
}

// Add performs element-wise addition.
func (b *CPUBackend) Add(a, c *RawTensor) *RawTensor {
    result := b.allocate(a.Shape())
    addFloat32(result.data, a.data, c.data)
    return result
}
```

### Documentation

All exported functions/types **MUST** have doc comments:
- Start with the name of the thing being described
- Use complete sentences
- **End with a period** (godot linter enforces this)

```go
// Tensor represents a multi-dimensional array with type T and backend B.
// It provides operations for mathematical computations and automatic
// differentiation when used with an autodiff backend.
type Tensor[T DType, B Backend] struct {
    raw     *RawTensor
    backend B
}

// Add performs element-wise addition of two tensors.
// Returns a new tensor containing the result.
// Panics if shapes are incompatible.
func (t *Tensor[T, B]) Add(other *Tensor[T, B]) *Tensor[T, B] {
    // ...
}
```

### Error Handling

**Return errors** for expected failures:
```go
func LoadModel(path string) (*Model, error) {
    if _, err := os.Stat(path); os.IsNotExist(err) {
        return nil, fmt.Errorf("model file not found: %s", path)
    }
    // ...
}
```

**Panic** for programmer errors (invariant violations):
```go
func (t *Tensor[T, B]) Shape() Shape {
    if t.raw == nil {
        panic("tensor is nil")
    }
    return t.raw.shape
}
```

### Testing

**Test naming**: `TestTypeName_MethodName` or `TestFunctionName`

```go
func TestTensor_Add(t *testing.T) {
    t.Run("compatible shapes", func(t *testing.T) {
        // ...
    })

    t.Run("broadcasting", func(t *testing.T) {
        // ...
    })
}
```

**Table-driven tests** for multiple cases:
```go
func TestMatMul(t *testing.T) {
    tests := []struct {
        name   string
        aShape Shape
        bShape Shape
        want   Shape
    }{
        {"2x2 * 2x2", Shape{2, 2}, Shape{2, 2}, Shape{2, 2}},
        {"2x3 * 3x4", Shape{2, 3}, Shape{3, 4}, Shape{2, 4}},
        {"batch matmul", Shape{8, 2, 3}, Shape{8, 3, 4}, Shape{8, 2, 4}},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // ...
        })
    }
}
```

---

## Commit Guidelines

### Format

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (no logic change) |
| `refactor` | Code restructuring (no behavior change) |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks (deps, CI, etc.) |

### Examples

```
feat(tensor): implement broadcasting for binary ops

Adds NumPy-style broadcasting for Add, Sub, Mul, Div operations.
Broadcasting follows standard rules: dimensions are compared
right-to-left, and must be equal or one of them must be 1.

- Added broadcastShapes helper function
- Updated all binary ops to use broadcasting
- Added comprehensive test coverage

Closes #42
```

```
fix(webgpu): resolve buffer leak in compute shaders

Fixed GPU buffer not being released when compute shader
completes with error. Added proper cleanup in defer block.

Fixes #87
```

```
chore: update go-webgpu to v0.1.1

Updated GPU backend dependencies:
- go-webgpu/webgpu v0.1.0 → v0.1.1
- go-webgpu/goffi v0.3.1 → v0.3.3
```

---

## Pull Request Process

### Branch Protection

The `main` branch is protected:
- **Required**: All CI checks must pass
- **Required**: At least 1 approval (maintainers can override)
- Direct pushes are blocked (except for admins)

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests written and passing (`make test`)
- [ ] Linter passes (`make lint`)
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow guidelines
- [ ] No merge conflicts with main

### PR Description Template

```markdown
## Summary

Brief description of what this PR does.

## Changes

- Change 1
- Change 2

## Test plan

- [ ] Unit tests added/updated
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] Manual testing performed (if applicable)

## Related

Closes #123
```

### CI Pipeline

All PRs run through CI:

| Check | Description |
|-------|-------------|
| **Unit Tests** | Ubuntu, macOS, Windows (Go 1.25) |
| **Lint** | golangci-lint with project config |
| **Code Formatting** | gofmt verification |
| **Build Examples** | Compile all examples |
| **Build Tools** | Compile CLI tools |
| **Benchmarks** | Run performance benchmarks |
| **Codecov** | Coverage reporting |

### Review Process

1. CI runs automatically on PR creation
2. Maintainer reviews code
3. Feedback addressed (if any)
4. Approval and merge (squash merge preferred)

---

## Areas for Contribution

### High Priority

| Area | Description | Skills |
|------|-------------|--------|
| **Performance** | SIMD optimizations, GPU kernels | Go, WGSL |
| **Testing** | Increase coverage, edge cases | Go |
| **Documentation** | Tutorials, API docs | Technical writing |
| **ONNX Support** | More operator implementations | ML, Go |

### Medium Priority

| Area | Description | Skills |
|------|-------------|--------|
| **WebGPU Backend** | New GPU operations | WGSL, WebGPU |
| **Model Zoo** | Pre-trained model imports | ML |
| **Examples** | Real-world use cases | Go, ML |
| **Quantization** | INT8/INT4 support | ML, optimization |

### Always Welcome

- **Bug Reports** - Detailed issue reports with reproduction steps
- **Bug Fixes** - Any bug fixes are appreciated
- **Documentation** - Improvements, corrections, translations
- **Examples** - New examples demonstrating features

### First-Time Contributors

Look for issues labeled:
- `good-first-issue` - Simple, well-defined tasks
- `help-wanted` - Tasks where we need help
- `documentation` - Doc improvements

---

## Communication

### GitHub

- **Issues** - Bug reports, feature requests
- **Pull Requests** - Code contributions
- **Discussions** - Questions, ideas, RFC

### Getting Help

1. Check existing [Issues](https://github.com/born-ml/born/issues)
2. Search [Discussions](https://github.com/born-ml/born/discussions)
3. Read documentation in `docs/`
4. Ask in GitHub Discussions

---

## Code of Conduct

We are committed to providing a welcoming and respectful environment.

**Expected behavior**:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical merits
- Help others learn and grow

**Unacceptable behavior**:
- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments

Report issues to the maintainers via GitHub.

---

## License

By contributing to Born, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

## Thank You!

Every contribution makes Born better. Whether it's:
- Reporting a bug
- Fixing a typo
- Adding a feature
- Improving documentation

**You're helping build production-ready ML for Go.**

---

**Questions?** Open a [Discussion](https://github.com/born-ml/born/discussions) or check existing [Issues](https://github.com/born-ml/born/issues).

---

*Last Updated: 2025-12-24*
