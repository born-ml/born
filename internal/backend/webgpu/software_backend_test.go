//go:build windows || linux

package webgpu

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/gogpu/gputypes"
)

// TestNewSoftwareBackendUsesCPUAdapter checks that GOGPU_GRAPHICS_API=software
// yields the CPU (software) adapter. CI runs every GPU test this way.
func TestNewSoftwareBackendUsesCPUAdapter(t *testing.T) {
	if testing.Short() {
		t.Skip("software backend test skipped in -short mode")
	}
	t.Setenv("GOGPU_GRAPHICS_API", "software")

	b, err := New()
	if err != nil {
		t.Fatalf("New() with GOGPU_GRAPHICS_API=software: %v", err)
	}
	defer b.Release()

	info := b.AdapterInfo()
	if info == nil {
		t.Fatal("AdapterInfo() = nil")
	}
	if info.DeviceType != gputypes.DeviceTypeCPU {
		t.Errorf("DeviceType = %v, want %v (adapter %q)", info.DeviceType, gputypes.DeviceTypeCPU, info.Name)
	}
}

// TestBackendUsesOnlyPublicWGPUAPI enforces ADR-005: the backend talks to
// gogpu/wgpu through its public root API only. core/ and hal/ are wgpu's
// implementation and are moving under internal/, so any import of them will
// break the build. hal/allbackends is the one exception for now: it only
// registers the backends with a blank import, and goes when wgpu offers a
// public way to do that.
func TestBackendUsesOnlyPublicWGPUAPI(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		for _, bad := range internalWGPUImports(t, fset, name) {
			t.Errorf("%s: use the public github.com/gogpu/wgpu API (ADR-005)", bad)
		}
	}
}

// internalWGPUImports returns "position imports path" for every import of
// wgpu/hal/... or wgpu/core/... in the named file, except hal/allbackends.
func internalWGPUImports(t *testing.T, fset *token.FileSet, name string) []string {
	t.Helper()
	const allowed = "github.com/gogpu/wgpu/hal/allbackends"
	forbidden := []string{"github.com/gogpu/wgpu/hal", "github.com/gogpu/wgpu/core"}

	src, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	f, err := parser.ParseFile(fset, name, src, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	var found []string
	for _, imp := range f.Imports {
		path, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			t.Fatalf("%s: bad import %s", name, imp.Path.Value)
		}
		if path == allowed {
			continue
		}
		for _, prefix := range forbidden {
			if path == prefix || strings.HasPrefix(path, prefix+"/") {
				found = append(found, fset.Position(imp.Pos()).String()+" imports "+path)
			}
		}
	}
	return found
}
