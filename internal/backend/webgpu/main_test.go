//go:build windows

package webgpu

import (
	"os"
	"testing"
)

var computeAvailable bool

func TestMain(m *testing.M) {
	computeAvailable = IsAvailable()
	os.Exit(m.Run())
}
