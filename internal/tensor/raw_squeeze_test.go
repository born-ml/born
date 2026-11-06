package tensor

import (
	"slices"
	"testing"
)

func TestRawSqueeze(t *testing.T) {
	tests := []struct {
		name    string
		shape   Shape
		axes    []int
		want    Shape
		wantErr bool
	}{
		{"all size-1 dims", Shape{1, 3, 1, 4}, nil, Shape{3, 4}, false},
		{"no size-1 dims", Shape{2, 3}, nil, Shape{2, 3}, false},
		{"all ones to scalar", Shape{1, 1}, nil, Shape{}, false},
		{"one axis", Shape{1, 3, 1, 4}, []int{2}, Shape{1, 3, 4}, false},
		{"negative axis", Shape{2, 3, 1}, []int{-1}, Shape{2, 3}, false},
		{"several axes", Shape{1, 3, 1, 4}, []int{0, 2}, Shape{3, 4}, false},
		{"duplicate axis", Shape{1, 3}, []int{0, -2}, Shape{3}, false},
		{"axis not size 1", Shape{2, 3}, []int{1}, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x, err := NewRaw(tt.shape, Float32, CPU)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Squeeze(x, tt.axes...)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Squeeze(%v, %v): want error, got shape %v", tt.shape, tt.axes, got.Shape())
				}
				return
			}
			if err != nil {
				t.Fatalf("Squeeze(%v, %v): %v", tt.shape, tt.axes, err)
			}
			if !slices.Equal(got.Shape(), tt.want) {
				t.Errorf("Squeeze(%v, %v) shape = %v, want %v", tt.shape, tt.axes, got.Shape(), tt.want)
			}
		})
	}
}

func TestRawSqueezeNil(t *testing.T) {
	if _, err := Squeeze(nil); err == nil {
		t.Fatal("Squeeze(nil): want error")
	}
}
