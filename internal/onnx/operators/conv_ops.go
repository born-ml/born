//go:build !wasm

package operators

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// registerConvOps registers the ONNX Conv operator.
func (r *Registry) registerConvOps() {
	r.Register("Conv", handleConv)
}

// convParams holds the resolved Conv attributes.
type convParams struct {
	stride                 int
	padT, padL, padB, padR int
	group                  int
}

func (p convParams) hasPad() bool {
	// pads are validated non-negative, so a positive sum means some padding.
	return p.padT+p.padL+p.padB+p.padR > 0
}

// handleConv implements ONNX Conv for 4D NCHW float32 input.
//
// It reuses born's Conv2D kernel: input is explicitly zero-padded to handle
// asymmetric pads, grouped/depthwise convolution is done by splitting channels
// per group and concatenating, and the optional bias is broadcast over the
// output channels. Unsupported attributes (auto_pad=SAME_*, dilations>1,
// non-square strides) are rejected rather than silently producing wrong output.
func handleConv(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	x, w, err := convInputs(inputs)
	if err != nil {
		return nil, err
	}
	p, err := parseConvParams(node)
	if err != nil {
		return nil, err
	}

	xp := x
	if p.hasPad() {
		xp, err = padNCHW(x, p.padT, p.padL, p.padB, p.padR)
		if err != nil {
			return nil, err
		}
	}

	out, err := convForward(ctx, xp, w, p)
	if err != nil {
		return nil, err
	}

	if len(inputs) >= 3 && inputs[2] != nil {
		if err := addConvBias(out, inputs[2]); err != nil {
			return nil, err
		}
	}
	return []*tensor.RawTensor{out}, nil
}

func convInputs(inputs []*tensor.RawTensor) (x, w *tensor.RawTensor, err error) {
	if len(inputs) < 2 || inputs[0] == nil || inputs[1] == nil {
		return nil, nil, fmt.Errorf("conv: requires input and weight")
	}
	x, w = inputs[0], inputs[1]
	if x.DType() != tensor.Float32 || w.DType() != tensor.Float32 {
		return nil, nil, fmt.Errorf("conv: only float32 supported")
	}
	if len(x.Shape()) != 4 {
		return nil, nil, fmt.Errorf("conv: expected 4D NCHW input, got %dD", len(x.Shape()))
	}
	if len(w.Shape()) != 4 {
		return nil, nil, fmt.Errorf("conv: expected 4D weight [Cout,Cin/group,kH,kW], got %dD", len(w.Shape()))
	}
	return x, w, nil
}

func parseConvParams(node *Node) (convParams, error) {
	var p convParams
	if err := rejectUnsupportedConvAttrs(node); err != nil {
		return p, err
	}
	stride, err := parseConvStride(node)
	if err != nil {
		return p, err
	}
	p.stride = stride
	if p.padT, p.padL, p.padB, p.padR, err = parseConvPads(node); err != nil {
		return p, err
	}
	// auto_pad=VALID means "no padding" and, per the ONNX spec, takes
	// precedence over any explicit pads attribute. Force the pads to zero so a
	// model that sets both does not silently get the explicit pads applied.
	if GetAttrString(node, "auto_pad", autoPadNotset) == autoPadValid {
		p.padT, p.padL, p.padB, p.padR = 0, 0, 0, 0
	}
	p.group = int(GetAttrInt(node, "group", 1))
	if p.group < 1 {
		return p, fmt.Errorf("conv: invalid group %d", p.group)
	}
	return p, nil
}

func rejectUnsupportedConvAttrs(node *Node) error {
	if ap := GetAttrString(node, "auto_pad", autoPadNotset); ap != autoPadNotset && ap != autoPadValid && ap != "" {
		return fmt.Errorf("conv: auto_pad=%q not supported (use explicit pads)", ap)
	}
	for _, d := range GetAttrInts(node, "dilations") {
		if d != 1 {
			return fmt.Errorf("conv: dilations>1 not supported")
		}
	}
	return nil
}

func parseConvStride(node *Node) (int, error) {
	sh, sw := 1, 1
	if s := GetAttrInts(node, "strides"); len(s) == 2 {
		sh, sw = int(s[0]), int(s[1])
	} else if len(s) != 0 {
		return 0, fmt.Errorf("conv: strides must have 2 entries, got %v", s)
	}
	if sh != sw {
		return 0, fmt.Errorf("conv: non-square stride %d,%d not supported", sh, sw)
	}
	if sh <= 0 {
		return 0, fmt.Errorf("conv: invalid stride %d", sh)
	}
	return sh, nil
}

func parseConvPads(node *Node) (t, l, b, r int, err error) {
	if pd := GetAttrInts(node, "pads"); len(pd) == 4 {
		t, l, b, r = int(pd[0]), int(pd[1]), int(pd[2]), int(pd[3])
	} else if len(pd) != 0 {
		return 0, 0, 0, 0, fmt.Errorf("conv: pads must have 4 entries for 2D, got %v", pd)
	}
	if t < 0 || l < 0 || b < 0 || r < 0 {
		return 0, 0, 0, 0, fmt.Errorf("conv: negative pads not supported")
	}
	return t, l, b, r, nil
}

func convForward(ctx *Context, xp, w *tensor.RawTensor, p convParams) (*tensor.RawTensor, error) {
	if err := validateConvShapes(xp, w, p); err != nil {
		return nil, err
	}
	if p.group == 1 {
		return ctx.Backend.Conv2D(xp, w, p.stride, 0), nil
	}
	return groupedConv2D(ctx, xp, w, p.stride, p.group)
}

// validateConvShapes checks channel agreement and positive output dims so the
// Conv2D backend kernel (which panics on these) is never reached with bad
// shapes; both the group==1 and grouped paths return a clean error instead.
func validateConvShapes(xp, w *tensor.RawTensor, p convParams) error {
	cin := xp.Shape()[1]
	cout := w.Shape()[0]
	if cin%p.group != 0 || cout%p.group != 0 {
		return fmt.Errorf("conv: channels (in=%d out=%d) not divisible by group %d", cin, cout, p.group)
	}
	if w.Shape()[1] != cin/p.group {
		return fmt.Errorf("conv: weight in-channels %d != input %d / group %d", w.Shape()[1], cin, p.group)
	}
	hp, wp := xp.Shape()[2], xp.Shape()[3]
	kh, kw := w.Shape()[2], w.Shape()[3]
	if (hp-kh)/p.stride+1 <= 0 || (wp-kw)/p.stride+1 <= 0 {
		return fmt.Errorf("conv: kernel %dx%d too large for padded input %dx%d", kh, kw, hp, wp)
	}
	return nil
}

// padNCHW returns a zero-padded copy of a 4D NCHW float32 tensor.
func padNCHW(x *tensor.RawTensor, padT, padL, padB, padR int) (*tensor.RawTensor, error) {
	s := x.Shape()
	n, c, h, w := s[0], s[1], s[2], s[3]
	hp, wp := h+padT+padB, w+padL+padR
	// Propagate the input's device and dtype rather than hardcoding CPU/Float32
	// so this stays correct once Conv runs on non-CPU devices. The caller has
	// already validated x is float32; the float32 copy below relies on that.
	out, err := tensor.NewRaw(tensor.Shape{n, c, hp, wp}, x.DType(), x.Device())
	if err != nil {
		return nil, fmt.Errorf("conv: %w", err)
	}
	src := x.AsFloat32()
	dst := out.AsFloat32() // zero-initialized
	for ni := range n {
		for ci := range c {
			sBase := (ni*c + ci) * h * w
			dBase := (ni*c+ci)*hp*wp + padT*wp + padL
			for ih := range h {
				copy(dst[dBase+ih*wp:dBase+ih*wp+w], src[sBase+ih*w:sBase+ih*w+w])
			}
		}
	}
	return out, nil
}

// groupedConv2D splits input channels and weights into `group` groups, runs
// Conv2D per group, and concatenates the outputs along the channel axis.
func groupedConv2D(ctx *Context, xp, w *tensor.RawTensor, stride, group int) (*tensor.RawTensor, error) {
	// Channel/group agreement is validated by validateConvShapes before this.
	xs := ctx.Backend.Chunk(xp, group, 1)
	ws := ctx.Backend.Chunk(w, group, 0)
	if len(xs) != group || len(ws) != group {
		return nil, fmt.Errorf("conv: chunk produced %d/%d groups, want %d", len(xs), len(ws), group)
	}
	outs := make([]*tensor.RawTensor, group)
	for g := range group {
		outs[g] = ctx.Backend.Conv2D(xs[g], ws[g], stride, 0)
	}
	return ctx.Backend.Cat(outs, 1), nil
}

// addConvBias adds a per-output-channel bias in place to an NCHW output.
func addConvBias(out, bias *tensor.RawTensor) error {
	if bias.DType() != tensor.Float32 {
		return fmt.Errorf("conv: bias must be float32")
	}
	s := out.Shape()
	n, co, h, w := s[0], s[1], s[2], s[3]
	b := bias.AsFloat32()
	if len(b) != co {
		return fmt.Errorf("conv: bias length %d != output channels %d", len(b), co)
	}
	d := out.AsFloat32() // direct view into out; the in-place mutation is intentional
	plane := h * w
	for ni := range n {
		for ci := range co {
			base := (ni*co + ci) * plane
			bv := b[ci]
			for k := range plane {
				d[base+k] += bv
			}
		}
	}
	return nil
}
