package engine

import (
	"context"
	"math"
	"testing"
)

// State on the tiny model in every go test: Fork, Rewind and StepBranches leave states that continue like a state
// that read the same tokens in the same passes, and memory comes back on Close.
func TestStatePlumbing(t *testing.T) {
	r := NewResources(ResourceOptions{})
	defer r.Close()
	m, err := Load(testModelPath(t), LoadOptions{Resources: r, Deterministic: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	ids := m.Tokenize("the quick brown fox jumps over the lazy dog")
	more := m.Tokenize(" and runs away")
	want := stepTwice(t, m, ids, more)

	base, err := m.NewState(StateOptions{MaxContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := base.Step(ctx, ids...); err != nil {
		t.Fatal(err)
	}
	f, err := base.Fork()
	if err != nil {
		t.Fatal(err)
	}
	// branches don't change the state
	before := append([]float32(nil), base.Logits()...)
	if err := base.StepBranches(ctx, [][]int{more, ids[:3], nil}, func(b, i int, logits []float32) {}); err != nil {
		t.Fatal(err)
	}
	sameBitsExact(t, "logits after StepBranches", before, base.Logits())
	if _, err := base.Step(ctx, ids...); err != nil {
		t.Fatal(err)
	}
	got, err := f.Step(ctx, more...)
	if err != nil {
		t.Fatal(err)
	}
	sameBitsExact(t, "fork", want, got)
	if err := base.Rewind(len(ids)); err != nil {
		t.Fatal(err)
	}
	if base.Logits() != nil {
		t.Errorf("logits after Rewind")
	}
	got, err = base.Step(ctx, more...)
	if err != nil {
		t.Fatal(err)
	}
	sameBitsExact(t, "rewind", want, got)

	if _, err := f.Step(ctx, make([]int, 300)...); err == nil {
		t.Errorf("stepping past MaxContext: no error")
	}
	base.Close()
	f.Close()
	if _, err := f.Step(ctx, 1); err != ErrStateClosed {
		t.Errorf("Step after Close: %v", err)
	}
	m.dropIdleStates()
}

func stepTwice(t *testing.T, m *Model, a, b []int) []float32 {
	t.Helper()
	st, err := m.NewState(StateOptions{MaxContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Step(context.Background(), a...); err != nil {
		t.Fatal(err)
	}
	l, err := st.Step(context.Background(), b...)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), l...)
}

func sameBitsExact(t *testing.T, what string, want, got []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s: logit %d is %v, want %v", what, i, got[i], want[i])
		}
	}
}

// Fork on the GPU (unified memory, and the discrete path forced on this GPU): the fork lands in a pooled state whose
// device cache last ended where the fork's next pass starts, so a discrete GPU must upload the copied cache.
func TestStateForkGPU(t *testing.T) {
	for _, discrete := range []bool{false, true} {
		m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
		if err != nil {
			t.Fatal(err)
		}
		gpuForceDiscrete = discrete
		g, err := newGPUModel(m, false)
		gpuForceDiscrete = false
		if err != nil {
			m.Close()
			t.Skip(err)
		}
		m.gpu = g
		ctx := context.Background()
		ids := m.Tokenize(ChatPrompt([]Message{{Role: "user", Content: "Write a haiku about rain on a tin roof."}}))
		more := m.Tokenize("Rain drums on the tin")
		want := stepTwice(t, m, ids, more)

		other := make([]int, len(ids)) // a state that ends where the fork will continue, then goes back to the pool
		copy(other, ids)
		other[3]++
		p, _ := m.NewState(StateOptions{MaxContext: 256})
		if _, err := p.Step(ctx, other...); err != nil {
			t.Fatal(err)
		}
		base, _ := m.NewState(StateOptions{MaxContext: 256})
		p.Close()
		if _, err := base.Step(ctx, ids...); err != nil {
			t.Fatal(err)
		}
		f, err := base.Fork()
		if err != nil {
			t.Fatal(err)
		}
		got, err := f.Step(ctx, more...)
		if err != nil {
			t.Fatal(err)
		}
		sameBitsExact(t, map[bool]string{false: "unified fork", true: "discrete fork"}[discrete], want, got)
		f.Close()
		base.Close()
		m.gpu = nil
		g.close()
		m.Close()
	}
}
