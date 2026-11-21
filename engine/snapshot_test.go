package engine

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestSnapshotFloats(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, 3.14159, -2.71828, 1e-5, 123.456, -65504}
	for _, full := range []bool{false, true} {
		for _, shuffled := range []bool{false, true} {
			buf := encodeFloats(nil, vals, full, shuffled)
			out := make([]float32, len(vals))
			decodeFloats(out, buf, full, shuffled)
			for i := range vals {
				if full && out[i] != vals[i] {
					t.Fatalf("full shuffled=%v: %v -> %v", shuffled, vals[i], out[i])
				}
				if !full && out[i] != f16Table[f32ToF16(vals[i])] {
					t.Fatalf("half shuffled=%v: %v -> %v", shuffled, vals[i], out[i])
				}
			}
		}
	}
}

func TestSnapshotNotASnapshot(t *testing.T) {
	if _, err := ReadSnapshotInfo(strings.NewReader("hello, world")); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("got %v", err)
	}
}

// nextLogits continues from e (its first n tokens) with tok at position n and returns the logits.
func nextLogits(m *Model, e *prefixEntry, n, tok int) []float32 {
	s := m.newState(n+8, prefillBatch)
	m.restorePrefix(s, e, n)
	m.forward(s, []int{tok}, n)
	return append([]float32(nil), s.logits[:m.hp.vocab]...)
}

func softmax(x []float32) []float64 {
	top := x[argmax(x)]
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		out[i] = math.Exp(float64(v - top))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// A thread saved with SaveThread and loaded into a fresh copy of the model (a restart) continues exactly like the
// original: the next token's logits are bit for bit the same at full precision, and within rounding (same top token)
// at half. Generate on the loaded thread reads only the new turn. GGUF=<a Qwen3 .gguf> go test -run TestSnapshotThread -v
func TestSnapshotThread(t *testing.T) {
	m := loadTestModel(t, LoadOptions{ThreadMemory: 256 << 20})
	o := GenerateOptions{MaxTokens: 16, Thread: "chat1"}
	sys := Message{Role: "system", Content: "You are a terse assistant. " + strings.Repeat("Answer in one short sentence. ", 4)}
	q1 := Message{Role: "user", Content: "Name a color of the sky."}
	r1, err := m.Generate(context.Background(), m.ChatPrompt([]Message{sys, q1}), o, nil)
	if err != nil {
		t.Fatal(err)
	}
	orig := m.findThread("", "chat1", "")
	if orig == nil {
		t.Fatal("no thread kept")
	}
	n := len(orig.tokens)
	probe := m.Tokenize("\n")[0]
	want := nextLogits(m, orig, n, probe)
	p2 := m.ChatPrompt([]Message{sys, q1, {Role: "assistant", Content: r1.Text}, {Role: "user", Content: "And of grass?"}})
	base, err := m.Generate(context.Background(), p2, GenerateOptions{MaxTokens: 12}, nil) // no thread: reads it all
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		o    SnapshotOptions
	}{
		{"half", SnapshotOptions{}},
		{"half-compressed", SnapshotOptions{Compress: true}},
		{"full", SnapshotOptions{Full: true}},
		{"full-compressed", SnapshotOptions{Full: true, Compress: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := m.SaveThread(&buf, "", "chat1", tc.o); err != nil {
				t.Fatal(err)
			}
			size := buf.Len()
			info, err := ReadSnapshotInfo(bytes.NewReader(buf.Bytes()))
			if err != nil || info.Kind != "thread" || info.Fingerprint != m.Fingerprint() || info.Full != tc.o.Full || info.Compressed != tc.o.Compress {
				t.Fatalf("info: %+v %v", info, err)
			}

			m2 := loadTestModel(t, LoadOptions{ThreadMemory: 256 << 20}) // a restart
			info, err = m2.LoadThread(&buf, "app", "resumed", SnapshotOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if info.Tokens != n || len(info.Labels) != 1 || info.Labels[0] != "chat1" {
				t.Fatalf("loaded: %+v", info)
			}
			t.Logf("%d tokens: %d bytes (%.1f KB/token)", n, size, float64(size)/float64(n)/1024)
			got := m2.findThread("app", "resumed", "")
			if got == nil || !equalInts(got.tokens, orig.tokens) {
				t.Fatal("loaded thread's tokens differ")
			}
			logits := nextLogits(m2, got, n, probe)
			worst := 0.0
			for i := range want {
				worst = math.Max(worst, math.Abs(float64(want[i]-logits[i])))
			}
			if tc.o.Full && worst != 0 {
				t.Fatalf("full precision: logits differ by up to %g", worst)
			}
			// half precision moves logits by rounding; what matters is the distribution sampled from
			pw, pg := softmax(want), softmax(logits)
			pworst := 0.0
			for i := range pw {
				pworst = math.Max(pworst, math.Abs(pw[i]-pg[i]))
			}
			if pworst > 0.02 || argmax(logits) != argmax(want) {
				t.Fatalf("probabilities differ by up to %g, logits by %g (top %d vs %d)", pworst, worst, argmax(logits), argmax(want))
			}
			t.Logf("worst difference: logit %g, probability %g", worst, pworst)

			r2, err := m2.Generate(context.Background(), p2, GenerateOptions{MaxTokens: 12, Thread: "resumed", CacheScope: "app"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if first := r1.PromptTokens + r1.OutputTokens - 2; r2.CachedTokens < first {
				t.Fatalf("resumed turn: %d cached, want at least %d", r2.CachedTokens, first)
			}
			if tc.o.Full && r2.Text != base.Text {
				t.Fatalf("resumed answer %q, want %q", r2.Text, base.Text)
			}
		})
	}

	t.Run("refusals", func(t *testing.T) {
		var buf bytes.Buffer
		if err := m.SaveThread(&buf, "", "chat1", SnapshotOptions{Compress: true}); err != nil {
			t.Fatal(err)
		}
		good := buf.Bytes()
		load := func(b []byte) error {
			_, err := m.LoadThread(bytes.NewReader(b), "", "x", SnapshotOptions{})
			return err
		}
		other := append([]byte(nil), good...)
		other[8] ^= 1 // inside the fingerprint: another model's snapshot
		if err := load(other); !errors.Is(err, ErrSnapshotModel) {
			t.Fatalf("other model: %v", err)
		}
		damaged := append([]byte(nil), good...)
		damaged[len(damaged)/2] ^= 0x40
		if err := load(damaged); !errors.Is(err, ErrSnapshotCorrupt) {
			t.Fatalf("damaged: %v", err)
		}
		if err := load(good[:len(good)-10]); !errors.Is(err, ErrSnapshotCorrupt) {
			t.Fatalf("truncated: %v", err)
		}
		if _, _, err := m.LoadPromptCache(bytes.NewReader(good), "", ""); err == nil {
			t.Fatal("a thread loaded as a prompt cache")
		}
		if err := m.SaveThread(&buf, "", "nobody", SnapshotOptions{}); !errors.Is(err, ErrNoThread) {
			t.Fatalf("missing thread: %v", err)
		}
		mOff := loadTestModel(t, LoadOptions{}) // no thread memory, no folder: nowhere to keep it
		if _, err := mOff.LoadThread(bytes.NewReader(good), "", "x", SnapshotOptions{}); err == nil {
			t.Fatal("loaded with nowhere to keep it")
		}
		dir := t.TempDir()
		if _, err := mOff.LoadThread(bytes.NewReader(good), "", "x", SnapshotOptions{ThreadDir: dir}); err != nil {
			t.Fatal(err)
		}
		if list := mOff.Threads("", dir); len(list) != 1 || list[0].ID != "x" {
			t.Fatalf("on disk: %+v", list)
		}
		var again bytes.Buffer // a thread kept only on disk saves too
		if err := mOff.SaveThread(&again, "", "x", SnapshotOptions{ThreadDir: dir}); err != nil {
			t.Fatal(err)
		}
	})
}

// A warmed prompt start saved with SavePromptCache and loaded into a fresh copy of the model is reused by the next
// call, pinned. GGUF=<a Qwen3 .gguf> go test -run TestSnapshotPromptCache -v
func TestSnapshotPromptCache(t *testing.T) {
	m := loadTestModel(t, LoadOptions{PromptCache: 64 << 20})
	pre := m.ChatPrompt([]Message{{Role: "system", Content: "You help with Acme Notes. " + strings.Repeat("Notes sync in seconds. ", 10)}})
	pre = strings.TrimSuffix(pre, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
	w, err := m.Warm(context.Background(), pre, "tenant", "")
	if err != nil || w.Kept == 0 {
		t.Fatalf("warm: %+v %v", w, err)
	}
	var buf bytes.Buffer
	if n, err := m.SavePromptCache(&buf, "tenant", SnapshotOptions{Compress: true}); err != nil || n != 1 {
		t.Fatalf("save: %d %v", n, err)
	}
	var empty bytes.Buffer
	if n, err := m.SavePromptCache(&empty, "someone-else", SnapshotOptions{}); err != nil || n != 0 {
		t.Fatalf("other scope: %d %v", n, err)
	}

	m2 := loadTestModel(t, LoadOptions{PromptCache: 64 << 20})
	info, kept, err := m2.LoadPromptCache(&buf, "tenant", "boot")
	if err != nil || kept != w.Kept || info.Entries != 1 {
		t.Fatalf("load: %+v kept %d (want %d) %v", info, kept, w.Kept, err)
	}
	if st := m2.PromptCache(); st.Pinned == 0 || st.Entries != 1 {
		t.Fatalf("not pinned: %+v", st)
	}
	r, err := m2.Generate(context.Background(), pre+"<|im_start|>user\nHow fast?<|im_end|>\n<|im_start|>assistant\n", GenerateOptions{MaxTokens: 2, CacheScope: "tenant"}, nil)
	if err != nil || r.CachedTokens < kept-1 {
		t.Fatalf("after load: %d cached, want about %d (%v)", r.CachedTokens, kept, err)
	}
	r, err = m2.Generate(context.Background(), pre+"<|im_start|>user\nHow fast?<|im_end|>\n<|im_start|>assistant\n", GenerateOptions{MaxTokens: 2, CacheScope: "other"}, nil)
	if err != nil || r.CachedTokens != 0 {
		t.Fatalf("another scope reused it: %d cached (%v)", r.CachedTokens, err)
	}
}

// Restoring a long thread from a snapshot against reading the conversation again.
// GGUF=<a Qwen3 .gguf> SNAPBENCH=1 go test -run TestSnapshotTiming -v
func TestSnapshotTiming(t *testing.T) {
	if os.Getenv("SNAPBENCH") == "" {
		t.Skip("set SNAPBENCH=1")
	}
	m := loadTestModel(t, LoadOptions{ThreadMemory: 1 << 30})
	var sb strings.Builder
	for i := 0; sb.Len() < 9000; i++ {
		fmt.Fprintf(&sb, "Ticket %d: the customer says the export to CSV drops the last row when the file has a trailing comma. ", i)
	}
	prompt := m.ChatPrompt([]Message{{Role: "system", Content: "Summarize tickets."}, {Role: "user", Content: sb.String()}})
	start := time.Now()
	r, err := m.Generate(context.Background(), prompt, GenerateOptions{MaxTokens: 1, Thread: "long"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	read := time.Since(start)
	t.Logf("reading %d prompt tokens: %v", r.PromptTokens, read)
	for _, o := range []SnapshotOptions{{}, {Compress: true}, {Full: true}, {Full: true, Compress: true}} {
		var buf bytes.Buffer
		start = time.Now()
		if err := m.SaveThread(&buf, "", "long", o); err != nil {
			t.Fatal(err)
		}
		save := time.Since(start)
		size := buf.Len()
		start = time.Now()
		if _, err := m.LoadThread(&buf, "", "restored", o); err != nil {
			t.Fatal(err)
		}
		load := time.Since(start)
		t.Logf("full=%v compress=%v: %.1f MB, save %v, load %v (%.0fx faster than reading)", o.Full, o.Compress, float64(size)/1e6, save.Round(time.Millisecond), load.Round(time.Millisecond), float64(read)/float64(load))
	}
}
