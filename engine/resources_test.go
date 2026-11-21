package engine

import (
	"context"
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"
)

// testModel is a model with hyperparameters but no weights, enough for the memory accounting.
func testModel(r *Resources) *Model {
	m := &Model{hp: hparams{dim: 64, layers: 2, heads: 4, kvHeads: 2, headDim: 16, ffn: 128, vocab: 1000, ctx: 4096}, res: r, pool: r.pool}
	m.prefix.budget, m.threads.budget = 1<<30, 1<<30
	if err := r.attach(m); err != nil {
		panic(err)
	}
	return m
}

// sliceBytes sums the backing memory of every slice reachable from v (not following maps).
func sliceBytes(v reflect.Value) int64 {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return 0
		}
		return sliceBytes(v.Elem())
	case reflect.Struct:
		var n int64
		for i := 0; i < v.NumField(); i++ {
			n += sliceBytes(v.Field(i))
		}
		return n
	case reflect.Slice:
		n := int64(v.Len()) * int64(v.Type().Elem().Size())
		if k := v.Type().Elem().Kind(); k == reflect.Slice || k == reflect.Pointer || k == reflect.Struct {
			for i := 0; i < v.Len(); i++ {
				n += sliceBytes(v.Index(i))
			}
			if k == reflect.Slice {
				n -= int64(v.Len()) * int64(v.Type().Elem().Size()) // count the headers' data, not the headers
			}
		}
		return n
	}
	return 0
}

// stateBytes, which the budget reserves, is what newState really allocates.
func TestStateBytes(t *testing.T) {
	r := NewResources(ResourceOptions{Threads: 3})
	defer r.Close()
	m := testModel(r)
	defer r.detach(m)
	for _, c := range [][2]int{{256, 1}, {512, 3}, {4096, 512}, {1000, 7}} {
		s := m.newState(c[0], c[1])
		if got, want := s.bytes, sliceBytes(reflect.ValueOf(s)); got != want {
			t.Errorf("state %v: stateBytes %d, allocated %d", c, got, want)
		}
	}
}

func TestMemoryError(t *testing.T) {
	r := NewResources(ResourceOptions{Threads: 1, Memory: 1000})
	defer r.Close()
	if err := r.reserve(600, "a"); err != nil {
		t.Fatal(err)
	}
	err := r.reserve(600, "loading the model")
	var me *MemoryError
	if !errors.Is(err, ErrMemoryLimit) || !errors.As(err, &me) || me.Need != 600 || me.Used != 600 || me.Limit != 1000 {
		t.Fatalf("got %v", err)
	}
	r.release(600)
	if err := r.reserve(1000, "b"); err != nil {
		t.Fatalf("after release: %v", err)
	}
	if u := r.Usage(); u.MemoryUsed != 1000 || u.MemoryLimit != 1000 {
		t.Errorf("usage %+v", u)
	}
	unlimited := NewResources(ResourceOptions{Threads: 1})
	defer unlimited.Close()
	if err := unlimited.reserve(1<<50, "x"); err != nil {
		t.Errorf("no limit: %v", err)
	}
}

// When the budget is short, idle states go first, then cache entries least recently used first across models; pinned
// entries stay. A cache entry that still doesn't fit isn't kept.
func TestReclaim(t *testing.T) {
	r := NewResources(ResourceOptions{Threads: 2})
	defer r.Close()
	a, b := testModel(r), testModel(r)
	per := a.tokenBytes()
	entry := func(m *Model, n int, at time.Time, pin string) *prefixEntry {
		e := &prefixEntry{tokens: make([]int, n), used: at}
		if pin != "" {
			e.pins = map[string]bool{pin: true}
		}
		m.prefix.entries = append(m.prefix.entries, e)
		m.prefix.used += int64(n) * per
		r.used += int64(n) * per
		return e
	}
	now := time.Now()
	oldA := entry(a, 100, now.Add(-3*time.Minute), "")
	pinned := entry(b, 100, now.Add(-4*time.Minute), "warm")
	oldB := entry(b, 100, now.Add(-2*time.Minute), "")
	newA := entry(a, 100, now.Add(-time.Minute), "")
	th := &prefixEntry{tokens: make([]int, 100)}
	a.threads.mem = map[string]*threadEntry{"x": {e: th, bytes: 100 * per, used: now}}
	a.threads.used += 100 * per
	r.used += 100 * per
	idle := a.newState(256, 1)
	a.states = []*state{idle}
	r.used += idle.bytes

	r.limit = r.used // full
	if err := r.reserve(idle.bytes+150*per, "test"); err != nil {
		t.Fatal(err)
	}
	if len(a.states) != 0 {
		t.Error("the idle state should go first")
	}
	has := func(m *Model, e *prefixEntry) bool {
		for _, c := range m.prefix.entries {
			if c == e {
				return true
			}
		}
		return false
	}
	if has(a, oldA) || has(b, oldB) {
		t.Error("the two least recently used unpinned entries should be evicted")
	}
	if !has(b, pinned) || !has(a, newA) || a.threads.mem["x"] == nil {
		t.Error("the pinned entry, the newest entry and the newer thread should stay")
	}
	if r.used > r.limit {
		t.Errorf("used %d over the limit %d", r.used, r.limit)
	}
	// nothing more to free: a load fails, a cache entry isn't kept
	r.limit = r.used + 10
	if err := r.reserve(int64(1<<40), "loading the model"); !errors.Is(err, ErrMemoryLimit) {
		t.Errorf("want ErrMemoryLimit, got %v", err)
	}
	r.detach(a)
	r.detach(b)
}

// Resources stop their worker threads only once the last model sharing them is gone.
func TestResourcesClose(t *testing.T) {
	r := NewResources(ResourceOptions{Threads: 2})
	m := testModel(r)
	r.Close()
	if r.pool.closed.Load() {
		t.Fatal("closed while a model still uses it")
	}
	if err := r.attach(&Model{}); err == nil {
		t.Error("attach after Close should fail")
	}
	r.detach(m)
	if !r.pool.closed.Load() {
		t.Error("not closed after the last model went")
	}
}

// With LowPriority, the work runs on a thread of lowered priority, never the caller's own, and the caller's thread
// keeps its priority.
func TestLowPriorityCall(t *testing.T) {
	if !LowPrioritySupported() {
		t.Skip("not supported on this OS")
	}
	r := NewResources(ResourceOptions{Threads: 2, LowPriority: true})
	defer r.Close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	before := threadLowered(t)
	if before {
		t.Skip("the test's thread is already low priority")
	}
	var inside bool
	r.call(func() { inside = threadLowered(t) })
	if !inside {
		t.Error("the work's thread wasn't lowered")
	}
	if threadLowered(t) {
		t.Error("the caller's thread was lowered")
	}
	// the workers too
	var mu sync.Mutex
	lowered := 0
	r.pool.run(func(w int) {
		if w > 0 && threadLowered(t) {
			mu.Lock()
			lowered++
			mu.Unlock()
		}
	})
	if lowered != r.pool.n-1 {
		t.Errorf("%d of %d workers lowered", lowered, r.pool.n-1)
	}
	if u := r.Usage(); !u.LowPriority {
		t.Errorf("usage says not low priority: %+v", u)
	}
	// a panic in the work reaches the caller
	defer func() {
		if recover() == nil {
			t.Error("panic lost")
		}
	}()
	r.call(func() { panic("boom") })
}

// Two models share one budget: their threads (Info.Threads), and their memory, which the load, generations and caches
// stay inside, and which closing gives back. GGUF=<a Qwen3 .gguf> go test -run TestSharedResources -v
func TestSharedResources(t *testing.T) {
	path := os.Getenv("GGUF")
	if path == "" {
		t.Skip("set GGUF to a Qwen3 GGUF file")
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	size := st.Size()

	// too small for the weights: the load fails
	r := NewResources(ResourceOptions{Threads: 2, Memory: size / 2})
	if _, err := Load(path, LoadOptions{Resources: r}); !errors.Is(err, ErrMemoryLimit) {
		t.Fatalf("load over budget: want ErrMemoryLimit, got %v", err)
	}
	if u := r.Usage(); u.MemoryUsed != 0 || u.Models != 0 {
		t.Fatalf("after a failed load: %+v", u)
	}
	r.Close()
	if _, err := Load(path, LoadOptions{Resources: NewResources(ResourceOptions{}), Threads: 2}); err == nil {
		t.Error("Resources with Threads: want an error")
	}

	// room for two models and a little more
	r = NewResources(ResourceOptions{Threads: 2, Memory: 2*size + 512<<20})
	defer r.Close()
	a, err := Load(path, LoadOptions{Resources: r, PromptCache: 1 << 30})
	if err != nil {
		t.Fatal(err)
	}
	b, err := Load(path, LoadOptions{Resources: r, PromptCache: 1 << 30})
	if err != nil {
		t.Fatal(err)
	}
	if a.Info().Threads != 2 || b.Info().Threads != 2 || a.pool != b.pool {
		t.Fatalf("threads: %d %d, shared pool %v", a.Info().Threads, b.Info().Threads, a.pool == b.pool)
	}
	ctx := context.Background()
	prompt := a.ChatPrompt([]Message{{Role: "user", Content: "Name three colors of the rainbow, and nothing else at all please."}})
	var wg sync.WaitGroup
	errs := make([]error, 4)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m := a
			if i%2 == 1 {
				m = b
			}
			_, errs[i] = m.Generate(ctx, prompt, GenerateOptions{MaxTokens: 8}, nil)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("generate %d: %v", i, err)
		}
	}
	u := r.Usage()
	if u.MemoryUsed > u.MemoryLimit || u.MemoryUsed < 2*size || u.Models != 2 {
		t.Errorf("usage after generating: %+v", u)
	}

	// a context too long for what's left: the generation fails, after freeing what it could
	_, err = a.Generate(ctx, prompt, GenerateOptions{MaxTokens: 30000}, nil)
	if !errors.Is(err, ErrMemoryLimit) {
		t.Errorf("long generation: want ErrMemoryLimit, got %v", err)
	}
	if a.PromptCache().Bytes != 0 || b.PromptCache().Bytes != 0 {
		t.Errorf("the prompt caches should have been freed for it: %d, %d", a.PromptCache().Bytes, b.PromptCache().Bytes)
	}
	if _, err := a.Generate(ctx, prompt, GenerateOptions{MaxTokens: 4}, nil); err != nil {
		t.Errorf("short generation after: %v", err)
	}

	a.Close()
	b.Close()
	if u := r.Usage(); u.MemoryUsed != 0 || u.Models != 0 {
		t.Errorf("after closing: %+v", u)
	}
}

// A model's own budget, from LoadOptions, works the same way and closes with the model.
func TestOwnResources(t *testing.T) {
	path := os.Getenv("GGUF")
	if path == "" {
		t.Skip("set GGUF to a Qwen3 GGUF file")
	}
	m, err := Load(path, LoadOptions{Threads: 2, MaxMemory: 8 << 30, LowPriority: LowPrioritySupported()})
	if err != nil {
		t.Fatal(err)
	}
	r := m.res
	res, err := m.Generate(context.Background(), m.ChatPrompt([]Message{{Role: "user", Content: "Say hi."}}), GenerateOptions{MaxTokens: 4}, nil)
	if err != nil || res.OutputTokens == 0 {
		t.Fatalf("%v %+v", err, res)
	}
	if u := r.Usage(); u.LowPriority != LowPrioritySupported() {
		t.Errorf("usage %+v", u)
	}
	m.Close()
	if !r.pool.closed.Load() {
		t.Error("the model's own Resources should close with it")
	}
}
