package engine

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Resources is a budget of CPU threads and memory that models share, so an app embedding nollama can keep it from
// starving the rest of the app. Pass one to LoadOptions.Resources for every model that should share it; a model
// loaded without one gets its own, sized by LoadOptions.Threads, MaxMemory and LowPriority.
//
// Threads: the models share one set of worker threads. Work is handed to them one step at a time, so concurrent
// generations (on one model or several) take turns on the same threads instead of adding their own. Each generation
// also runs on its caller's goroutine, which takes part in the work, so N concurrent calls can briefly use up to
// Threads + N - 1 cores; bound concurrency (the server's Parallel) to bound that.
//
// Memory: the budget covers what the engine allocates in bulk: model weights (the file's size: they're mapped from
// the file, and every token reads all of them, so they stay resident), each generation's keys and values and scratch
// buffers, idle generation states kept for reuse, the prompt cache and threads kept in memory. When something new
// doesn't fit, the engine first frees what it can rebuild (idle states, then the least recently used prompt cache
// entries and threads, across all models sharing the budget). A load or a generation that still doesn't fit fails
// with an error that matches ErrMemoryLimit; a cache entry that doesn't fit is simply not kept. Go's own heap overhead,
// the tokenizer and small per-call buffers aren't counted: leave some headroom, and consider debug.SetMemoryLimit
// (GOMEMLIMIT) for the process.
//
// Priority: with LowPriority, the worker threads, and the thread each generation runs on, ask the OS for a lower
// scheduling priority, so the host app's threads win when the CPU is busy. See LowPrioritySupported for what each OS
// offers.
type Resources struct {
	pool        *pool
	threads     int
	lowPriority bool
	lowered     atomic.Int64 // threads whose priority was lowered
	lowerFailed atomic.Int64 // threads where the OS refused

	mu      sync.Mutex
	limit   int64 // bytes; <= 0: no limit
	used    int64
	models  map[*Model]struct{}
	closing bool
	closed  bool
}

// ResourceOptions configures NewResources.
type ResourceOptions struct {
	Threads int // worker threads; 0: the number of CPUs Go uses (GOMAXPROCS), capped at 8
	// Memory is the budget in bytes for weights, generation states and caches of every model sharing these
	// Resources; 0: no limit.
	Memory int64
	// LowPriority lowers the OS scheduling priority of the threads doing the work (see LowPrioritySupported).
	LowPriority bool
}

// ResourceUsage reports a Resources' budget and what's used.
type ResourceUsage struct {
	Threads     int
	MemoryLimit int64 // 0: no limit
	MemoryUsed  int64
	Models      int
	// LowPriority is true when low priority was asked for and the OS lowered every thread asked so far.
	LowPriority bool
}

// ErrMemoryLimit matches (errors.Is) the error a load or generation returns when it doesn't fit in the memory budget.
var ErrMemoryLimit = errors.New("engine: memory limit reached")

// MemoryError says what didn't fit in the memory budget. It matches ErrMemoryLimit.
type MemoryError struct {
	What  string // "loading the model", "a generation's state"
	Need  int64  // bytes it needed
	Used  int64  // bytes in use by everything else (after freeing what could be freed)
	Limit int64
}

func (e *MemoryError) Error() string {
	return fmt.Sprintf("engine: memory limit: %s needs %s, but %s of the %s budget is in use", e.What, fmtBytes(e.Need),
		fmtBytes(e.Used), fmtBytes(e.Limit))
}

// Is makes errors.Is(err, ErrMemoryLimit) true.
func (e *MemoryError) Is(target error) bool { return target == ErrMemoryLimit }

func fmtBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d bytes", n)
}

// NewResources starts the shared worker threads. Close it after the models using it are closed.
func NewResources(o ResourceOptions) *Resources {
	n := o.Threads
	if n <= 0 {
		n = min(runtime.GOMAXPROCS(0), 8)
	}
	r := &Resources{limit: o.Memory, lowPriority: o.LowPriority && LowPrioritySupported(), models: map[*Model]struct{}{}}
	r.pool = newPool(n, r.workerStart)
	r.threads = r.pool.n
	return r
}

// Close stops the worker threads once no model uses them (right away, or when the last model sharing them closes).
func (r *Resources) Close() {
	r.mu.Lock()
	r.closing = true
	done := len(r.models) == 0 && !r.closed
	if done {
		r.closed = true
	}
	r.mu.Unlock()
	if done {
		r.pool.close()
	}
}

// Usage reports the budget and what's in use.
func (r *Resources) Usage() ResourceUsage {
	r.mu.Lock()
	defer r.mu.Unlock()
	lim := max(r.limit, 0)
	return ResourceUsage{Threads: r.threads, MemoryLimit: lim, MemoryUsed: r.used, Models: len(r.models),
		LowPriority: r.lowPriority && r.lowerFailed.Load() == 0}
}

func (r *Resources) attach(m *Model) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closing {
		return errors.New("engine: the Resources are closed")
	}
	r.models[m] = struct{}{}
	return nil
}

func (r *Resources) detach(m *Model) {
	r.mu.Lock()
	delete(r.models, m)
	done := r.closing && len(r.models) == 0 && !r.closed
	if done {
		r.closed = true
	}
	r.mu.Unlock()
	if done {
		r.pool.close()
	}
}

// tryReserve takes n bytes from the budget if they fit.
func (r *Resources) tryReserve(n int64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.limit > 0 && r.used+n > r.limit {
		return false
	}
	r.used += n
	return true
}

// reserve takes n bytes, freeing idle states and cache entries (of any model sharing r) to make room. It fails with a
// *MemoryError when they still don't fit.
func (r *Resources) reserve(n int64, what string) error {
	if r.tryReserve(n) {
		return nil
	}
	for range 3 { // others may take what was freed in between; try a few times
		r.reclaim(n)
		if r.tryReserve(n) {
			return nil
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return &MemoryError{What: what, Need: n, Used: r.used, Limit: r.limit}
}

// reserveCache takes n bytes for a cache entry: like reserve, but it reports false instead of failing.
func (r *Resources) reserveCache(n int64) bool { return r.reserve(n, "") == nil }

func (r *Resources) release(n int64) {
	if n == 0 {
		return
	}
	r.mu.Lock()
	r.used -= n
	r.mu.Unlock()
}

func (r *Resources) snapshot() []*Model {
	r.mu.Lock()
	defer r.mu.Unlock()
	ms := make([]*Model, 0, len(r.models))
	for m := range r.models {
		ms = append(ms, m)
	}
	return ms
}

func (r *Resources) free() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.limit - r.used
}

// reclaim frees memory until need bytes are free or nothing more can go: first the idle states of every model, then
// prompt cache entries and threads in memory, least recently used first across all models. Pinned entries stay.
func (r *Resources) reclaim(need int64) {
	if r.limit <= 0 {
		return
	}
	ms := r.snapshot()
	for _, m := range ms {
		if r.free() >= need {
			return
		}
		r.release(m.dropIdleStates())
	}
	type victim struct {
		m     *Model
		used  time.Time
		evict func() int64
	}
	var vs []victim
	for _, m := range ms {
		m.prefix.mu.Lock()
		for _, e := range m.prefix.entries {
			if len(e.pins) == 0 {
				vs = append(vs, victim{m, e.used, func() int64 { return m.evictPrefix(e) }})
			}
		}
		m.prefix.mu.Unlock()
		m.threads.mu.Lock()
		for key, te := range m.threads.mem {
			vs = append(vs, victim{m, te.used, func() int64 { return m.evictThread(key, te) }})
		}
		m.threads.mu.Unlock()
	}
	sort.Slice(vs, func(i, j int) bool { return vs[i].used.Before(vs[j].used) })
	for _, v := range vs {
		if r.free() >= need {
			return
		}
		r.release(v.evict())
	}
}

// LowPrioritySupported reports whether LowPriority can lower thread priorities on this OS.
//
// Linux: each thread doing the work gets nice 10 (setpriority on its thread id), so the app's own threads, at nice 0,
// win the CPU when it's busy. Kernels with autogroup scheduling (most desktops) compare nice values only within a
// session's group, so it ranks nollama below the rest of its own process (the app embedding it), not below other
// programs.
//
// macOS: the threads get the utility QoS class (pthread_set_qos_class_self_np, called through libSystem without cgo),
// below the default class the app's threads run at. The OS may also favor the efficiency cores for them on Apple
// silicon. (The background class would be lower still, but it throttles a thread so hard that Go's runtime stalls
// behind it.)
//
// Windows: the threads get THREAD_PRIORITY_LOWEST.
//
// Other systems: not supported; LowPriority does nothing there (ResourceUsage.LowPriority says so).
func LowPrioritySupported() bool { return lowPrioritySupported }

// call runs fn on a thread of lowered priority when LowPriority is in effect, and on the caller's goroutine otherwise.
func (r *Resources) call(fn func()) {
	if !r.lowPriority {
		fn()
		return
	}
	done := make(chan struct{})
	var panicked any
	go func() {
		// Never unlocked: when this goroutine ends, Go ends its thread too, so a thread of lowered priority never
		// goes back to running the app's other goroutines (raising a priority again needs privileges on Linux).
		runtime.LockOSThread()
		r.lowerThisThread()
		defer close(done)
		defer func() { panicked = recover() }()
		fn()
	}()
	<-done
	if panicked != nil {
		panic(panicked)
	}
}

// workerStart runs at the start of each worker goroutine.
func (r *Resources) workerStart() {
	if r.lowPriority {
		runtime.LockOSThread() // never unlocked, as in call: the thread ends with the pool
		r.lowerThisThread()
	}
}

func (r *Resources) lowerThisThread() {
	if err := lowerThreadPriority(); err != nil {
		r.lowerFailed.Add(1)
		return
	}
	r.lowered.Add(1)
}
