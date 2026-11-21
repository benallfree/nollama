package engine

import (
	"syscall"
	"testing"
)

// threadLowered reports whether the calling thread's nice value is above 0. The caller locks its goroutine to the
// thread.
func threadLowered(t *testing.T) bool {
	t.Helper()
	p, err := syscall.Getpriority(syscall.PRIO_PROCESS, syscall.Gettid())
	if err != nil {
		t.Fatal(err)
	}
	return 20-p > 0 // the raw syscall returns 20 - nice
}
