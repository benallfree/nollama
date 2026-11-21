//go:build darwin || linux || freebsd

package engine

import (
	"os"
	"syscall"
)

// mmapRange maps size bytes of f starting at off, read-only. mapping is what munmap takes; data is the requested
// range inside it (mmap wants a page-aligned offset, so the mapping can start a little earlier).
func mmapRange(f *os.File, off, size int64) (mapping, data []byte, err error) {
	page := int64(os.Getpagesize())
	start := off &^ (page - 1)
	mapping, err = syscall.Mmap(int(f.Fd()), start, int(off-start+size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return mapping, mapping[off-start:], nil
}

func munmap(b []byte) error { return syscall.Munmap(b) }
