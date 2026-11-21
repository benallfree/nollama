//go:build !(darwin || linux || freebsd)

package engine

import "os"

// Fallback where nollama doesn't mmap: read the range into memory.
func mmapRange(f *os.File, off, size int64) (mapping, data []byte, err error) {
	b, err := readRange(f, off, size)
	return nil, b, err
}

func munmap(b []byte) error { return nil }
