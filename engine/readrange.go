package engine

import (
	"io"
	"os"
)

// readRange reads size bytes of f starting at off: the fallback for systems where the engine doesn't mmap.
func readRange(f *os.File, off, size int64) ([]byte, error) {
	b := make([]byte, size)
	_, err := io.ReadFull(io.NewSectionReader(f, off, size), b)
	return b, err
}
