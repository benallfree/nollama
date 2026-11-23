//go:build !darwin

// Package metal is the engine's GPU backend on macOS; elsewhere Open always fails.
package metal

import (
	"errors"
	"time"
)

type Size struct{ W, H, D uint }

type Device struct {
	Name       string
	Unified    bool
	MaxWorking uint64
	LastGPU    time.Duration
}

func Open() (*Device, error) { return nil, errors.New("metal: macOS only") }

func (d *Device) Close() {}

type Pipeline struct {
	Name       string
	MaxThreads int
	SIMDWidth  int
}

func (d *Device) Compile(src string, kernels ...string) (map[string]*Pipeline, error) {
	return nil, errors.New("metal: macOS only")
}

func (d *Device) CompilePrecise(src string, kernels ...string) (map[string]*Pipeline, error) {
	return nil, errors.New("metal: macOS only")
}

type Buffer struct{}

func (b *Buffer) Bytes() []byte { return nil }
func (b *Buffer) Release()      {}

func (d *Device) NewBuffer(n int) (*Buffer, error) { return nil, errors.New("metal: macOS only") }
func (d *Device) Wrap(mem []byte) (*Buffer, error) { return nil, errors.New("metal: macOS only") }

type Arg struct {
	Buf    *Buffer
	Offset int
	Bytes  []byte
}

type Dispatch struct {
	P       *Pipeline
	Args    []Arg
	Groups  Size
	Threads Size
	Shared  int
	Writes  uint32
}

func (d *Device) Run(ds ...Dispatch) error { return errors.New("metal: macOS only") }

// SetRunChunk does nothing without Metal.
func SetRunChunk(n int) int { return n }
