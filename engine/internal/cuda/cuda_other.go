//go:build !linux && !windows

// Package cuda is the engine's NVIDIA backend on Linux and Windows; elsewhere Open always fails.
package cuda

import "errors"

type Device struct {
	Name         string
	Memory       uint64
	Major, Minor int
	DriverCUDA   int
}

var errNo = errors.New("cuda: Linux and Windows only")

func Open(ordinal int) (*Device, error) { return nil, errNo }
func (d *Device) Close()                {}

type Ptr uint64

func (d *Device) Alloc(n int) (Ptr, error)        { return 0, errNo }
func (d *Device) Free(p Ptr)                      {}
func (d *Device) Write(dst Ptr, src []byte) error { return errNo }
func (d *Device) Read(src Ptr, dst []byte) error  { return errNo }
func (d *Device) HostAlloc(n int) ([]byte, error) { return nil, errNo }
func (d *Device) HostFree(b []byte)               {}

type Module struct{}
type Kernel struct{ Name string }

func (d *Device) Load(ptx string, names ...string) (*Module, map[string]*Kernel, error) {
	return nil, nil, errNo
}
func (m *Module) Unload()                             {}
func (d *Device) SetMaxShared(k *Kernel, n int) error { return errNo }

type Arg struct {
	Ptr   Ptr
	Bytes []byte
}

type Launch struct {
	K       *Kernel
	Args    []Arg
	Grid    [3]uint32
	Block   [3]uint32
	SharedB int
}

func (d *Device) Run(ls ...Launch) error { return errNo }

type Graph struct{}

func (d *Device) Record(ls ...Launch) (*Graph, error) { return nil, errNo }
func (g *Graph) Replay() error                        { return errNo }
func (g *Graph) Destroy()                             {}

func (d *Device) Profile(on bool) {}

func (d *Device) Queue(ls ...Launch) error             { return errNo }
func (g *Graph) Launch() error                         { return errNo }
func (d *Device) WriteAsync(dst Ptr, src []byte) error { return errNo }
func (d *Device) ReadAsync(src Ptr, dst []byte) error  { return errNo }

type Event struct{}

func (d *Device) NewEvent() (*Event, error) { return nil, errNo }
func (e *Event) Record() error              { return errNo }
func (e *Event) Wait() error                { return errNo }
func (e *Event) Destroy()                   {}
