//go:build linux || windows

// Package cuda is a minimal binding to NVIDIA's CUDA driver API in pure Go (purego, no cgo): enough to load PTX
// modules, move memory and launch kernels on one stream. It needs only the NVIDIA driver (libcuda.so.1 or
// nvcuda.dll), not the CUDA toolkit: the driver compiles PTX for the GPU at load time.
package cuda

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type result int32

var (
	initOnce sync.Once
	initErr  error

	cuInit                    func(flags uint32) result
	cuDriverGetVersion        func(v *int32) result
	cuDeviceGetCount          func(n *int32) result
	cuDeviceGet               func(dev *int32, ordinal int32) result
	cuDeviceGetName           func(name *byte, n int32, dev int32) result
	cuDeviceTotalMem          func(bytes *uint64, dev int32) result
	cuDeviceGetAttribute      func(v *int32, attr int32, dev int32) result
	cuDevicePrimaryCtxRetain  func(ctx *uintptr, dev int32) result
	cuDevicePrimaryCtxRelease func(dev int32) result
	cuCtxSetCurrent           func(ctx uintptr) result
	cuModuleLoadData          func(mod *uintptr, image unsafe.Pointer) result
	cuModuleUnload            func(mod uintptr) result
	cuModuleGetFunction       func(fn *uintptr, mod uintptr, name string) result
	cuFuncSetAttribute        func(fn uintptr, attr int32, v int32) result
	cuMemAlloc                func(ptr *uint64, n uint64) result
	cuMemFree                 func(ptr uint64) result
	cuMemcpyHtoD              func(dst uint64, src unsafe.Pointer, n uint64) result
	cuMemcpyDtoH              func(dst unsafe.Pointer, src uint64, n uint64) result
	cuMemsetD8                func(dst uint64, v uint8, n uint64) result
	cuMemAllocHost            func(p *unsafe.Pointer, n uint64) result
	cuMemFreeHost             func(p unsafe.Pointer) result
	cuStreamCreate            func(s *uintptr, flags uint32) result
	cuStreamDestroy           func(s uintptr) result
	cuStreamSynchronize       func(s uintptr) result
	cuLaunchKernel            func(fn uintptr, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, params unsafe.Pointer, extra unsafe.Pointer) result
	cuGetErrorString          func(r result, s **byte) result
	cuStreamBeginCapture      func(s uintptr, mode int32) result
	cuStreamEndCapture        func(s uintptr, graph *uintptr) result
	cuGraphInstantiate        func(exec *uintptr, graph uintptr, flags uint64) result
	cuGraphLaunch             func(exec uintptr, s uintptr) result
	cuGraphExecDestroy        func(exec uintptr) result
	cuGraphDestroy            func(graph uintptr) result
)

const (
	attrComputeMajor            = 75
	attrComputeMinor            = 76
	funcAttrMaxDynamicSharedMem = 8
)

func libName() string {
	if runtime.GOOS == "windows" {
		return "nvcuda.dll"
	}
	return "libcuda.so.1"
}

func load() error {
	initOnce.Do(func() {
		if os.Getenv("NOLLAMA_NO_GPU") != "" {
			initErr = errors.New("cuda: disabled by NOLLAMA_NO_GPU")
			return
		}
		lib, err := openLibrary(libName())
		if err != nil {
			initErr = fmt.Errorf("cuda: no NVIDIA driver (%s): %w", libName(), err)
			return
		}
		for _, f := range []struct {
			ptr  any
			name string
		}{
			{&cuInit, "cuInit"}, {&cuDriverGetVersion, "cuDriverGetVersion"},
			{&cuDeviceGetCount, "cuDeviceGetCount"}, {&cuDeviceGet, "cuDeviceGet"},
			{&cuDeviceGetName, "cuDeviceGetName"}, {&cuDeviceTotalMem, "cuDeviceTotalMem_v2"},
			{&cuDeviceGetAttribute, "cuDeviceGetAttribute"},
			{&cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"}, {&cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
			{&cuCtxSetCurrent, "cuCtxSetCurrent"},
			{&cuModuleLoadData, "cuModuleLoadData"}, {&cuModuleUnload, "cuModuleUnload"},
			{&cuModuleGetFunction, "cuModuleGetFunction"}, {&cuFuncSetAttribute, "cuFuncSetAttribute"},
			{&cuMemAlloc, "cuMemAlloc_v2"}, {&cuMemFree, "cuMemFree_v2"},
			{&cuMemcpyHtoD, "cuMemcpyHtoD_v2"}, {&cuMemcpyDtoH, "cuMemcpyDtoH_v2"}, {&cuMemsetD8, "cuMemsetD8_v2"},
			{&cuMemAllocHost, "cuMemAllocHost_v2"}, {&cuMemFreeHost, "cuMemFreeHost"},
			{&cuStreamCreate, "cuStreamCreate"}, {&cuStreamDestroy, "cuStreamDestroy_v2"},
			{&cuStreamSynchronize, "cuStreamSynchronize"}, {&cuLaunchKernel, "cuLaunchKernel"},
			{&cuGetErrorString, "cuGetErrorString"},
			{&cuStreamBeginCapture, "cuStreamBeginCapture_v2"}, {&cuStreamEndCapture, "cuStreamEndCapture"},
			{&cuGraphInstantiate, "cuGraphInstantiateWithFlags"}, {&cuGraphLaunch, "cuGraphLaunch"},
			{&cuGraphExecDestroy, "cuGraphExecDestroy"}, {&cuGraphDestroy, "cuGraphDestroy"},
		} {
			sym, err := symbol(lib, f.name)
			if err != nil {
				initErr = fmt.Errorf("cuda: driver lacks %s: %w", f.name, err)
				return
			}
			purego.RegisterFunc(f.ptr, sym)
		}
		if r := cuInit(0); r != 0 {
			initErr = errorOf("cuInit", r)
		}
	})
	return initErr
}

func errorOf(what string, r result) error {
	if r == 0 {
		return nil
	}
	var s *byte
	msg := fmt.Sprintf("error %d", r)
	if cuGetErrorString != nil && cuGetErrorString(r, &s) == 0 && s != nil {
		msg = goString(s)
	}
	return fmt.Errorf("cuda: %s: %s", what, msg)
}

func goString(p *byte) string {
	var b []byte
	for ; *p != 0; p = (*byte)(unsafe.Add(unsafe.Pointer(p), 1)) {
		b = append(b, *p)
	}
	return string(b)
}

// Device is one GPU, its primary context and one stream. Calls lock the OS thread and make the
// context current on it, so a Device can be used from any goroutine (one call at a time).
type Device struct {
	mu         sync.Mutex
	dev        int32
	ctx        uintptr
	stream     uintptr
	Name       string
	Memory     uint64 // bytes
	Major      int    // compute capability
	Minor      int
	DriverCUDA int // highest CUDA version the driver supports, e.g. 12060
}

// do runs f with the device's context current on a locked OS thread.
func (d *Device) do(f func() error) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if r := cuCtxSetCurrent(d.ctx); r != 0 {
		return errorOf("cuCtxSetCurrent", r)
	}
	return f()
}

// Open returns GPU number ordinal, or an error when there's no NVIDIA driver or GPU.
func Open(ordinal int) (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	var n int32
	if r := cuDeviceGetCount(&n); r != 0 {
		return nil, errorOf("cuDeviceGetCount", r)
	}
	if int(n) <= ordinal {
		return nil, fmt.Errorf("cuda: no GPU %d (%d found)", ordinal, n)
	}
	d := &Device{}
	if r := cuDeviceGet(&d.dev, int32(ordinal)); r != 0 {
		return nil, errorOf("cuDeviceGet", r)
	}
	name := make([]byte, 256)
	cuDeviceGetName(&name[0], int32(len(name)), d.dev)
	d.Name = goString(&name[0])
	cuDeviceTotalMem(&d.Memory, d.dev)
	var maj, min, ver int32
	cuDeviceGetAttribute(&maj, attrComputeMajor, d.dev)
	cuDeviceGetAttribute(&min, attrComputeMinor, d.dev)
	cuDriverGetVersion(&ver)
	d.Major, d.Minor, d.DriverCUDA = int(maj), int(min), int(ver)
	if r := cuDevicePrimaryCtxRetain(&d.ctx, d.dev); r != 0 {
		return nil, errorOf("cuDevicePrimaryCtxRetain", r)
	}
	err := d.do(func() error { return errorOf("cuStreamCreate", cuStreamCreate(&d.stream, 1 /* non-blocking */)) })
	if err != nil {
		cuDevicePrimaryCtxRelease(d.dev)
		return nil, err
	}
	return d, nil
}

// Close releases the stream and the context.
func (d *Device) Close() {
	d.do(func() error { cuStreamDestroy(d.stream); return nil })
	cuDevicePrimaryCtxRelease(d.dev)
}

// Ptr is device memory.
type Ptr uint64

// Alloc allocates n bytes of device memory, zeroed.
func (d *Device) Alloc(n int) (Ptr, error) {
	var p uint64
	err := d.do(func() error {
		if r := cuMemAlloc(&p, uint64(n)); r != 0 {
			return errorOf(fmt.Sprintf("cuMemAlloc(%d)", n), r)
		}
		if r := cuMemsetD8(p, 0, uint64(n)); r != 0 {
			return errorOf("cuMemsetD8", r)
		}
		return errorOf("cuStreamSynchronize", cuStreamSynchronize(0)) // see Write
	})
	return Ptr(p), err
}

// Free frees device memory.
func (d *Device) Free(p Ptr) {
	d.do(func() error { cuMemFree(uint64(p)); return nil })
}

// Write copies src to device memory at dst (after the stream's queued work).
// cuMemsetD8 and cuMemcpyHtoD from pageable memory run on the legacy default stream and can return before
// the device has finished them, and d.stream is non-blocking, so it doesn't wait for that stream. Without
// synchronizing stream 0 after them, a kernel launched next could run first, and the memset or copy would then
// overwrite its output (seen on an A16: a freshly zeroed output buffer came back with zeros after the kernel ran).
func (d *Device) Write(dst Ptr, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	return d.do(func() error {
		if r := cuStreamSynchronize(d.stream); r != 0 {
			return errorOf("cuStreamSynchronize", r)
		}
		if r := cuMemcpyHtoD(uint64(dst), unsafe.Pointer(unsafe.SliceData(src)), uint64(len(src))); r != 0 {
			return errorOf("cuMemcpyHtoD", r)
		}
		return errorOf("cuStreamSynchronize", cuStreamSynchronize(0))
	})
}

// Read copies device memory at src into dst (after the stream's queued work).
func (d *Device) Read(src Ptr, dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	return d.do(func() error {
		if r := cuStreamSynchronize(d.stream); r != 0 {
			return errorOf("cuStreamSynchronize", r)
		}
		return errorOf("cuMemcpyDtoH", cuMemcpyDtoH(unsafe.Pointer(unsafe.SliceData(dst)), uint64(src), uint64(len(dst))))
	})
}

// HostAlloc allocates n bytes of page-locked host memory: copies between it and the device run at full DMA speed,
// where pageable memory goes through the driver's own staging.
func (d *Device) HostAlloc(n int) ([]byte, error) {
	var p unsafe.Pointer
	err := d.do(func() error { return errorOf(fmt.Sprintf("cuMemAllocHost(%d)", n), cuMemAllocHost(&p, uint64(n))) })
	if err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(p), n), nil
}

// HostFree frees memory from HostAlloc.
func (d *Device) HostFree(b []byte) {
	if len(b) > 0 {
		d.do(func() error { cuMemFreeHost(unsafe.Pointer(unsafe.SliceData(b))); return nil })
	}
}

// Module is loaded PTX.
type Module struct {
	d   *Device
	mod uintptr
}

// Kernel is a function in a module.
type Kernel struct {
	fn   uintptr
	Name string
}

// Load JIT-compiles PTX for this GPU and returns the named kernels.
func (d *Device) Load(ptx string, names ...string) (*Module, map[string]*Kernel, error) {
	m := &Module{d: d}
	ks := map[string]*Kernel{}
	src := []byte(ptx + "\x00")
	err := d.do(func() error {
		if r := cuModuleLoadData(&m.mod, unsafe.Pointer(&src[0])); r != 0 {
			return errorOf("cuModuleLoadData", r)
		}
		for _, n := range names {
			var fn uintptr
			if r := cuModuleGetFunction(&fn, m.mod, n+"\x00"); r != 0 {
				return errorOf("cuModuleGetFunction "+n, r)
			}
			ks[n] = &Kernel{fn: fn, Name: n}
		}
		return nil
	})
	runtime.KeepAlive(src)
	if err != nil {
		if m.mod != 0 {
			d.do(func() error { cuModuleUnload(m.mod); return nil })
		}
		return nil, nil, err
	}
	return m, ks, nil
}

// Unload frees the module.
func (m *Module) Unload() { m.d.do(func() error { cuModuleUnload(m.mod); return nil }) }

// SetMaxShared lets k use up to n bytes of dynamic shared memory (above the default 48 KB).
func (d *Device) SetMaxShared(k *Kernel, n int) error {
	return d.do(func() error {
		return errorOf("cuFuncSetAttribute", cuFuncSetAttribute(k.fn, funcAttrMaxDynamicSharedMem, int32(n)))
	})
}

// Arg is a kernel argument: a device pointer, or a value's bytes (a scalar or a struct passed by value).
type Arg struct {
	Ptr   Ptr
	Bytes []byte
}

// Launch is one kernel launch.
type Launch struct {
	K       *Kernel
	Args    []Arg
	Grid    [3]uint32
	Block   [3]uint32
	SharedB int // dynamic shared memory bytes
}

// Run queues the launches in order on the device's stream and waits for them.
func (d *Device) Run(ls ...Launch) error {
	return d.do(func() error {
		if err := d.enqueue(ls); err != nil {
			return err
		}
		return errorOf("cuStreamSynchronize", cuStreamSynchronize(d.stream))
	})
}

// Graph is a recorded list of launches (a CUDA graph): replaying it costs one launch, not one per kernel. Kernel
// arguments are fixed when it is recorded; what the kernels read from device memory may change between replays.
type Graph struct {
	d           *Device
	graph, exec uintptr
}

// Record captures the launches as a graph (nothing runs).
func (d *Device) Record(ls ...Launch) (*Graph, error) {
	g := &Graph{d: d}
	err := d.do(func() error {
		if r := cuStreamBeginCapture(d.stream, 2 /* relaxed */); r != 0 {
			return errorOf("cuStreamBeginCapture", r)
		}
		err := d.enqueue(ls)
		if r := cuStreamEndCapture(d.stream, &g.graph); r != 0 && err == nil {
			err = errorOf("cuStreamEndCapture", r)
		}
		if err != nil {
			return err
		}
		return errorOf("cuGraphInstantiate", cuGraphInstantiate(&g.exec, g.graph, 0))
	})
	if err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// Replay runs the graph and waits for it.
func (g *Graph) Replay() error {
	return g.d.do(func() error {
		if r := cuGraphLaunch(g.exec, g.d.stream); r != 0 {
			return errorOf("cuGraphLaunch", r)
		}
		return errorOf("cuStreamSynchronize", cuStreamSynchronize(g.d.stream))
	})
}

// Destroy frees the graph.
func (g *Graph) Destroy() {
	g.d.do(func() error {
		if g.exec != 0 {
			cuGraphExecDestroy(g.exec)
		}
		if g.graph != 0 {
			cuGraphDestroy(g.graph)
		}
		g.exec, g.graph = 0, 0
		return nil
	})
}

// enqueue queues the launches on the stream (the caller holds the context).
func (d *Device) enqueue(ls []Launch) error {
	var pin runtime.Pinner
	defer pin.Unpin()
	for _, l := range ls {
		// kernelParams: an array of pointers, each to one argument's value.
		vals := make([]uint64, len(l.Args))
		params := make([]unsafe.Pointer, len(l.Args))
		pin.Pin(&vals[0])
		for i, a := range l.Args {
			if a.Bytes != nil {
				pin.Pin(&a.Bytes[0])
				params[i] = unsafe.Pointer(&a.Bytes[0])
			} else {
				vals[i] = uint64(a.Ptr)
				params[i] = unsafe.Pointer(&vals[i])
			}
		}
		pin.Pin(&params[0])
		r := cuLaunchKernel(l.K.fn, l.Grid[0], l.Grid[1], l.Grid[2], l.Block[0], l.Block[1], l.Block[2],
			uint32(l.SharedB), d.stream, unsafe.Pointer(&params[0]), nil)
		if r != 0 {
			return errorOf("cuLaunchKernel "+l.K.Name, r)
		}
	}
	return nil
}
