// Package metal is a minimal Metal binding in pure Go (purego, no cgo): enough to compile shaders, wrap memory
// in buffers and dispatch compute kernels. It is the engine's GPU backend on macOS.
package metal

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// Size is MTLSize.
type Size struct{ W, H, D uint }

var (
	initOnce sync.Once
	initErr  error

	createSystemDefaultDevice func() objc.ID
	autoreleasePoolPush       func() uintptr
	autoreleasePoolPop        func(uintptr)

	selNewCommandQueue      = objc.RegisterName("newCommandQueue")
	selNewLibrary           = objc.RegisterName("newLibraryWithSource:options:error:")
	selNewFunction          = objc.RegisterName("newFunctionWithName:")
	selNewPipeline          = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selNewBufferNoCopy      = objc.RegisterName("newBufferWithBytesNoCopy:length:options:deallocator:")
	selNewBufferLength      = objc.RegisterName("newBufferWithLength:options:")
	selContents             = objc.RegisterName("contents")
	selLength               = objc.RegisterName("length")
	selCommandBuffer        = objc.RegisterName("commandBuffer")
	selComputeEncoder       = objc.RegisterName("computeCommandEncoder")
	selSetPipeline          = objc.RegisterName("setComputePipelineState:")
	selSetBuffer            = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBytes             = objc.RegisterName("setBytes:length:atIndex:")
	selSetThreadgroupMemory = objc.RegisterName("setThreadgroupMemoryLength:atIndex:")
	selDispatch             = objc.RegisterName("dispatchThreadgroups:threadsPerThreadgroup:")
	selEndEncoding          = objc.RegisterName("endEncoding")
	selCommit               = objc.RegisterName("commit")
	selWaitUntilCompleted   = objc.RegisterName("waitUntilCompleted")
	selError                = objc.RegisterName("error")
	selName                 = objc.RegisterName("name")
	selRelease              = objc.RegisterName("release")
	selStringWithUTF8       = objc.RegisterName("stringWithUTF8String:")
	selUTF8String           = objc.RegisterName("UTF8String")
	selLocalizedDescription = objc.RegisterName("localizedDescription")
	selMaxThreads           = objc.RegisterName("maxTotalThreadsPerThreadgroup")
	selExecWidth            = objc.RegisterName("threadExecutionWidth")
	selHasUnifiedMemory     = objc.RegisterName("hasUnifiedMemory")
	selRecommendedMaxWS     = objc.RegisterName("recommendedMaxWorkingSetSize")
	selAlloc                = objc.RegisterName("alloc")
	selInit                 = objc.RegisterName("init")
	selSetFastMath          = objc.RegisterName("setFastMathEnabled:")

	classNSString objc.Class // after Foundation loads
)

const (
	resourceStorageModeShared = 0 << 4
	resourceHazardTrackingDef = 0
)

func load() error {
	initOnce.Do(func() {
		if os.Getenv("NOLLAMA_NO_GPU") != "" {
			initErr = errors.New("metal: disabled by NOLLAMA_NO_GPU")
			return
		}
		mtl, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			initErr = fmt.Errorf("metal: %w", err)
			return
		}
		// Metal's device discovery needs CoreGraphics loaded, or MTLCreateSystemDefaultDevice returns nil.
		if _, err := purego.Dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
			initErr = fmt.Errorf("metal: %w", err)
			return
		}
		if _, err := purego.Dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
			initErr = fmt.Errorf("metal: %w", err)
			return
		}
		classNSString = objc.GetClass("NSString")
		lobjc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			initErr = fmt.Errorf("metal: %w", err)
			return
		}
		purego.RegisterLibFunc(&createSystemDefaultDevice, mtl, "MTLCreateSystemDefaultDevice")
		purego.RegisterLibFunc(&autoreleasePoolPush, lobjc, "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&autoreleasePoolPop, lobjc, "objc_autoreleasePoolPop")
	})
	return initErr
}

// withPool runs f inside an autorelease pool on a locked OS thread, so autoreleased objects
// (command buffers, encoders, strings) are freed when f returns.
func withPool(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p := autoreleasePoolPush()
	defer autoreleasePoolPop(p)
	f()
}

func nsString(s string) objc.ID {
	return objc.ID(classNSString).Send(selStringWithUTF8, s+"\x00")
}

func goString(ns objc.ID) string {
	if ns == 0 {
		return ""
	}
	return objc.Send[string](ns, selUTF8String)
}

func nsError(e objc.ID) error {
	if e == 0 {
		return nil
	}
	return errors.New(goString(e.Send(selLocalizedDescription)))
}

// Device is a GPU and one command queue.
type Device struct {
	dev, queue objc.ID
	Name       string
	Unified    bool
	MaxWorking uint64 // recommended max working set, bytes
}

// Open returns the system default GPU, or an error when there is none (or Metal can't load).
func Open() (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	var d *Device
	var err error
	withPool(func() {
		dev := createSystemDefaultDevice()
		if dev == 0 {
			err = errors.New("metal: no GPU")
			return
		}
		d = &Device{
			dev:        dev,
			queue:      dev.Send(selNewCommandQueue),
			Name:       goString(dev.Send(selName)),
			Unified:    objc.Send[bool](dev, selHasUnifiedMemory),
			MaxWorking: objc.Send[uint64](dev, selRecommendedMaxWS),
		}
	})
	return d, err
}

// Close releases the device.
func (d *Device) Close() {
	d.queue.Send(selRelease)
	d.dev.Send(selRelease)
}

// Pipeline is a compiled compute kernel.
type Pipeline struct {
	ps         objc.ID
	Name       string
	MaxThreads int // per threadgroup
	SIMDWidth  int
}

// Compile compiles Metal Shading Language source and returns a pipeline for each named kernel.
func (d *Device) Compile(src string, kernels ...string) (map[string]*Pipeline, error) {
	return d.compile(src, false, kernels)
}

// CompilePrecise is Compile with fast math off: IEEE division and square roots, no reassociation, and no fusing of
// a multiply and an add written as separate operations into one multiply-add.
func (d *Device) CompilePrecise(src string, kernels ...string) (map[string]*Pipeline, error) {
	return d.compile(src, true, kernels)
}

func (d *Device) compile(src string, precise bool, kernels []string) (map[string]*Pipeline, error) {
	out := map[string]*Pipeline{}
	var err error
	withPool(func() {
		opts := objc.ID(0)
		if precise {
			opts = objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
			opts.Send(selSetFastMath, false)
			defer opts.Send(selRelease)
		}
		var e objc.ID
		lib := d.dev.Send(selNewLibrary, nsString(src), opts, unsafe.Pointer(&e))
		if lib == 0 {
			err = fmt.Errorf("metal: compile: %w", nsError(e))
			return
		}
		defer lib.Send(selRelease)
		for _, k := range kernels {
			fn := lib.Send(selNewFunction, nsString(k))
			if fn == 0 {
				err = fmt.Errorf("metal: no kernel %q", k)
				return
			}
			var e objc.ID
			ps := d.dev.Send(selNewPipeline, fn, unsafe.Pointer(&e))
			fn.Send(selRelease)
			if ps == 0 {
				err = fmt.Errorf("metal: pipeline %s: %w", k, nsError(e))
				return
			}
			out[k] = &Pipeline{
				ps: ps, Name: k,
				MaxThreads: int(objc.Send[uint](ps, selMaxThreads)),
				SIMDWidth:  int(objc.Send[uint](ps, selExecWidth)),
			}
		}
	})
	return out, err
}

// Buffer is GPU-visible memory. On unified-memory Macs the CPU sees the same bytes (Bytes).
type Buffer struct {
	buf   objc.ID
	bytes []byte
}

// Bytes returns the buffer's memory as seen by the CPU.
func (b *Buffer) Bytes() []byte { return b.bytes }

// Release frees the buffer (the memory itself, for Wrap, stays the caller's).
func (b *Buffer) Release() {
	if b.buf != 0 {
		b.buf.Send(selRelease)
		b.buf = 0
	}
}

// NewBuffer allocates a shared buffer of n bytes.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	var b *Buffer
	withPool(func() {
		buf := d.dev.Send(selNewBufferLength, uint(n), uint(resourceStorageModeShared))
		if buf == 0 {
			return
		}
		p := objc.Send[unsafe.Pointer](buf, selContents)
		b = &Buffer{buf: buf, bytes: unsafe.Slice((*byte)(p), n)}
	})
	if b == nil {
		return nil, fmt.Errorf("metal: can't allocate %d bytes", n)
	}
	return b, nil
}

// Wrap makes a buffer over existing memory without copying it (an mmapped model file, say).
// mem must start on a page boundary; its length is rounded up to a whole page, so the
// mapping must extend that far (mmap maps whole pages). The caller keeps mem alive and
// unchanged for the buffer's lifetime.
func (d *Device) Wrap(mem []byte) (*Buffer, error) {
	page := os.Getpagesize()
	if uintptr(unsafe.Pointer(unsafe.SliceData(mem)))%uintptr(page) != 0 {
		return nil, errors.New("metal: Wrap needs page-aligned memory")
	}
	n := (len(mem) + page - 1) &^ (page - 1)
	var b *Buffer
	withPool(func() {
		buf := d.dev.Send(selNewBufferNoCopy, unsafe.Pointer(unsafe.SliceData(mem)), uint(n), uint(resourceStorageModeShared), objc.ID(0))
		if buf != 0 {
			b = &Buffer{buf: buf, bytes: mem}
		}
	})
	if b == nil {
		return nil, fmt.Errorf("metal: can't wrap %d bytes", n)
	}
	return b, nil
}

// Arg is one kernel argument: a buffer range or inline bytes (constants up to 4 KB).
type Arg struct {
	Buf    *Buffer
	Offset int
	Bytes  []byte
}

// Dispatch is one kernel launch.
type Dispatch struct {
	P       *Pipeline
	Args    []Arg
	Groups  Size // threadgroups in the grid
	Threads Size // threads per threadgroup
	Shared  int  // threadgroup memory bytes at index 0
}

// Run encodes the dispatches in order into one command buffer, commits it and waits.
func (d *Device) Run(ds ...Dispatch) error {
	var err error
	withPool(func() {
		// Commit a command buffer every runChunk dispatches, so the GPU starts on the first ones while the rest are
		// encoded (a decode step is hundreds of small dispatches, and encoding them takes milliseconds). The queue
		// runs its buffers in order; waiting on the last one waits for all.
		var cbs []objc.ID
		for start := 0; start < len(ds) || start == 0; start += runChunk {
			cb := d.queue.Send(selCommandBuffer)
			enc := cb.Send(selComputeEncoder)
			for _, x := range ds[start:min(start+runChunk, len(ds))] {
				enc.Send(selSetPipeline, x.P.ps)
				for i, a := range x.Args {
					if a.Buf != nil {
						enc.Send(selSetBuffer, a.Buf.buf, uint(a.Offset), uint(i))
					} else {
						enc.Send(selSetBytes, unsafe.Pointer(unsafe.SliceData(a.Bytes)), uint(len(a.Bytes)), uint(i))
					}
				}
				if x.Shared > 0 {
					enc.Send(selSetThreadgroupMemory, uint(x.Shared), uint(0))
				}
				enc.Send(selDispatch, x.Groups, x.Threads)
			}
			enc.Send(selEndEncoding)
			cb.Send(selCommit)
			cbs = append(cbs, cb)
			if len(ds) == 0 {
				break
			}
		}
		cbs[len(cbs)-1].Send(selWaitUntilCompleted)
		for _, cb := range cbs {
			if err == nil {
				err = nsError(cb.Send(selError))
			}
		}
		runtime.KeepAlive(ds)
	})
	return err
}

// runChunk is how many dispatches Run puts in one command buffer.
const runChunk = 48
