package engine

// The integer models' CUDA backend: gpu_int.cu (generated from gpu_int.metal) compiled to PTX, run through the driver
// API (no toolkit needed). The model file is uploaded once; nothing is shared with the CPU, so tables and counts are
// copied back.

import (
	_ "embed"
	"fmt"
	"os"
	"time"

	"github.com/benallfree/nollama/engine/internal/cuda"
)

// gpu_int_cuda.ptx is gpu_int.cu compiled for compute capability 7.5 (T4 and newer), by gen_ptx.sh.
//
//go:embed gpu_int_cuda.ptx
var cudaIntPTX string

// gpu_int_mma.ptx is gpu_int_mma.cu for compute 8.0+: kernels on int8 tensor cores that stand in for gpu_int.cu's
// (same arguments, grid and results), loaded where they work unless NOLLAMA_INT_NO_MMA is set.
//
//go:embed gpu_int_mma.ptx
var cudaIntMMAPTX string

// With the byte-plane KV cache: scores (maxima only) and weights x V (recomputing scores) on tensor cores.
var cudaIntMMA = map[string]string{"int_bscores64": "int_bmax_mma64", "int_bscores128": "int_bmax_mma128",
	"int_bpv64": "int_bpv2_mma64", "int_bpv128": "int_bpv2_mma128", "int_kvplanes": "int_kvplanes",
	"int_gemm": "int_gemm_mma"}

type cudaIntDev struct {
	rec  map[replayKey]*cuda.Graph // replay's recordings
	dev  *cuda.Device
	mod  *cuda.Module
	mma  *cuda.Module // or nil
	ks   map[string]*cuda.Kernel
	ls   []cuda.Launch // run's scratch
	last time.Duration
}

type cudaIntBuf struct {
	p cuda.Ptr
	n int
}

func (b *cudaIntBuf) size() int { return b.n }

func openIntCUDA(m *IntModel) (intDev, ibuf, error) {
	dev, err := cuda.Open(0)
	if err != nil {
		return nil, nil, err
	}
	if dev.Major < 7 || dev.Major == 7 && dev.Minor < 5 {
		dev.Close()
		return nil, nil, fmt.Errorf("cuda: %s is compute %d.%d; integer models need 7.5 or newer", dev.Name, dev.Major, dev.Minor)
	}
	mod, ks, err := dev.Load(cudaIntPTX, intKernels...)
	if err != nil {
		dev.Close()
		return nil, nil, err
	}
	d := &cudaIntDev{dev: dev, mod: mod, ks: ks, rec: map[replayKey]*cuda.Graph{}}
	if dev.Major >= 8 && os.Getenv("NOLLAMA_INT_NO_MMA") == "" {
		names := []string{"int_dattn1_64", "int_dattn1_128", "int_dattn2_64", "int_dattn2_128", "int_matvec8"} // used by name
		for _, n := range cudaIntMMA {
			names = append(names, n)
		}
		if mm, mks, err := dev.Load(cudaIntMMAPTX, names...); err == nil {
			for _, n := range names {
				ks[n] = mks[n]
			}
			d.mma = mm
			for k, n := range cudaIntMMA {
				ks[k] = mks[n] // the tensor-core kernel runs under the portable one's name
			}
		}
	}
	f, err := d.alloc(len(m.g.data))
	if err != nil {
		d.close()
		return nil, nil, err
	}
	d.write(f, 0, m.g.data)
	return d, f, nil
}

func (d *cudaIntDev) name() string {
	tc := ""
	if d.mma != nil {
		tc = ", tensor cores"
	}
	return fmt.Sprintf("cuda (%s, compute %d.%d%s)", d.dev.Name, d.dev.Major, d.dev.Minor, tc)
}

func (d *cudaIntDev) close() {
	if d.mma != nil {
		d.mma.Unload()
	}
	d.mod.Unload()
	d.dev.Close()
}

// alloc returns zeroed device memory (as Metal's buffers are).
func (d *cudaIntDev) alloc(n int) (ibuf, error) {
	p, err := d.dev.Alloc(n)
	if err != nil {
		return nil, err
	}
	b := &cudaIntBuf{p, n}
	zero := make([]byte, min(n, 64<<20))
	for off := 0; off < n; off += len(zero) {
		d.write(b, off, zero[:min(len(zero), n-off)])
	}
	return b, nil
}

func (d *cudaIntDev) free(b ibuf) { d.dev.Free(b.(*cudaIntBuf).p) }

func (d *cudaIntDev) write(dst ibuf, off int, src []byte) {
	if err := d.dev.Write(dst.(*cudaIntBuf).p+cuda.Ptr(off), src); err != nil {
		panic(err) // a failed copy to the device is a lost device; nothing sensible can follow
	}
}

func (d *cudaIntDev) read(src ibuf, off int, dst []byte) {
	if err := d.dev.Read(src.(*cudaIntBuf).p+cuda.Ptr(off), dst); err != nil {
		panic(err)
	}
}

func (d *cudaIntDev) host(ibuf) []byte       { return nil }
func (d *cudaIntDev) lastGPU() time.Duration { return d.last }

func (d *cudaIntDev) arg(a iarg) cuda.Arg {
	if a.Buf == nil {
		return cuda.Arg{Bytes: a.Bytes}
	}
	return cuda.Arg{Ptr: a.Buf.(*cudaIntBuf).p + cuda.Ptr(a.Offset)}
}

// prepare converts the arguments once; constants stay kernel parameters.
func (d *cudaIntDev) prepare(ds []idisp, _ func(int) (ibuf, error)) error {
	for i := range ds {
		pre := make([]cuda.Arg, len(ds[i].Args))
		for j, a := range ds[i].Args {
			pre[j] = d.arg(a)
		}
		ds[i].pre = pre
	}
	return nil
}

func (d *cudaIntDev) graphs() bool { return true }
func (d *cudaIntDev) planes() bool { return d.mma != nil }

// replay records ds as a CUDA graph the first time (per key) and replays it: one launch instead of one per kernel.
func (d *cudaIntDev) replay(key replayKey, ds []idisp) error {
	g := d.rec[key]
	if g == nil {
		if err := d.launches(ds); err != nil {
			return err
		}
		var err error
		if g, err = d.dev.Record(d.ls...); err != nil {
			return err
		}
		d.rec[key] = g
	}
	t0 := time.Now()
	err := g.Replay()
	d.last = time.Since(t0)
	return err
}

func (d *cudaIntDev) forget(owner any) {
	for k, g := range d.rec {
		if k.owner == owner {
			g.Destroy()
			delete(d.rec, k)
		}
	}
}

func (d *cudaIntDev) run(ds []idisp) error {
	if err := d.launches(ds); err != nil {
		return err
	}
	t0 := time.Now()
	err := d.dev.Run(d.ls...)
	d.last = time.Since(t0)
	return err
}

// launches converts ds into d.ls.
func (d *cudaIntDev) launches(ds []idisp) error {
	d.ls = d.ls[:0]
	for _, x := range ds {
		args, ok := x.pre.([]cuda.Arg)
		if !ok {
			args = make([]cuda.Arg, len(x.Args))
			for j, a := range x.Args {
				args[j] = d.arg(a)
			}
		}
		k := d.ks[x.K]
		if k == nil {
			return fmt.Errorf("cuda: no kernel %s", x.K)
		}
		d.ls = append(d.ls, cuda.Launch{K: k, Args: args, Grid: [3]uint32{uint32(x.Groups.W), uint32(x.Groups.H), uint32(x.Groups.D)},
			Block: [3]uint32{uint32(x.Threads.W), uint32(x.Threads.H), uint32(x.Threads.D)}})
	}
	return nil
}
