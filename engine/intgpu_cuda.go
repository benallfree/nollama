package engine

// The integer models' CUDA backend: gpu_int.cu (generated from gpu_int.metal) compiled to PTX, run through the driver
// API (no toolkit needed). The model file is uploaded once; nothing is shared with the CPU, so tables and counts are
// copied back.

import (
	_ "embed"
	"fmt"
	"os"
	"strings"
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
// cudaIntDyn: dynamic shared memory per tensor-core kernel (EXPA's nonzero head as ushorts, ETN entries).
// int_gemm_mma3: GEMM3_SMEM, 3 stages of 64 x 160 + 128 x 80 + 256 bytes.
var cudaIntDyn = map[string]int{"int_bpv2_mma64": 2 * 11392, "int_bpv2_mma128": 2 * 11392, "int_gemm_mma3": 3 * (64*160 + 128*80 + 256),
	"int_gemm_mma3_4": 4 * (64*160 + 128*80 + 256), "int_bpv2_mma64_lb1": 2 * 11392, "int_bpv2_mma128_lb1": 2 * 11392,
	"int_bpv2_mma64_np": 2 * 11392, "int_bpv2_mma128_np": 2 * 11392, "int_bpv2_mma64_cnt": 2 * 11392, "int_bpv2_mma128_cnt": 2 * 11392, "int_bpv2_mma64_lb4": 2 * 11392, "int_bpv2_mma128_lb4": 2 * 11392}

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
	pipe *cudaPipe     // pipelined decoding's buffers, made on first use
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
		names := []string{"int_dattn1_64", "int_dattn1_128", "int_dattn2_64", "int_dattn2_128", "int_matvec8",
			"int_dattn2a_64", "int_dattn2a_128", "int_mv2", "int_mvlog", "int_qkv2", "int_norm2", "int_mlp2", "int_mergea", "int_norm2s", "int_battn64", "int_battn128", "int_gemm_mma2", "int_gemm_mma3", "int_find2"} // used by name
		alias := map[string]string{}
		for k, n := range cudaIntMMA {
			alias[k] = n
		}
		// NOLLAMA_INT_ALIAS="int_bpv64=int_bpv2_mma64_lb3,...": run another tensor-core kernel under a name (experiments)
		for _, kv := range strings.Split(os.Getenv("NOLLAMA_INT_ALIAS"), ",") {
			if k, n, ok := strings.Cut(kv, "="); ok {
				alias[k] = n
			}
		}
		for _, n := range alias {
			names = append(names, n)
		}
		mm, mks, err := dev.Load(cudaIntMMAPTX, names...)
		if err != nil { // a kernel missing from the PTX (a build mistake): loud, not a silent fallback
			d.close()
			return nil, nil, fmt.Errorf("cuda: tensor-core kernels: %w (NOLLAMA_INT_NO_MMA=1 runs without them)", err)
		}
		{
			for _, n := range names {
				ks[n] = mks[n]
			}
			d.mma = mm
			for n, b := range cudaIntDyn {
				if mks[n] == nil {
					continue // a variant not loaded
				}
				if err := dev.SetMaxShared(mks[n], b); err != nil {
					d.close()
					return nil, nil, err
				}
			}
			for k, n := range alias {
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
	if d.pipe != nil {
		d.pipe.ev.Destroy()
		d.dev.HostFree(d.pipe.h)
	}
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
	return &cudaIntBuf{p, n}, nil // zeroed on the device by Alloc
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

// launch queues the recorded graph for key (recording it first if new) without waiting.
func (d *cudaIntDev) launch(key replayKey, ds []idisp) error {
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
	return g.Launch()
}

// queue queues ds without waiting.
func (d *cudaIntDev) queue(ds []idisp) error {
	if err := d.launches(ds); err != nil {
		return err
	}
	return d.dev.Queue(d.ls...)
}

// cudaPipe: pipelined decoding's page-locked copies and event (findFeed).
type cudaPipe struct {
	h, stH, resH []byte
	ev           *cuda.Event
}

func (d *cudaIntDev) pipeline() (*cudaPipe, error) {
	if d.pipe != nil {
		return d.pipe, nil
	}
	h, err := d.dev.HostAlloc(48)
	if err != nil {
		return nil, err
	}
	ev, err := d.dev.NewEvent()
	if err != nil {
		d.dev.HostFree(h)
		return nil, err
	}
	d.pipe = &cudaPipe{h: h, stH: h[:16], resH: h[16:48], ev: ev}
	return d.pipe, nil
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
			Block: [3]uint32{uint32(x.Threads.W), uint32(x.Threads.H), uint32(x.Threads.D)}, SharedB: cudaIntDyn[k.Name]})
	}
	return nil
}
