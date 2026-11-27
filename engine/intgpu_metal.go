package engine

// The integer models' Metal backend (gpu_int.metal): the model file wrapped zero-copy, buffers in shared memory.

import (
	_ "embed"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/benallfree/nollama/engine/internal/metal"
)

//go:embed gpu_int.metal
var metalIntSource string

//go:embed gpu_int_sg.metal
var metalIntSGSource string

//go:embed gpu_int_sga.metal
var metalIntSGASource string

// intKernels are the integer models' kernels, the same names in gpu_int.metal and gpu_int.cu.
var intKernels = []string{"int_embed", "int_norm", "int_matvec", "int_qkv", "int_blockf", "int_mlp", "int_table1",
	"int_table2", "int_table3", "int_table4", "int_attn1", "int_attn2", "int_attn3", "int_gemm", "int_attnb64", "int_attnb128",
	"int_counts", "int_bscores64", "int_bscores128", "int_bpv64", "int_bpv128", "int_bcomb", "int_find", "int_mlpb"}

type metalIntDev struct {
	dev   *metal.Device
	ps    map[string]*metal.Pipeline
	buf   []metal.Dispatch // run's scratch
	chunk int              // dispatches per command buffer (0: the package's default)
	pvBR  [2]int           // rows per block of int_bpv64, int_bpv128 when they aren't gpu_int.metal's (64, 32)
}

// metalBigModel: models at least this big (file bytes) commit a command buffer every metalBigChunk dispatches. A
// 7B batch's dispatches are long, and 48 to a buffer kept the GPU busy long enough for macOS's watchdog to kill the
// process ("Impacting Interactivity", after 707 s); 6 to a buffer ran clean.
const (
	metalBigModel = 3 << 30
	metalBigChunk = 6
)

type metalIntBuf struct{ b *metal.Buffer }

func (b *metalIntBuf) size() int { return len(b.b.Bytes()) }

func openIntMetal(m *IntModel) (intDev, ibuf, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, nil, err
	}
	if !dev.Unified {
		dev.Close()
		return nil, nil, errors.New("metal: GPUs without unified memory aren't supported")
	}
	ps, err := dev.Compile(metalIntSource, append(intKernels[:len(intKernels):len(intKernels)], "int_matvec8")...)
	if err != nil {
		dev.Close()
		return nil, nil, err
	}
	for _, k := range []string{"int_qkv", "int_blockf", "int_counts"} {
		if ps[k].MaxThreads < intTG {
			dev.Close()
			return nil, nil, fmt.Errorf("metal: %s runs at most %d threads per group, needs %d", k, ps[k].MaxThreads, intTG)
		}
	}
	// int_gemm through simdgroup matrices: the same integers, 2-4x faster. Kept only if it compiles here (Apple GPU
	// family 7+) and fits the 256-thread groups int_gemm's dispatch uses; NOLLAMA_INT_NOSG=1 keeps the plain kernel.
	if os.Getenv("NOLLAMA_INT_NOSG") == "" {
		src := "#define AQ " + strconv.Itoa(intActBits) + "\n" + metalIntSGSource
		if sp, err := dev.Compile(src, "int_gemm_sg"); err == nil && sp["int_gemm_sg"].MaxThreads >= 256 {
			ps["int_gemm"] = sp["int_gemm_sg"]
		}
	}
	// batched attention through simdgroup matrices (gpu_int_sga.metal): the same integers; NOLLAMA_INT_NOSGA=1 keeps
	// the plain kernels. Its weights x V kernels take 32-row blocks at head size 64, 16 at 128 (pvBR).
	pvBR := [2]int{}
	if os.Getenv("NOLLAMA_INT_NOSGA") == "" {
		names := []string{"int_bscores_sg64", "int_bscores_sg128", "int_bpv_sg64", "int_bpv_sg128"}
		if sp, err := dev.Compile(metalIntSource+"\n"+metalIntSGASource, names...); err == nil {
			ok := true
			for _, k := range names {
				ok = ok && sp[k].MaxThreads >= 256
			}
			if ok {
				ps["int_bscores64"], ps["int_bscores128"] = sp["int_bscores_sg64"], sp["int_bscores_sg128"]
				ps["int_bpv64"], ps["int_bpv128"] = sp["int_bpv_sg64"], sp["int_bpv_sg128"]
				pvBR = [2]int{32, 16}
			}
		}
	}
	w, err := dev.Wrap(m.g.data)
	if err != nil {
		dev.Close()
		return nil, nil, err
	}
	d := &metalIntDev{dev: dev, ps: ps, pvBR: pvBR}
	if int64(len(m.g.data)) >= metalBigModel {
		d.chunk = metalBigChunk
	}
	return d, &metalIntBuf{w}, nil
}

func (d *metalIntDev) name() string { return "metal (" + d.dev.Name + ")" }
func (d *metalIntDev) close()       { d.dev.Close() }

func (d *metalIntDev) alloc(n int) (ibuf, error) {
	b, err := d.dev.NewBuffer(n)
	if err != nil {
		return nil, err
	}
	return &metalIntBuf{b}, nil
}

func (d *metalIntDev) free(b ibuf) { b.(*metalIntBuf).b.Release() }
func (d *metalIntDev) write(dst ibuf, off int, src []byte) {
	copy(dst.(*metalIntBuf).b.Bytes()[off:], src)
}
func (d *metalIntDev) read(src ibuf, off int, dst []byte) {
	copy(dst, src.(*metalIntBuf).b.Bytes()[off:])
}
func (d *metalIntDev) host(b ibuf) []byte                   { return b.(*metalIntBuf).b.Bytes() }
func (d *metalIntDev) lastGPU() time.Duration               { return d.dev.LastGPU }
func (d *metalIntDev) replay(_ replayKey, ds []idisp) error { return d.run(ds) }
func (d *metalIntDev) graphs() bool                         { return false }
func (d *metalIntDev) forget(any)                           {}

func (d *metalIntDev) arg(a iarg) metal.Arg {
	if a.Buf == nil {
		return metal.Arg{Bytes: a.Bytes}
	}
	return metal.Arg{Buf: a.Buf.(*metalIntBuf).b, Offset: a.Offset}
}

// prepare moves the constant arguments into one buffer (256-byte aligned), so every argument is a buffer and Run binds
// a launch's arguments in one call, and converts the arguments once.
func (d *metalIntDev) prepare(ds []idisp, alloc func(int) (ibuf, error)) error {
	var consts []byte
	type slot struct{ d, a, off int }
	var slots []slot
	for di := range ds {
		for ai, a := range ds[di].Args {
			if a.Buf == nil {
				slots = append(slots, slot{di, ai, len(consts)})
				consts = append(consts, a.Bytes...)
				consts = append(consts, make([]byte, (256-len(consts)%256)%256)...)
			}
		}
	}
	cb, err := alloc(max(len(consts), 16))
	if err != nil {
		return err
	}
	d.write(cb, 0, consts)
	for _, sl := range slots {
		args := append([]iarg(nil), ds[sl.d].Args...)
		args[sl.a] = iarg{Buf: cb, Offset: sl.off}
		ds[sl.d].Args = args
	}
	for i := range ds {
		pre := make([]metal.Arg, len(ds[i].Args))
		for j, a := range ds[i].Args {
			pre[j] = d.arg(a)
		}
		ds[i].pre = pre
	}
	return nil
}

func (d *metalIntDev) run(ds []idisp) error {
	d.buf = d.buf[:0]
	for _, x := range ds {
		args, ok := x.pre.([]metal.Arg)
		if !ok {
			args = make([]metal.Arg, len(x.Args))
			for j, a := range x.Args {
				args[j] = d.arg(a)
			}
		}
		p := d.ps[x.K]
		if p == nil {
			return fmt.Errorf("metal: no kernel %s", x.K)
		}
		// more threads per group than the kernel can run gives wrong results without an error from Metal
		if n := x.Threads.W * x.Threads.H * x.Threads.D; int(n) > p.MaxThreads {
			return fmt.Errorf("metal: %s runs at most %d threads per group, dispatched with %d", x.K, p.MaxThreads, n)
		}
		d.buf = append(d.buf, metal.Dispatch{P: p, Args: args, Groups: metal.Size{W: x.Groups.W, H: x.Groups.H, D: x.Groups.D},
			Threads: metal.Size{W: x.Threads.W, H: x.Threads.H, D: x.Threads.D}})
	}
	if d.chunk > 0 {
		defer metal.SetRunChunk(metal.SetRunChunk(d.chunk))
	}
	return d.dev.Run(d.buf...)
}
func (d *metalIntDev) planes() bool { return false }
