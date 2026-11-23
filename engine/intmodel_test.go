package engine

import (
	"bufio"
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/benallfree/nollama/engine/internal/metal"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unsafe"
)

// The integer model's conformance tests need the converted models (GGUF, 0.5 to 8 GB): set NOLLAMA_INTMODEL_DIR to
// the directory holding them. testdata/intmodel holds data from the reference implementation (compress13k): inputs,
// their token ids, and hashes of the frequency tables a session must produce.

var intModelFiles = map[string]struct{ file, sha256 string }{
	"0.5B": {"qwen2.5-coder-0.5b-int-v1.gguf", "2b9fa35d48b67e9170488af611179f46e7d151a4ca7bc015d75da9c1ee554b45"},
	"1.5B": {"qwen2.5-coder-1.5b-int-v1.gguf", "6d3e399f0493dbcf463f31586d3aedb790f7c5ea78104687deefdedc4951270d"},
	"7B":   {"qwen2.5-coder-7b-int-v1.gguf", "4f447ba7697edf55399910cbff7a74e261e19e6896e4a7dda3decb6053d9496a"},
}

func intModelPath(t *testing.T, size string) string {
	dir := os.Getenv("NOLLAMA_INTMODEL_DIR")
	if dir == "" {
		t.Skip("NOLLAMA_INTMODEL_DIR not set")
	}
	p := filepath.Join(dir, intModelFiles[size].file)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s not found", p)
	}
	return p
}

func loadIntTest(t *testing.T, size string) *IntModel { return loadIntDev(t, size, "off") }

// loadIntDev loads on the CPU (gpu "off") or a GPU ("auto"; the test skips when there is none).
func loadIntDev(t *testing.T, size, gpu string) *IntModel {
	m, err := LoadIntModelOptions(intModelPath(t, size), IntOptions{GPU: gpu})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if gpu != "off" && m.gpu == nil {
		t.Skip("no GPU: ", m.Device())
	}
	return m
}

func readVec(t *testing.T, name string) []byte {
	b, err := os.ReadFile(filepath.Join("testdata", "intmodel", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func vecIDs(t *testing.T, name string) []int {
	var ids []int
	if err := json.Unmarshal(readVec(t, name), &ids); err != nil {
		t.Fatal(err)
	}
	return ids
}

func tableHash(cum []int64) string {
	b := make([]byte, 8*len(cum))
	for i, v := range cum {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(v))
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])[:16]
}

func TestIntTokenizer(t *testing.T) {
	m := loadIntTest(t, "0.5B")
	for _, f := range []string{"tiny", "slice", "bjj"} {
		got := m.Tokenize(string(readVec(t, f+".html")))
		want := vecIDs(t, f+".ids.json")
		if len(got) != len(want) {
			t.Fatalf("%s: %d tokens, want %d", f, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: token %d is %d, want %d", f, i, got[i], want[i])
			}
		}
	}
}

func checkTables(t *testing.T, size, gpu, input, tables string) {
	m := loadIntDev(t, size, gpu)
	ids := vecIDs(t, input)
	var want []string
	sc := bufio.NewScanner(bytes.NewReader(readVec(t, tables)))
	for sc.Scan() {
		want = append(want, strings.Fields(sc.Text())[1])
	}
	ses, err := m.NewSession(len(want))
	if err != nil {
		t.Fatal(err)
	}
	defer ses.Close()
	prev := m.BOS()
	t0 := time.Now()
	for k, w := range want {
		cum, err := ses.Feed(prev)
		if err != nil {
			t.Fatal(err)
		}
		if got := tableHash(cum); got != w {
			t.Fatalf("position %d: table hash %s, want %s", k, got, w)
		}
		prev = ids[k]
	}
	t.Logf("%s: %d tables match (%.1f ms/token)", m.Device(), len(want), float64(time.Since(t0).Milliseconds())/float64(len(want)))
}

func TestIntTables05B(t *testing.T) {
	checkTables(t, "0.5B", "off", "slice.ids.json", "m1-slice-tables.txt")
}
func TestIntTables05BGPU(t *testing.T) {
	checkTables(t, "0.5B", "auto", "slice.ids.json", "m1-slice-tables.txt")
}
func TestIntTables15B(t *testing.T) {
	checkTables(t, "1.5B", "off", "tiny.ids.json", "m2-tiny-tables.txt")
}
func TestIntTables15BGPU(t *testing.T) {
	checkTables(t, "1.5B", "auto", "tiny.ids.json", "m2-tiny-tables.txt")
}

func TestIntDotRow(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, ng := range []int{1, 2, 14, 76, 296} {
		q, x, s := make([]int8, 64*ng), make([]int16, 64*ng), make([]int16, ng)
		for trial := 0; trial < 50; trial++ {
			for i := range q {
				q[i] = int8(r.Intn(255) - 127)
				x[i] = int16(r.Intn(65535) - 32767)
			}
			for i := range s {
				s[i] = int16(r.Intn(32767) + 1)
			}
			if trial == 0 { // extremes
				for i := range q {
					q[i], x[i] = -127, 32767
				}
				for i := range s {
					s[i] = 32767
				}
			}
			if got, want := intDotRow(q, x, s), intDotRowGo(q, x, s); got != want {
				t.Fatalf("ng %d trial %d: %d, want %d", ng, trial, got, want)
			}
		}
	}
}

func TestIntAttnKernels(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, n := range []int{8, 64, 128} {
		q, k := make([]int32, n), make([]int32, n)
		o1, o2 := make([]int64, n), make([]int64, n)
		for trial := 0; trial < 100; trial++ {
			for i := range q {
				q[i] = int32(r.Intn(65535) - 32767)
				k[i] = int32(r.Intn(1<<23-1) - (1<<22 - 1))
				o1[i] = r.Int63n(1<<52) - 1<<51
				o2[i] = o1[i]
			}
			if trial == 0 {
				for i := range q {
					q[i], k[i] = -32767, 1<<22-1
				}
			}
			if got, want := intDotQK(q, k), intDotQKGo(q, k); got != want {
				t.Fatalf("intDotQK n %d trial %d: %d, want %d", n, trial, got, want)
			}
			w := int64(r.Intn(32769))
			intAxpy(o1, w, k)
			intAxpyGo(o2, w, k)
			for i := range o1 {
				if o1[i] != o2[i] {
					t.Fatalf("intAxpy n %d trial %d lane %d: %d, want %d", n, trial, i, o1[i], o2[i])
				}
			}
		}
	}
}

// TestIntConvert converts a Qwen2.5 bf16 snapshot directory (NOLLAMA_INTMODEL_SRC, of the size in
// NOLLAMA_INTMODEL_SRC_SIZE, default 0.5B) and checks the result's SHA-256.
func TestIntConvert(t *testing.T) {
	src := os.Getenv("NOLLAMA_INTMODEL_SRC")
	if src == "" {
		t.Skip("NOLLAMA_INTMODEL_SRC not set")
	}
	size := cmp.Or(os.Getenv("NOLLAMA_INTMODEL_SRC_SIZE"), "0.5B")
	want := intModelFiles[size]
	dst := filepath.Join(t.TempDir(), want.file)
	t0 := time.Now()
	if err := ConvertIntModel(src, dst, nil); err != nil {
		t.Fatal(err)
	}
	h, err := FileSHA256(dst)
	if err != nil {
		t.Fatal(err)
	}
	if h != want.sha256 {
		t.Fatalf("converted %s: sha256 %s, want %s", want.file, h, want.sha256)
	}
	t.Logf("%s: converted in %v, sha256 matches", want.file, time.Since(t0).Round(time.Second))
}

// checkBatch: FeedBatch's counts equal those of stepping one token at a time (on the same device).
func checkBatch(t *testing.T, size, input string, n int) {
	m := loadIntDev(t, size, "auto")
	if m.gpu == nil {
		t.Skip("no GPU")
	}
	ids := vecIDs(t, input)
	if n > 0 && n < len(ids) {
		ids = ids[:n]
	}
	toks := append([]int{m.BOS()}, ids[:len(ids)-1]...)
	a, err := m.NewSession(len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	t0 := time.Now()
	got, err := a.FeedBatch(toks, ids)
	if err != nil {
		t.Fatal(err)
	}
	tb := time.Since(t0)
	b, err := m.NewSession(len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	t0 = time.Now()
	for k, tok := range toks {
		cum, err := b.Feed(tok)
		if err != nil {
			t.Fatal(err)
		}
		if w := countOf(cum, ids[k]); got[k] != w {
			t.Fatalf("position %d: batch %+v, step %+v", k, got[k], w)
		}
	}
	t.Logf("%s, %d positions: batch %.2f s (%.0f tok/s), step %.2f s", m.Device(), len(ids), tb.Seconds(),
		float64(len(ids))/tb.Seconds(), time.Since(t0).Seconds())
}

func TestIntBatch05B(t *testing.T) { checkBatch(t, "0.5B", "slice.ids.json", 0) }
func TestIntBatch15B(t *testing.T) { checkBatch(t, "1.5B", "slice.ids.json", 600) }
func TestIntBatch7B(t *testing.T)  { checkBatch(t, "7B", "slice.ids.json", 100) }

// TestIntBatchBJJ times FeedBatch over the whole BJJ input (no comparison; signal's vector test checks the archive).
func TestIntBatchBJJ(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	ids := vecIDs(t, "bjj.ids.json")
	toks := append([]int{m.BOS()}, ids[:len(ids)-1]...)
	a, err := m.NewSession(len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	t0 := time.Now()
	for k := 0; k < len(ids); k += 4096 {
		e := min(k+4096, len(ids))
		if _, err := a.FeedBatch(toks[k:e], ids[k:e]); err != nil {
			t.Fatal(err)
		}
		t.Logf("%5d: %.1f s, GPU %.1f s, slow batches %d", e, time.Since(t0).Seconds(), intBatchGPU.Seconds(), intBatchSlow)
	}
}

// TestIntBatchBJJProfile: after BJJ's batches, one batch at position 23000 (real keys and values), kernel by kernel.
func TestIntBatchBJJProfile(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	ids := vecIDs(t, "bjj.ids.json")
	toks := append([]int{m.BOS()}, ids[:len(ids)-1]...)
	a, err := m.NewSession(len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.FeedBatch(toks, ids); err != nil {
		t.Fatal(err)
	}
	gs := a.g
	b := gs.b
	pos, R := 23000, b.P
	tk := unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(b.toks.Bytes()))), R)
	tg := unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(b.tgt.Bytes()))), R)
	for i := range R {
		tk[i], tg[i] = int32(toks[pos+i]), int32(ids[pos+i])
	}
	binary.LittleEndian.PutUint32(b.params.Bytes()[0:], uint32(R))
	binary.LittleEndian.PutUint32(b.params.Bytes()[4:], uint32(pos))
	nch := uint((pos+R-1)/intKC + 1)
	ds := append([]metal.Dispatch(nil), b.ds...)
	for i := range ds {
		switch b.kind[i] {
		case scoreG:
			ds[i].Groups.H, ds[i].Groups.D = uint((R+b.scoreBR-1)/b.scoreBR), nch
		case pvG:
			ds[i].Groups.H, ds[i].Groups.D = uint((R+b.pvBR-1)/b.pvBR), nch
		}
	}
	for _, h := range b.head {
		ds = append(ds, h[0], h[1])
	}
	run := func(ds []metal.Dispatch) float64 {
		gs.g.dev.Run(ds...)
		return float64(gs.g.dev.LastGPU.Microseconds()) / 1000
	}
	line := ""
	for _, k := range []string{"int_gemm", "int_bscores64", "int_bpv64", "int_bcomb", "int_counts", "int_norm", "int_qkv"} {
		var sub []metal.Dispatch
		for _, d := range ds {
			if d.P.Name == k {
				sub = append(sub, d)
			}
		}
		line += fmt.Sprintf(" %s %.0f", k, run(sub))
	}
	t.Logf("pos %d, %d rows: %.0f ms |%s", pos, R, run(ds), line)
}
