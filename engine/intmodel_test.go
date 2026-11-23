package engine

import (
	"bufio"
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
