package engine

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The integer codec's conformance tests need the converted models (GGUF, several hundred MB to 8 GB): set
// NOLLAMA_INTCODEC_DIR to the directory holding them (file names as in intdata/models-v1.json). The vectors in
// testdata/intcodec come from the compress13k reference implementation. NOLLAMA_INTCODEC_LONG=1 adds the full-file
// archives (minutes on a CPU).

func intModelPath(t *testing.T, id int) string {
	dir := os.Getenv("NOLLAMA_INTCODEC_DIR")
	if dir == "" {
		t.Skip("NOLLAMA_INTCODEC_DIR not set")
	}
	mi, ok := IntModelByID(id)
	if !ok {
		t.Fatalf("no model %d in the manifest", id)
	}
	p := filepath.Join(dir, mi.Bundle.File)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s not found", p)
	}
	return p
}

func loadIntTest(t *testing.T, id int) *IntModel {
	m, err := LoadIntModel(intModelPath(t, id), 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func readVec(t *testing.T, name string) []byte {
	b, err := os.ReadFile(filepath.Join("testdata", "intcodec", name))
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
	m := loadIntTest(t, 1)
	for _, f := range []string{"tiny", "slice", "bjj"} {
		got := m.encodeText(readVec(t, f+".html"))
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

func checkTables(t *testing.T, id int, input, tables string) {
	m := loadIntTest(t, id)
	ids := vecIDs(t, input)
	var want []string
	sc := bufio.NewScanner(bytes.NewReader(readVec(t, tables)))
	for sc.Scan() {
		want = append(want, strings.Fields(sc.Text())[1])
	}
	st := m.newState(len(want))
	prev := m.bos
	t0 := time.Now()
	for k, w := range want {
		st.step(prev)
		if got := tableHash(st.cum); got != w {
			t.Fatalf("position %d: table hash %s, want %s", k, got, w)
		}
		prev = ids[k]
	}
	t.Logf("%d tables match (%.1f ms/token)", len(want), float64(time.Since(t0).Milliseconds())/float64(len(want)))
}

func TestIntTables05B(t *testing.T) { checkTables(t, 1, "slice.ids.json", "m1-slice-tables.txt") }
func TestIntTables15B(t *testing.T) { checkTables(t, 2, "tiny.ids.json", "m2-tiny-tables.txt") }

func checkArchive(t *testing.T, id int, input, archive string, pack bool) {
	m := loadIntTest(t, id)
	want, a := readVec(t, input), readVec(t, archive)
	t0 := time.Now()
	got, err := m.Unpack(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s: unpacked %d bytes, differ from %s", archive, len(got), input)
	}
	_, n, _, _ := IntArchiveInfo(a)
	t.Logf("%s: unpacked %d tokens in %v (%.1f ms/token)", archive, n, time.Since(t0).Round(time.Millisecond),
		float64(time.Since(t0).Milliseconds())/float64(n))
	if pack {
		p, err := m.Pack(want, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(p, a) {
			t.Fatalf("%s: packed %d bytes, not byte-identical to the vector (%d bytes)", input, len(p), len(a))
		}
	}
}

func TestIntArchiveTiny(t *testing.T)  { checkArchive(t, 1, "tiny.html", "m1-tiny.13k", true) }
func TestIntArchiveSlice(t *testing.T) { checkArchive(t, 1, "slice.html", "m1-slice.13k", true) }

func TestIntArchiveLong(t *testing.T) {
	if os.Getenv("NOLLAMA_INTCODEC_LONG") == "" {
		t.Skip("NOLLAMA_INTCODEC_LONG not set")
	}
	checkArchive(t, 1, "bjj.html", "m1-bjj.13k", false)
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

// TestIntConvert converts a Qwen2.5 bf16 snapshot directory (NOLLAMA_INTCODEC_SRC) and checks the result against the
// manifest's SHA-256 (NOLLAMA_INTCODEC_SRC_ID names the model id, default 1).
func TestIntConvert(t *testing.T) {
	src := os.Getenv("NOLLAMA_INTCODEC_SRC")
	if src == "" {
		t.Skip("NOLLAMA_INTCODEC_SRC not set")
	}
	id := 1
	if v := os.Getenv("NOLLAMA_INTCODEC_SRC_ID"); v != "" {
		fmt.Sscan(v, &id)
	}
	mi, _ := IntModelByID(id)
	dst := filepath.Join(t.TempDir(), mi.Bundle.File)
	t0 := time.Now()
	if err := ConvertIntModel(src, dst, nil); err != nil {
		t.Fatal(err)
	}
	h, err := FileSHA256(dst)
	if err != nil {
		t.Fatal(err)
	}
	if h != mi.Bundle.SHA256 {
		t.Fatalf("converted %s: sha256 %s, want %s", mi.Bundle.File, h, mi.Bundle.SHA256)
	}
	t.Logf("%s: converted in %v, sha256 matches the manifest", mi.Bundle.File, time.Since(t0).Round(time.Second))
}

func TestIntArchive7BTiny(t *testing.T) { checkArchive(t, 3, "tiny.html", "m3-tiny.13k", true) }
