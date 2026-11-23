package engine

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
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
