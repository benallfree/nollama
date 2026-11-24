package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// testSignal is the deterministic signal testdata/logmel_80.f32 was computed from: a 440 Hz tone plus a chirp, one
// second long, with a silent stretch.
func testSignal() []float32 {
	x := make([]float32, SampleRate)
	for i := range x {
		n := float64(i)
		x[i] = float32(0.5*math.Sin(2*math.Pi*440*n/SampleRate) + 0.3*math.Sin(2*math.Pi*(200+3000*n/SampleRate)*n/SampleRate))
	}
	clear(x[4000:4400])
	return x
}

// TestLogMelReference checks LogMel and MelFilters against a reference computed by a numpy port of
// openai/whisper's audio.log_mel_spectrogram (np.fft.rfft, reflect padding, periodic Hann) with the mel filterbank
// shipped in Whisper's model files (mel_filters.npz, librosa's Slaney filters).
func TestLogMelReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/logmel_80.f32")
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(raw)/4)
	binary.Read(bytes.NewReader(raw), binary.LittleEndian, want)
	got := LogMel(testSignal(), MelFilters(80), 80)
	if len(got) != len(want) || len(got) != 80*100 {
		t.Fatalf("got %d values, want %d (80 x 100)", len(got), len(want))
	}
	var maxDiff float64
	for i := range got {
		maxDiff = math.Max(maxDiff, math.Abs(float64(got[i]-want[i])))
	}
	t.Logf("max abs difference %.2e", maxDiff)
	// The log-mel values span about 2 (8 decades of power / 4); 1e-4 is far below what changes a transcript.
	if maxDiff > 1e-4 {
		t.Fatalf("log-mel differs from the reference by %g", maxDiff)
	}
}

func TestMelFiltersShape(t *testing.T) {
	for _, n := range []int{80, 128} {
		f := MelFilters(n)
		if len(f) != n*(NFFT/2+1) {
			t.Fatalf("%d mels: %d weights", n, len(f))
		}
		for m := 0; m < n; m++ {
			var s float32
			for _, v := range f[m*(NFFT/2+1) : (m+1)*(NFFT/2+1)] {
				s += v
			}
			if s <= 0 {
				t.Fatalf("%d mels: filter %d is empty", n, m)
			}
		}
	}
}

func TestLogMelSilence(t *testing.T) {
	got := LogMel(make([]float32, 3200), MelFilters(80), 80)
	for _, v := range got {
		if v != (-10+4)/4.0 {
			t.Fatalf("silence gave %v, want -1.5", v)
		}
	}
}

func TestWAVRoundTrip(t *testing.T) {
	x := testSignal()
	var b bytes.Buffer
	if err := EncodeWAV(&b, x); err != nil {
		t.Fatal(err)
	}
	y, err := DecodeWAV(&b)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != len(x) {
		t.Fatalf("got %d samples, want %d", len(y), len(x))
	}
	for i := range x {
		if d := math.Abs(float64(x[i] - y[i])); d > 1e-4 { // 16-bit rounding, and scaling by 32767 out and 32768 in
			t.Fatalf("sample %d: %v vs %v", i, y[i], x[i])
		}
	}
}

func TestResample(t *testing.T) {
	// a 1 kHz tone at 44.1 kHz resampled to 16 kHz is still a 1 kHz tone
	const from = 44100
	x := make([]float32, from/2)
	for i := range x {
		x[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / from))
	}
	y := Resample(x, from, SampleRate)
	if len(y) != SampleRate/2 {
		t.Fatalf("got %d samples", len(y))
	}
	var maxErr float64
	for i := 200; i < len(y)-200; i++ {
		want := math.Sin(2 * math.Pi * 1000 * float64(i) / SampleRate)
		maxErr = math.Max(maxErr, math.Abs(float64(y[i])-want))
	}
	if maxErr > 0.01 {
		t.Fatalf("resampled tone off by %g", maxErr)
	}
}
