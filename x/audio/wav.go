// Package audio reads WAV files and computes the log-mel spectrogram speech models such as Whisper take.
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// SampleRate is the rate speech models take: 16 kHz.
const SampleRate = 16000

// ReadWAVFile reads a WAV file as mono float32 samples in [-1, 1] at 16 kHz (see DecodeWAV).
func ReadWAVFile(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return DecodeWAV(f)
}

// DecodeWAV reads a RIFF/WAVE stream (PCM 8/16/24/32-bit integer or 32/64-bit float, any channel count and rate) as
// mono float32 samples at 16 kHz: channels are averaged and other rates resampled (Resample).
func DecodeWAV(r io.Reader) ([]float32, error) {
	pcm, rate, err := decodeWAV(r)
	if err != nil {
		return nil, err
	}
	if rate != SampleRate {
		pcm = Resample(pcm, rate, SampleRate)
	}
	return pcm, nil
}

func decodeWAV(r io.Reader) ([]float32, int, error) {
	var hdr [12]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, 0, fmt.Errorf("audio: reading WAV header: %w", err)
	}
	if string(hdr[0:4]) != "RIFF" || string(hdr[8:12]) != "WAVE" {
		return nil, 0, errors.New("audio: not a RIFF/WAVE file")
	}
	var (
		format, channels, bits uint16
		rate                   uint32
		haveFmt                bool
	)
	for {
		var ch [8]byte
		if _, err := io.ReadFull(r, ch[:]); err != nil {
			return nil, 0, errors.New("audio: WAV file has no data chunk")
		}
		id, size := string(ch[0:4]), binary.LittleEndian.Uint32(ch[4:8])
		switch id {
		case "fmt ":
			if size < 16 || size > 1<<16 {
				return nil, 0, errors.New("audio: bad fmt chunk")
			}
			b := make([]byte, size+size%2)
			if _, err := io.ReadFull(r, b); err != nil {
				return nil, 0, err
			}
			format = binary.LittleEndian.Uint16(b[0:])
			channels = binary.LittleEndian.Uint16(b[2:])
			rate = binary.LittleEndian.Uint32(b[4:])
			bits = binary.LittleEndian.Uint16(b[14:])
			if format == 0xFFFE && size >= 26 { // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the format code
				format = binary.LittleEndian.Uint16(b[24:])
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, 0, errors.New("audio: data chunk before fmt chunk")
			}
			if channels == 0 || rate == 0 {
				return nil, 0, errors.New("audio: bad fmt chunk")
			}
			data, err := io.ReadAll(io.LimitReader(r, int64(size)))
			if err != nil {
				return nil, 0, err
			}
			pcm, err := toMono(data, format, int(channels), int(bits))
			return pcm, int(rate), err
		default:
			if _, err := io.CopyN(io.Discard, r, int64(size+size%2)); err != nil {
				return nil, 0, errors.New("audio: WAV file has no data chunk")
			}
		}
	}
}

func toMono(data []byte, format uint16, channels, bits int) ([]float32, error) {
	bps := bits / 8
	var sample func(b []byte) float32
	switch {
	case format == 1 && bits == 8:
		sample = func(b []byte) float32 { return (float32(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768 }
	case format == 1 && bits == 24:
		sample = func(b []byte) float32 {
			return float32(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / (1 << 23)
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float32 { return float32(int32(binary.LittleEndian.Uint32(b))) / (1 << 31) }
	case format == 3 && bits == 32:
		sample = func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
	case format == 3 && bits == 64:
		sample = func(b []byte) float32 { return float32(math.Float64frombits(binary.LittleEndian.Uint64(b))) }
	default:
		return nil, fmt.Errorf("audio: unsupported WAV encoding (format %d, %d bits)", format, bits)
	}
	frame := bps * channels
	n := len(data) / frame
	out := make([]float32, n)
	inv := 1 / float32(channels)
	for i := range out {
		var s float32
		for c := 0; c < channels; c++ {
			s += sample(data[i*frame+c*bps:])
		}
		out[i] = s * inv
	}
	return out, nil
}

// Resample converts x from rate from to rate to with a windowed-sinc low-pass interpolator (16 zero crossings, Hann
// window). It is good enough for speech recognition, not for music production.
func Resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	ratio := float64(to) / float64(from)
	n := int(math.Floor(float64(len(x)) * ratio))
	out := make([]float32, n)
	cutoff := math.Min(1, ratio) // fraction of the input Nyquist frequency
	const zeros = 16
	half := zeros / cutoff // filter half-width in input samples
	for i := range out {
		t := float64(i) / ratio // position in the input
		lo, hi := int(math.Ceil(t-half)), int(math.Floor(t+half))
		var s, wsum float64
		for j := max(lo, 0); j <= min(hi, len(x)-1); j++ {
			d := (float64(j) - t) * cutoff
			w := 0.5 + 0.5*math.Cos(math.Pi*d/zeros) // Hann over ±zeros
			if d != 0 {
				w *= math.Sin(math.Pi*d) / (math.Pi * d)
			}
			s += w * float64(x[j])
			wsum += w
		}
		if wsum != 0 {
			out[i] = float32(s / wsum)
		}
	}
	return out
}

// EncodeWAV writes 16 kHz mono 16-bit PCM (for tests and tools).
func EncodeWAV(w io.Writer, pcm []float32) error {
	b := make([]byte, 44+2*len(pcm))
	copy(b[0:], "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*len(pcm)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], SampleRate)
	binary.LittleEndian.PutUint32(b[28:], SampleRate*2)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*len(pcm)))
	for i, v := range pcm {
		s := math.Round(float64(v) * 32767)
		s = math.Max(-32768, math.Min(32767, s))
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(int16(s)))
	}
	_, err := w.Write(b)
	return err
}
