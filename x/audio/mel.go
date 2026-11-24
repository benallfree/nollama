package audio

import (
	"math"
	"sync"
)

// Whisper's front end: a 25 ms window every 10 ms at 16 kHz.
const (
	NFFT      = 400
	HopLength = 160
	// ChunkSamples is a 30 s window, the length Whisper's encoder takes; ChunkFrames is its frame count.
	ChunkSamples = 30 * SampleRate
	ChunkFrames  = ChunkSamples / HopLength
)

// MelFilters returns the mel filterbank Whisper uses: librosa.filters.mel(sr=16000, n_fft=400, n_mels=nMels) with
// its defaults (Slaney mel scale, Slaney area normalization, 0 Hz to 8 kHz), as nMels rows of NFFT/2+1 weights.
func MelFilters(nMels int) []float32 {
	nBins := NFFT/2 + 1
	hzToMel := func(f float64) float64 { // Slaney: linear below 1 kHz, logarithmic above
		const fsp, minLogHz, logstep = 200.0 / 3, 1000.0, 0.06875177742094912 // log(6.4)/27
		if f < minLogHz {
			return f / fsp
		}
		return minLogHz/fsp + math.Log(f/minLogHz)/logstep
	}
	melToHz := func(m float64) float64 {
		const fsp, minLogHz, logstep = 200.0 / 3, 1000.0, 0.06875177742094912
		minLogMel := minLogHz / fsp
		if m < minLogMel {
			return m * fsp
		}
		return minLogHz * math.Exp(logstep*(m-minLogMel))
	}
	fmax := float64(SampleRate) / 2
	mlo, mhi := hzToMel(0), hzToMel(fmax)
	pts := make([]float64, nMels+2)
	for i := range pts {
		pts[i] = melToHz(mlo + (mhi-mlo)*float64(i)/float64(nMels+1))
	}
	out := make([]float32, nMels*nBins)
	for m := 0; m < nMels; m++ {
		lower, center, upper := pts[m], pts[m+1], pts[m+2]
		enorm := 2 / (upper - lower)
		for k := 0; k < nBins; k++ {
			f := fmax * float64(k) / float64(nBins-1)
			w := math.Max(0, math.Min((f-lower)/(center-lower), (upper-f)/(upper-center)))
			out[m*nBins+k] = float32(w * enorm)
		}
	}
	return out
}

// LogMel computes Whisper's log-mel spectrogram of 16 kHz samples, as whisper/audio.py's log_mel_spectrogram does:
// a centered STFT (reflect padding, periodic Hann window, NFFT 400, hop 160) dropping the last frame, power spectrum
// through filters (nMels rows of NFFT/2+1, as MelFilters returns or a model file carries), log10 clamped at 1e-10,
// floored at 8 below the maximum, then (x+4)/4. The result is nMels rows of len(pcm)/HopLength frames, row-major.
//
// To feed the encoder one 30 s window, pad pcm with zeros to at least ChunkSamples and take ChunkFrames frames from
// each row (Chunk does that).
func LogMel(pcm []float32, filters []float32, nMels int) []float32 {
	nBins := NFFT/2 + 1
	nFrames := len(pcm) / HopLength
	if nFrames == 0 {
		return nil
	}
	// reflect-pad NFFT/2 on both sides (torch.stft center=True, pad_mode="reflect")
	pad := NFFT / 2
	x := make([]float64, len(pcm)+2*pad)
	for i, v := range pcm {
		x[pad+i] = float64(v)
	}
	for i := 1; i <= pad; i++ {
		x[pad-i] = x[pad+min(i, len(pcm)-1)]
		x[pad+len(pcm)-1+i] = x[pad+max(len(pcm)-1-i, 0)]
	}
	win := hann()
	out := make([]float32, nMels*nFrames)
	workers := 8
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			frame := make([]float64, NFFT)
			re, im := make([]float64, NFFT), make([]float64, NFFT)
			pow := make([]float64, nBins)
			for t := w; t < nFrames; t += workers {
				seg := x[t*HopLength : t*HopLength+NFFT]
				silent := true
				for i, v := range seg {
					frame[i] = v * win[i]
					if v != 0 {
						silent = false
					}
				}
				if silent {
					clear(pow)
				} else {
					fft(frame, re, im)
					for k := 0; k < nBins; k++ {
						pow[k] = re[k]*re[k] + im[k]*im[k]
					}
				}
				for m := 0; m < nMels; m++ {
					fr := filters[m*nBins : (m+1)*nBins]
					var s float64
					for k, p := range pow {
						s += float64(fr[k]) * p
					}
					out[m*nFrames+t] = float32(math.Log10(math.Max(s, 1e-10)))
				}
			}
		}(w)
	}
	wg.Wait()
	mx := float32(math.Inf(-1))
	for _, v := range out {
		mx = max(mx, v)
	}
	for i, v := range out {
		out[i] = (max(v, mx-8) + 4) / 4
	}
	return out
}

// Chunk returns ChunkFrames frames of a log-mel spectrogram (nMels rows of frames each) starting at frame start,
// padding past the end with pad (the value silence takes in the same spectrogram).
func Chunk(mel []float32, nMels, start int, pad float32) []float32 {
	frames := len(mel) / nMels
	out := make([]float32, nMels*ChunkFrames)
	for m := 0; m < nMels; m++ {
		row := out[m*ChunkFrames : (m+1)*ChunkFrames]
		n := 0
		if start < frames {
			n = copy(row, mel[m*frames+start:(m+1)*frames])
		}
		for i := n; i < len(row); i++ {
			row[i] = pad
		}
	}
	return out
}

var hannOnce = sync.OnceValue(func() []float64 {
	w := make([]float64, NFFT)
	for i := range w {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/NFFT)) // periodic, as torch.hann_window
	}
	return w
})

func hann() []float64 { return hannOnce() }

// twCos, twSin are exp(-2πi·j/NFFT): every sub-transform's twiddles, since each length divides NFFT.
var twCos, twSin = func() (c, s [NFFT]float64) {
	for j := range c {
		s[j], c[j] = math.Sincos(-2 * math.Pi * float64(j) / NFFT)
	}
	return
}()

// fft computes the DFT of real x (len NFFT = 400 = 2^4·25) into re, im: radix-2 splits down to length 25, then a
// direct DFT.
func fft(x, re, im []float64) {
	n := len(x)
	xr := make([]float64, n)
	copy(xr, x)
	xi := make([]float64, n)
	fftRec(xr, xi, re, im)
}

func fftRec(xr, xi, re, im []float64) {
	n := len(xr)
	if n%2 != 0 {
		for k := 0; k < n; k++ {
			var sr, si float64
			for j := 0; j < n; j++ {
				idx := (j * k % n) * (NFFT / n)
				s, c := twSin[idx], twCos[idx]
				sr += xr[j]*c - xi[j]*s
				si += xr[j]*s + xi[j]*c
			}
			re[k], im[k] = sr, si
		}
		return
	}
	h := n / 2
	er, ei := make([]float64, h), make([]float64, h)
	or, oi := make([]float64, h), make([]float64, h)
	for i := 0; i < h; i++ {
		er[i], ei[i] = xr[2*i], xi[2*i]
		or[i], oi[i] = xr[2*i+1], xi[2*i+1]
	}
	Er, Ei := make([]float64, h), make([]float64, h)
	Or, Oi := make([]float64, h), make([]float64, h)
	fftRec(er, ei, Er, Ei)
	fftRec(or, oi, Or, Oi)
	for k := 0; k < h; k++ {
		s, c := twSin[k*(NFFT/n)], twCos[k*(NFFT/n)]
		tr := c*Or[k] - s*Oi[k]
		ti := c*Oi[k] + s*Or[k]
		re[k], im[k] = Er[k]+tr, Ei[k]+ti
		re[k+h], im[k+h] = Er[k]-tr, Ei[k]-ti
	}
}
