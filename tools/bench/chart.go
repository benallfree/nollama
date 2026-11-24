package main

// SVG charts for bench report: a summary of every comparison (nollama ÷ Ollama per metric) and one dot chart per
// metric. They're plain SVG with light and dark colors, so GitHub shows them in a README.

import (
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const chartStyle = `<style>
.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}
.n{fill:#2a78d6}.o{fill:#eb6834}.tie{fill:#8a8984}.link{stroke:#c3c2b7}.band{fill:#f0efec}.ring{stroke:#fcfcfb}
@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#2c2c2a}
.axis{stroke:#383835}.n{fill:#3987e5}.o{fill:#d95926}.tie{fill:#8a8984}.link{stroke:#55544f}.band{fill:#2c2c2a}.ring{stroke:#1a1a19}}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
</style>`

func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	dash := false
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
			dash = false
		} else if !dash && b.Len() > 0 {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.TrimSuffix(b.String(), "-")
}

// logAxis maps values in [lo, hi] (lo > 0) onto [x0, x1] on a log scale.
type logAxis struct{ lo, hi, x0, x1 float64 }

func (a logAxis) x(v float64) float64 {
	return a.x0 + (math.Log(v)-math.Log(a.lo))/(math.Log(a.hi)-math.Log(a.lo))*(a.x1-a.x0)
}

// ticks are 1-2-5 steps inside the axis.
func (a logAxis) ticks() []float64 {
	var out []float64
	for p := math.Floor(math.Log10(a.lo)); p <= math.Ceil(math.Log10(a.hi)); p++ {
		for _, m := range []float64{1, 2, 5} {
			v := m * math.Pow(10, p)
			if v >= a.lo*0.999 && v <= a.hi*1.001 {
				out = append(out, v)
			}
		}
	}
	return out
}

// niceBounds widens [lo, hi] to the nearest 1-2-5 steps.
func niceBounds(lo, hi float64) (float64, float64) {
	down, up := math.Pow(10, math.Floor(math.Log10(lo))), math.Pow(10, math.Floor(math.Log10(hi)))
	for _, m := range []float64{5, 2, 1} {
		if m*down <= lo {
			down *= m
			break
		}
	}
	for _, m := range []float64{1, 2, 5, 10} {
		if m*up >= hi {
			up *= m
			break
		}
	}
	return down, up
}

func tickLabel(v float64, unit string) string {
	switch unit {
	case "s":
		if v < 1 {
			return fmt.Sprintf("%g ms", v*1000)
		}
		return fmt.Sprintf("%g s", v)
	case "×":
		return fmt.Sprintf("%g×", v)
	}
	return fmt.Sprintf("%g", v)
}

type svgBuf struct{ strings.Builder }

func (b *svgBuf) f(format string, a ...any) { fmt.Fprintf(&b.Builder, format, a...) }

func (b *svgBuf) text(x, y float64, class, anchor string, size int, weight, s string) {
	w := ""
	if weight != "" {
		w = ` font-weight="` + weight + `"`
	}
	b.f(`<text x="%.1f" y="%.1f" class="%s" text-anchor="%s" font-size="%d"%s>%s</text>`+"\n", x, y, class, anchor, size, w, html.EscapeString(s))
}

func (b *svgBuf) legend(x, y float64) {
	b.f(`<circle cx="%.1f" cy="%.1f" r="5" class="n"/>`, x, y-4)
	b.text(x+10, y, "t1", "start", 12, "", "nollama")
	b.f(`<circle cx="%.1f" cy="%.1f" r="5" class="o"/>`, x+80, y-4)
	b.text(x+90, y, "t1", "start", 12, "", "Ollama")
}

func ratioOf(m reportMetric, p reportPair) (float64, bool) {
	a, b := m.get(p.ours), m.get(p.theirs)
	if a <= 0 || b <= 0 {
		return 0, false
	}
	if m.higher {
		return a / b, true
	}
	return b / a, true
}

// summarySVG: one row per metric, one dot per comparison at nollama's advantage (log scale, 1× in the middle).
func summarySVG(pairs []reportPair) string {
	const w, left, right, top, rowH = 900.0, 210.0, 170.0, 86.0, 46.0
	h := top + rowH*float64(len(reportMetrics)) + 50
	lo, hi := 1/30.0, 30.0
	ax := logAxis{lo, hi, left, w - right}
	var b svgBuf
	b.f(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" width="%.0f" height="%.0f" role="img" aria-label="nollama compared with Ollama, every metric">`+"\n", w, h, w, h)
	b.f("%s\n", chartStyle)
	b.f(`<rect class="bg" width="%.0f" height="%.0f" rx="8"/>`+"\n", w, h)
	b.text(20, 30, "t1", "start", 17, "600", "nollama vs Ollama, CPU on small cloud servers")
	b.text(20, 52, "t2", "start", 12, "", "Each dot is one machine and model. Right of 1× nollama is ahead, left of it Ollama is; the shaded band is within 5%.")
	bandX0, bandX1 := ax.x(1/1.05), ax.x(1.05)
	y0, y1 := top-8, top+rowH*float64(len(reportMetrics))-8
	b.f(`<rect class="band" x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>`+"\n", bandX0, y0, bandX1-bandX0, y1-y0)
	for _, t := range []float64{1.0 / 30, 0.1, 1.0 / 3, 1, 3, 10, 30} {
		x := ax.x(t)
		class := "grid"
		if t == 1 {
			class = "axis"
		}
		b.f(`<line class="%s" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke-width="1"/>`+"\n", class, x, x, y0, y1)
		label := fmt.Sprintf("%.3g×", t)
		if t < 1 {
			label = fmt.Sprintf("1/%.0f", 1/t)
		}
		b.text(x, y1+18, "t2", "middle", 11, "", label)
	}
	b.text(ax.x(1)-10, top-16, "o", "end", 12, "600", "← Ollama ahead")
	b.text(ax.x(1)+10, top-16, "n", "start", 12, "600", "nollama ahead →")
	b.text(w-right+20, top-16, "t2", "start", 11, "", "nollama · tie · Ollama")
	for i, m := range reportMetrics {
		cy := top + rowH*float64(i) + 10
		b.text(left-14, cy+4, "t1", "end", 13, "", m.title)
		var nw, tie, ow int
		for _, p := range pairs {
			r, ok := ratioOf(m, p)
			if !ok {
				continue
			}
			class := "tie"
			switch {
			case r > 1.05:
				class, nw = "n", nw+1
			case r < 1/1.05:
				class, ow = "o", ow+1
			default:
				tie++
			}
			r = math.Max(lo, math.Min(hi, r))
			b.f(`<circle class="%s ring" cx="%.1f" cy="%.1f" r="5.5" stroke-width="1.5" fill-opacity="0.85"><title>%s, %s: %.2f×</title></circle>`+"\n",
				class, ax.x(r), cy, html.EscapeString(p.machine), html.EscapeString(p.model), r)
		}
		b.text(w-right+20, cy+4, "t1", "start", 13, "", fmt.Sprintf("%d · %d · %d", nw, tie, ow))
	}
	b.f("</svg>\n")
	return b.String()
}

// metricSVG: one row per machine and model, nollama's and Ollama's values as two dots joined by a line (log scale).
func metricSVG(m reportMetric, pairs []reportPair) string {
	var rows []reportPair
	lo, hi := math.Inf(1), 0.0
	for _, p := range pairs {
		a, b := m.get(p.ours), m.get(p.theirs)
		if a <= 0 || b <= 0 {
			continue
		}
		rows = append(rows, p)
		lo, hi = math.Min(lo, math.Min(a, b)), math.Max(hi, math.Max(a, b))
	}
	lo, hi = niceBounds(lo, hi)
	const w, left, right, top, rowH, groupGap = 900.0, 250.0, 40.0, 92.0, 22.0, 14.0
	groups := 0
	for i, p := range rows {
		if i == 0 || rows[i-1].machine != p.machine {
			groups++
		}
	}
	h := top + rowH*float64(len(rows)) + groupGap*float64(groups) + 40
	ax := logAxis{lo, hi, left, w - right}
	var b svgBuf
	b.f(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" width="%.0f" height="%.0f" role="img" aria-label="%s: nollama and Ollama">`+"\n", w, h, w, h, html.EscapeString(m.title))
	b.f("%s\n", chartStyle)
	b.f(`<rect class="bg" width="%.0f" height="%.0f" rx="8"/>`+"\n", w, h)
	b.text(20, 30, "t1", "start", 17, "600", m.title)
	better := "Lower is better."
	if m.higher {
		better = "Higher is better."
	}
	b.text(20, 52, "t2", "start", 12, "", m.about+" "+better+" Log scale.")
	b.legend(20, 76)
	plotBottom := h - 32
	for _, t := range ax.ticks() {
		x := ax.x(t)
		b.f(`<line class="grid" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke-width="1"/>`+"\n", x, x, top-6, plotBottom)
		b.text(x, plotBottom+16, "t2", "middle", 11, "", tickLabel(t, m.unit))
	}
	y := top
	for i, p := range rows {
		if i == 0 || rows[i-1].machine != p.machine {
			y += groupGap
			b.text(20, y, "t1", "start", 12, "600", p.machine)
			y += 2
		}
		cy := y + rowH/2
		b.text(left-14, cy+4, "t2", "end", 12, "", p.model)
		a, o := m.get(p.ours), m.get(p.theirs)
		xa, xo := ax.x(a), ax.x(o)
		b.f(`<line class="link" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke-width="2"/>`+"\n", xa, xo, cy, cy)
		// value labels go outside the pair: the lower value on the left, the higher on the right
		lv, rv := a, o
		lx, rx := xa, xo
		if xa > xo {
			lv, rv, lx, rx = o, a, xo, xa
		}
		b.text(lx-9, cy+4, "t2", "end", 10, "", fmtVal(lv, m.unit))
		b.text(rx+9, cy+4, "t2", "start", 10, "", fmtVal(rv, m.unit))
		b.f(`<circle class="o ring" cx="%.1f" cy="%.1f" r="5" stroke-width="1.5"><title>Ollama: %s</title></circle>`+"\n", xo, cy, fmtVal(o, m.unit))
		b.f(`<circle class="n ring" cx="%.1f" cy="%.1f" r="5" stroke-width="1.5"><title>nollama: %s</title></circle>`+"\n", xa, cy, fmtVal(a, m.unit))
		y += rowH
	}
	b.f("</svg>\n")
	return b.String()
}

// writeCharts writes summary.svg and one SVG per metric into dir.
func writeCharts(dir string, reps []VPSReport) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	pairs := reportPairs(reps)
	if err := os.WriteFile(filepath.Join(dir, "summary.svg"), []byte(summarySVG(pairs)), 0o644); err != nil {
		return err
	}
	for _, m := range reportMetrics {
		if err := os.WriteFile(filepath.Join(dir, slug(m.title)+".svg"), []byte(metricSVG(m, pairs)), 0o644); err != nil {
			return err
		}
	}
	return nil
}
