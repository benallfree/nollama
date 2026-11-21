package engine

// Portable float math. The Go compiler may fuse x*y+z into one FMA instruction on arm64, ppc64le and s390x (and
// amd64 with GOAMD64=v3) but not on plain amd64, so the same Go source can round differently per architecture. An
// explicit conversion, float64(x*y) or float32(x*y), rounds the product and blocks the fusion. The standard library's
// math.Sin/Cos/Exp/Log/Pow are written without such conversions (and some have per-architecture assembly), so their
// results also differ in the last bit between arm64 and amd64. The functions here give the same bits everywhere.

// sincos is math.Sincos for 0 <= x < 2^29 (Cephes, as in Go's math/sin.go), with every product rounded.
func sincos(x float64) (sin, cos float64) {
	const (
		pi4a = 7.85398125648498535156e-1  // 0x3fe921fb40000000, Pi/4 split into three parts
		pi4b = 3.77489470793079817668e-8  // 0x3e64442d00000000,
		pi4c = 2.69515142907905952645e-15 // 0x3ce8469898cc5170,
	)
	j := uint64(float64(x * (4 / 3.14159265358979323846264338327950288419716939937510582097494459)))
	y := float64(j)
	if j&1 == 1 {
		j++
		y++
	}
	j &= 7
	z := ((x - float64(y*pi4a)) - float64(y*pi4b)) - float64(y*pi4c)
	sinSign, cosSign := false, false
	if j > 3 {
		j -= 4
		sinSign, cosSign = !sinSign, !cosSign
	}
	if j > 1 {
		cosSign = !cosSign
	}
	zz := float64(z * z)
	pc := -1.13585365213876817300e-11
	pc = float64(pc*zz) + 2.08757008419747316778e-9
	pc = float64(pc*zz) + -2.75573141792967388112e-7
	pc = float64(pc*zz) + 2.48015872888517045348e-5
	pc = float64(pc*zz) + -1.38888888888730564116e-3
	pc = float64(pc*zz) + 4.16666666666665929218e-2
	ps := 1.58962301576546568060e-10
	ps = float64(ps*zz) + -2.50507477628578072866e-8
	ps = float64(ps*zz) + 2.75573136213857245213e-6
	ps = float64(ps*zz) + -1.98412698295895385996e-4
	ps = float64(ps*zz) + 8.33333333332211858878e-3
	ps = float64(ps*zz) + -1.66666666666666307295e-1
	cos = 1.0 - float64(0.5*zz) + float64(float64(zz*zz)*pc)
	sin = z + float64(float64(z*zz)*ps)
	if j == 1 || j == 2 {
		sin, cos = cos, sin
	}
	if cosSign {
		cos = -cos
	}
	if sinSign {
		sin = -sin
	}
	return sin, cos
}
