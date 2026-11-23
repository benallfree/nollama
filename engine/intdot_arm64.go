//go:build arm64

package engine

//go:noescape
func intDotRowAsm(q *int8, x *int16, s *int16, ng int) int64

// intDotRow returns sum_g s[g] * sum_{k<64} q[64g+k]*x[64g+k] (exact; see intdot_arm64.s).
func intDotRow(q []int8, x []int16, s []int16) int64 {
	if len(s) == 0 {
		return 0
	}
	_ = q[len(s)*64-1]
	_ = x[len(s)*64-1]
	return intDotRowAsm(&q[0], &x[0], &s[0], len(s))
}

//go:noescape
func intDotQKAsm(q *int32, k *int32, n int) int64

//go:noescape
func intAxpyAsm(o *int64, w int64, v *int32, n int)

// intDotQK returns sum q[i]*k[i] (exact int64; len a multiple of 8).
func intDotQK(q, k []int32) int64 {
	if len(q)%8 != 0 {
		return intDotQKGo(q, k)
	}
	_ = k[len(q)-1]
	return intDotQKAsm(&q[0], &k[0], len(q))
}

// intAxpy does o[i] += w*v[i] (exact int64; |w| < 2^31, len a multiple of 4).
func intAxpy(o []int64, w int64, v []int32) {
	if len(o)%4 != 0 {
		intAxpyGo(o, w, v)
		return
	}
	_ = v[len(o)-1]
	intAxpyAsm(&o[0], w, &v[0], len(o))
}
