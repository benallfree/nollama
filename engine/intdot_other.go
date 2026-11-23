//go:build !arm64

package engine

// intDotRow returns sum_g s[g] * sum_{k<64} q[64g+k]*x[64g+k] (exact).
func intDotRow(q []int8, x []int16, s []int16) int64 { return intDotRowGo(q, x, s) }

func intDotQK(q, k []int32) int64           { return intDotQKGo(q, k) }
func intAxpy(o []int64, w int64, v []int32) { intAxpyGo(o, w, v) }
