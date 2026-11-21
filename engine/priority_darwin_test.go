package engine

import "testing"

// threadLowered reports whether the calling thread runs at utility QoS or below. The caller locks its goroutine to the
// thread.
func threadLowered(t *testing.T) bool {
	t.Helper()
	q := threadQoS()
	return q != 0 && q <= qosClassUtility // 0 is QOS_CLASS_UNSPECIFIED (a plain thread)
}
