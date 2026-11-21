//go:build !linux && !darwin

package engine

import "testing"

func threadLowered(t *testing.T) bool {
	t.Skip("no check for thread priority on this OS")
	return false
}
