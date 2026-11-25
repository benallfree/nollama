//go:build !amd64

package engine

import "math/rand/v2"

type kabKern struct {
	name string
	run  func()
	toks int
}

func kab4x16(*rand.Rand, int, int) []kabKern { return nil }
