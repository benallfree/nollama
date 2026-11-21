//go:build linux

package cuda

import "github.com/ebitengine/purego"

func openLibrary(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func symbol(lib uintptr, name string) (uintptr, error) { return purego.Dlsym(lib, name) }
