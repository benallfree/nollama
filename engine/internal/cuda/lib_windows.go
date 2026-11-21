//go:build windows

package cuda

import "syscall"

func openLibrary(name string) (uintptr, error) {
	h, err := syscall.LoadLibrary(name)
	return uintptr(h), err
}

func symbol(lib uintptr, name string) (uintptr, error) {
	p, err := syscall.GetProcAddress(syscall.Handle(lib), name)
	return uintptr(p), err
}
