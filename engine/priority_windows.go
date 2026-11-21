package engine

import "syscall"

const lowPrioritySupported = true

var (
	kernel32              = syscall.NewLazyDLL("kernel32.dll")
	procGetCurrentThread  = kernel32.NewProc("GetCurrentThread")
	procSetThreadPriority = kernel32.NewProc("SetThreadPriority")
)

const threadPriorityLowest = -2 // THREAD_PRIORITY_LOWEST

func lowerThreadPriority() error {
	h, _, _ := procGetCurrentThread.Call()
	lowest := threadPriorityLowest
	ok, _, err := procSetThreadPriority.Call(h, uintptr(lowest))
	if ok == 0 {
		return err
	}
	return nil
}
