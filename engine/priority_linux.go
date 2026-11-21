package engine

import "syscall"

const lowPrioritySupported = true

// lowNice is the nice value of a thread doing the work (0 is normal, 19 the lowest).
const lowNice = 10

func lowerThreadPriority() error {
	return syscall.Setpriority(syscall.PRIO_PROCESS, syscall.Gettid(), lowNice)
}
