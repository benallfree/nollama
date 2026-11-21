package engine

import (
	"syscall"
	_ "unsafe" // go:linkname
)

const lowPrioritySupported = true

// QOS_CLASS_UTILITY, from <sys/qos.h>: below the default and user-initiated classes, above background. (Background,
// PRIO_DARWIN_BG, throttles a thread so hard that Go's runtime stalls behind it: a lowered thread can hold a runtime
// lock the rest of the program waits for.)
const qosClassUtility = 0x11

// pthread_set_qos_class_self_np is called through libSystem's trampoline, the way golang.org/x/sys calls libc on
// macOS without cgo.
//
//go:cgo_import_dynamic libc_pthread_set_qos_class_self_np pthread_set_qos_class_self_np "/usr/lib/libSystem.B.dylib"
var libc_pthread_set_qos_class_self_np_trampoline_addr uintptr

//go:linkname syscall_syscall syscall.syscall
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno)

func lowerThreadPriority() error {
	r, _, _ := syscall_syscall(libc_pthread_set_qos_class_self_np_trampoline_addr, qosClassUtility, 0, 0)
	if r != 0 {
		return syscall.Errno(r)
	}
	return nil
}

//go:cgo_import_dynamic libc_qos_class_self qos_class_self "/usr/lib/libSystem.B.dylib"
var libc_qos_class_self_trampoline_addr uintptr

// threadQoS is the calling thread's QoS class (qos_class_self), for tests.
func threadQoS() uintptr {
	r, _, _ := syscall_syscall(libc_qos_class_self_trampoline_addr, 0, 0, 0)
	return r
}
