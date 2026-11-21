//go:build !linux && !darwin && !windows

package engine

import "errors"

const lowPrioritySupported = false

func lowerThreadPriority() error {
	return errors.New("engine: lowering thread priority isn't supported on this OS")
}
