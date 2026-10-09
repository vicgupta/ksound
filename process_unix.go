//go:build unix

package main

import (
	"errors"
	"syscall"
)

// processIsRunning reports whether pid belongs to a live process.
// ESRCH means no such process; EPERM means it exists but we cannot signal it.
func processIsRunning(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
