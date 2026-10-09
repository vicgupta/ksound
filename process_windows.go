//go:build windows

package main

import (
	"errors"
	"syscall"
)

const (
	processQueryLimitedInformation = 0x1000
	windowsErrorInvalidParameter   = syscall.Errno(87)
	windowsProcessStillActive      = 259
)

// processIsRunning reports whether pid belongs to a live process on Windows.
func processIsRunning(pid int) bool {
	if pid <= 0 {
		return false
	}
	handle, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		return !errors.Is(err, windowsErrorInvalidParameter)
	}
	defer syscall.CloseHandle(handle)

	var exitCode uint32
	if err := syscall.GetExitCodeProcess(handle, &exitCode); err != nil {
		return true
	}
	return exitCode == windowsProcessStillActive
}
