//go:build windows

package main

import (
	"errors"
	"os"
	"syscall"
)

func isTerminal(file *os.File) bool {
	var mode uint32
	return syscall.GetConsoleMode(syscall.Handle(file.Fd()), &mode) == nil
}

const (
	processQueryLimitedInformation = 0x1000
	windowsErrorInvalidParameter   = syscall.Errno(87)
	windowsProcessStillActive      = 259
)

func processIsRunning(pid int) bool {
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
