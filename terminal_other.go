//go:build !linux && !darwin && !windows

package main

import "os"

func isTerminal(*os.File) bool {
	return false
}

func processIsRunning(int) bool {
	return false
}
