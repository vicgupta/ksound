//go:build !unix && !windows

package main

func processIsRunning(int) bool {
	return false
}
