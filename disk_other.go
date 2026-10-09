//go:build !darwin && !linux && !windows

package main

import "errors"

// freeDiskBytes is unsupported on this platform; callers skip the check.
func freeDiskBytes(string) (int64, error) {
	return 0, errors.New("disk space probe not supported on this platform")
}
