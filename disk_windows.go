//go:build windows

package main

import "golang.org/x/sys/windows"

// freeDiskBytes returns the writable space on the volume containing path.
func freeDiskBytes(path string) (int64, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &free, nil, nil); err != nil {
		return 0, err
	}
	return int64(free), nil
}
