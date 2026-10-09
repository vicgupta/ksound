package main

import (
	"fmt"
	"os"
)

// infof prints user-facing progress to stderr so stdout stays clean for
// piped transcript output.
func infof(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
}

// warnf prints a non-fatal warning to stderr.
func warnf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "warning: "+format+"\n", args...)
}
