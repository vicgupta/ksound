package main

import (
	"os"

	"golang.org/x/term"
)

// isTerminal reports whether file is attached to a terminal.
func isTerminal(file *os.File) bool {
	return term.IsTerminal(int(file.Fd()))
}
