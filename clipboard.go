package main

import (
	"errors"
	"fmt"
	"io"
	"os/exec"
	"runtime"
)

// Indirections so tests can substitute fake clipboard utilities.
var (
	execLookPath = exec.LookPath
	execCommand  = exec.Command
)

// clipboardCommands returns the clipboard utilities to try for goos, in
// preference order.
func clipboardCommands(goos string) ([][]string, error) {
	switch goos {
	case "darwin":
		return [][]string{{"pbcopy"}}, nil
	case "windows":
		return [][]string{{"clip"}}, nil
	case "linux":
		return [][]string{{"wl-copy"}, {"xclip", "-selection", "clipboard"}, {"xsel", "--clipboard", "--input"}}, nil
	default:
		return nil, fmt.Errorf("clipboard copying is not supported on %s", goos)
	}
}

func copyToClipboard(text string) error {
	commands, err := clipboardCommands(runtime.GOOS)
	if err != nil {
		return err
	}
	return copyViaCommands(commands, text)
}

// copyViaCommands pipes text into the first available utility, falling back
// to the next one when a utility is missing or fails.
func copyViaCommands(commands [][]string, text string) error {
	var missingErr, waitErr error
	for _, args := range commands {
		path, err := execLookPath(args[0])
		if err != nil {
			missingErr = err
			continue
		}
		cmd := execCommand(path, args[1:]...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return err
		}
		if err := cmd.Start(); err != nil {
			missingErr = err
			continue
		}
		_, writeErr := io.WriteString(stdin, text)
		closeErr := stdin.Close()
		if writeErr != nil {
			return writeErr
		}
		if closeErr != nil {
			return closeErr
		}
		// Fall through to the next utility when this one fails at runtime
		// (e.g. xclip installed but no X display available).
		if err := cmd.Wait(); err != nil {
			waitErr = err
			continue
		}
		return nil
	}
	if waitErr != nil {
		return fmt.Errorf("clipboard copy failed: %w", waitErr)
	}
	if missingErr != nil {
		return fmt.Errorf("no clipboard utility found: %w", missingErr)
	}
	return errors.New("no clipboard utility available")
}
