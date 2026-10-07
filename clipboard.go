package main

import (
	"fmt"
	"io"
	"os/exec"
	"runtime"
)

func copyToClipboard(text string) error {
	var commands [][]string
	switch runtime.GOOS {
	case "darwin":
		commands = [][]string{{"pbcopy"}}
	case "windows":
		commands = [][]string{{"clip"}}
	case "linux":
		commands = [][]string{{"wl-copy"}, {"xclip", "-selection", "clipboard"}, {"xsel", "--clipboard", "--input"}}
	default:
		return fmt.Errorf("clipboard copying is not supported on %s", runtime.GOOS)
	}

	var lastErr error
	for _, args := range commands {
		path, err := exec.LookPath(args[0])
		if err != nil {
			lastErr = err
			continue
		}
		cmd := exec.Command(path, args[1:]...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return err
		}
		if err := cmd.Start(); err != nil {
			lastErr = err
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
		return cmd.Wait()
	}
	return fmt.Errorf("no clipboard utility found: %w", lastErr)
}
