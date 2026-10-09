package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestClipboardHelperProcess is not a real test: it is spawned as a child
// process by the clipboard tests to act as a fake clipboard utility. It reads
// stdin and either writes it to $CLIPBOARD_HELPER_OUT or fails when
// $CLIPBOARD_HELPER_FAIL is set.
func TestClipboardHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_CLIPBOARD_HELPER") != "1" {
		return
	}
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		os.Exit(2)
	}
	if os.Getenv("CLIPBOARD_HELPER_FAIL") == "1" {
		os.Exit(1)
	}
	if err := os.WriteFile(os.Getenv("CLIPBOARD_HELPER_OUT"), data, 0o644); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

// helperCmd returns a command that runs TestClipboardHelperProcess as a fake
// clipboard utility with the given behaviour.
func helperCmd(t *testing.T, fail bool, out string) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestClipboardHelperProcess")
	cmd.Env = append(os.Environ(), "GO_WANT_CLIPBOARD_HELPER=1")
	if fail {
		cmd.Env = append(cmd.Env, "CLIPBOARD_HELPER_FAIL=1")
	}
	if out != "" {
		cmd.Env = append(cmd.Env, "CLIPBOARD_HELPER_OUT="+out)
	}
	return cmd
}

func TestClipboardCommandsPerOS(t *testing.T) {
	cases := []struct {
		goos string
		want []string // first binary only
		err  bool
	}{
		{"darwin", []string{"pbcopy"}, false},
		{"windows", []string{"clip"}, false},
		{"linux", []string{"wl-copy", "xclip", "xsel"}, false},
		{"plan9", nil, true},
	}
	for _, c := range cases {
		cmds, err := clipboardCommands(c.goos)
		if c.err {
			if err == nil {
				t.Errorf("%s: want error, got %v", c.goos, cmds)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.goos, err)
			continue
		}
		if len(cmds) != len(c.want) {
			t.Errorf("%s: %d commands, want %d", c.goos, len(cmds), len(c.want))
			continue
		}
		for i, w := range c.want {
			if cmds[i][0] != w {
				t.Errorf("%s: cmd[%d]=%q, want %q", c.goos, i, cmds[i][0], w)
			}
		}
	}
}

func TestCopyViaCommandsWritesText(t *testing.T) {
	out := filepath.Join(t.TempDir(), "clip.txt")
	origLook, origCmd := execLookPath, execCommand
	t.Cleanup(func() { execLookPath, execCommand = origLook, origCmd })
	execLookPath = func(name string) (string, error) { return name, nil }
	execCommand = func(name string, args ...string) *exec.Cmd {
		return helperCmd(t, false, out)
	}
	if err := copyViaCommands([][]string{{"fakeclip"}}, "hello world"); err != nil {
		t.Fatalf("copyViaCommands: %v", err)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello world" {
		t.Fatalf("clipboard got %q, want %q", got, "hello world")
	}
}

func TestCopyViaCommandsFallsBackOnRuntimeFailure(t *testing.T) {
	out := filepath.Join(t.TempDir(), "clip.txt")
	origLook, origCmd := execLookPath, execCommand
	t.Cleanup(func() { execLookPath, execCommand = origLook, origCmd })
	var tried []string
	execLookPath = func(name string) (string, error) { return name, nil }
	execCommand = func(name string, args ...string) *exec.Cmd {
		tried = append(tried, name)
		// First utility exists but fails at runtime; second succeeds.
		return helperCmd(t, name == "first", out)
	}
	commands := [][]string{{"first"}, {"second"}}
	if err := copyViaCommands(commands, "fallback text"); err != nil {
		t.Fatalf("copyViaCommands: %v", err)
	}
	if len(tried) != 2 || tried[0] != "first" || tried[1] != "second" {
		t.Fatalf("tried = %v, want [first second]", tried)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "fallback text" {
		t.Fatalf("clipboard got %q, want %q", got, "fallback text")
	}
}

func TestCopyViaCommandsNoUtility(t *testing.T) {
	origLook, origCmd := execLookPath, execCommand
	t.Cleanup(func() { execLookPath, execCommand = origLook, origCmd })
	execLookPath = func(name string) (string, error) { return "", os.ErrNotExist }
	execCommand = func(name string, args ...string) *exec.Cmd {
		t.Fatal("execCommand must not be called when LookPath fails")
		return nil
	}
	err := copyViaCommands([][]string{{"missing1"}, {"missing2"}}, "text")
	if err == nil {
		t.Fatal("want error when no utility is found")
	}
	if !strings.Contains(err.Error(), "no clipboard utility found") {
		t.Fatalf("err = %v", err)
	}
}

func TestCopyViaCommandsAllFailAtRuntime(t *testing.T) {
	origLook, origCmd := execLookPath, execCommand
	t.Cleanup(func() { execLookPath, execCommand = origLook, origCmd })
	execLookPath = func(name string) (string, error) { return name, nil }
	execCommand = func(name string, args ...string) *exec.Cmd {
		return helperCmd(t, true, "")
	}
	err := copyViaCommands([][]string{{"a"}, {"b"}}, "text")
	if err == nil {
		t.Fatal("want error when every utility fails")
	}
	if !strings.Contains(err.Error(), "clipboard copy failed") {
		t.Fatalf("err = %v", err)
	}
}
