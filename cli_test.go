package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// runCmd runs a command with args, silencing cobra output and capturing it.
func runCmd(t *testing.T, cmd *cobra.Command, args ...string) (string, error) {
	t.Helper()
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true
	buf := &bytes.Buffer{}
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	if args == nil {
		// cobra falls back to os.Args[1:] when args is nil; keep it empty.
		args = []string{}
	}
	cmd.SetArgs(args)
	err := cmd.Execute()
	return buf.String(), err
}

func TestTranscribeCmdRejectsBadFlagsBeforeModelDownload(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"no args", nil, "accepts 1 arg(s)"},
		{"bad format", []string{"a.flac", "--transcript-format", "html"}, "transcript format"},
		{"bad vad threshold", []string{"a.flac", "--vad-threshold", "5"}, "vad-threshold"},
		{"negative threads", []string{"a.flac", "--threads", "-1"}, "threads"},
		{"encoder without tokens", []string{"a.flac", "--encoder", "e.onnx"}, "must be given together"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := runCmd(t, newTranscribeCmd(), tc.args...)
			if err == nil {
				t.Fatalf("expected error, got none (out: %s)", out)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestRecordCmdRejectsBadGainBeforeRecording(t *testing.T) {
	out, err := runCmd(t, newRecordCmd(), "--gain", "0")
	if err == nil {
		t.Fatalf("expected error, got none (out: %s)", out)
	}
	if !strings.Contains(err.Error(), "gain") {
		t.Fatalf("error %q does not mention gain", err)
	}
}

func TestRecordAndTranscribeFastFail(t *testing.T) {
	// Flag errors must surface before model download / mic access.
	for _, args := range [][]string{
		{"--gain", "0"},
		{"--vad-threshold", "-1"},
		{"--transcript-format", "html"},
		{"--threads", "-3"},
	} {
		out, err := runCmd(t, newRecordAndTranscribeCmd(), args...)
		if err == nil {
			t.Fatalf("args %v: expected error, got none (out: %s)", args, out)
		}
	}
}

func TestRootVersionFlag(t *testing.T) {
	out, err := runCmd(t, rootCmd, "--version")
	if err != nil {
		t.Fatalf("--version: %v", err)
	}
	if !strings.Contains(out, "version") {
		t.Fatalf("--version output = %q", out)
	}
}

func TestDevicesCmdHasJSONFlag(t *testing.T) {
	cmd := newDevicesCmd()
	if cmd.Flags().Lookup("json") == nil {
		t.Fatal("--json flag missing")
	}
}

func TestCombinedFlagsRegisteredAndAliasesGone(t *testing.T) {
	cmd := newRecordAndTranscribeCmd()
	for _, name := range []string{
		"audio-out", "output", "transcript-format", "device", "format",
		"gain", "duration", "threads", "keep-audio", "encoder", "decoder",
		"joiner", "tokens", "vad-threshold",
	} {
		if cmd.Flags().Lookup(name) == nil {
			t.Errorf("flag --%s missing", name)
		}
	}
	for _, gone := range []string{"wav-out", "keep-wav"} {
		if cmd.Flags().Lookup(gone) != nil {
			t.Errorf("deprecated flag --%s should be removed", gone)
		}
	}
}

func TestTranscribeCmdRequiresSingleFileArg(t *testing.T) {
	_, err := runCmd(t, newTranscribeCmd(), "a.flac", "b.flac")
	if err == nil {
		t.Fatal("two args should be rejected")
	}
}
