package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultTranscriptPathByFormat(t *testing.T) {
	if got := defaultTranscriptPath("recordings/standup.flac", "text"); got != "recordings/standup.txt" {
		t.Fatalf("text path = %q", got)
	}
	if got := defaultTranscriptPath("recordings/standup.wav", "markdown"); got != "recordings/standup.md" {
		t.Fatalf("markdown path = %q", got)
	}
}

func TestFormatTranscriptMarkdownIncludesMetadataAndText(t *testing.T) {
	created := time.Date(2026, 10, 7, 9, 5, 0, 0, time.FixedZone("EDT", -4*60*60))
	got, err := formatTranscript("recordings/product-interview.flac", "We should test this with users.", "markdown", created)
	if err != nil {
		t.Fatalf("formatTranscript: %v", err)
	}
	for _, want := range []string{
		"# product-interview",
		"Source: `product-interview.flac`",
		"Transcribed: 2026-10-07 09:05 EDT",
		"We should test this with users.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("formatted transcript missing %q:\n%s", want, got)
		}
	}
}

func TestFormatTranscriptTextPreservesPlainText(t *testing.T) {
	got, err := formatTranscript("memo.wav", "hello\nworld", "text", time.Time{})
	if err != nil {
		t.Fatalf("formatTranscript: %v", err)
	}
	if got != "hello\nworld\n" {
		t.Fatalf("plain transcript = %q", got)
	}
}

func TestFormatTranscriptRejectsUnknownFormat(t *testing.T) {
	if _, err := formatTranscript("memo.wav", "hello", "html", time.Time{}); err == nil {
		t.Fatal("expected unsupported format error")
	}
}

func TestExplicitTranscriptPathIsNotChanged(t *testing.T) {
	got := transcriptOutputPath("memo.wav", "notes/custom.out", "markdown")
	if filepath.Clean(got) != filepath.Clean("notes/custom.out") {
		t.Fatalf("explicit path = %q", got)
	}
}

func TestValidateGain(t *testing.T) {
	for _, bad := range []float64{0, -1, 21, 100} {
		if err := validateGain(bad); err == nil {
			t.Fatalf("gain %v should be rejected", bad)
		}
	}
	for _, good := range []float64{0.5, 1.0, 2.0, 20} {
		if err := validateGain(good); err != nil {
			t.Fatalf("gain %v: %v", good, err)
		}
	}
}

func TestDefaultThreadsPositive(t *testing.T) {
	if got := defaultThreads(); got < 1 || got > 8 {
		t.Fatalf("defaultThreads = %d, want 1..8", got)
	}
}

func TestVadValidate(t *testing.T) {
	if err := defaultVadOptions().validate(); err != nil {
		t.Fatalf("defaults: %v", err)
	}
	bad := defaultVadOptions()
	bad.threshold = 0
	if err := bad.validate(); err == nil {
		t.Fatal("threshold 0 should be rejected")
	}
	bad = defaultVadOptions()
	bad.maxSpeech = 0
	if err := bad.validate(); err == nil {
		t.Fatal("max-speech 0 should be rejected")
	}
}

func TestWriteTranscriptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := writeTranscriptFile(path, "hello\n"); err != nil {
		t.Fatalf("writeTranscriptFile: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(got) != "hello\n" {
		t.Fatalf("content = %q", got)
	}
}
