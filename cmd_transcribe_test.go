package main

import (
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
