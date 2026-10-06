package main

import (
	"strings"
	"testing"
)

func TestRenderBox(t *testing.T) {
	for _, st := range []boxStyle{unicodeBoxStyle, asciiBoxStyle} {
		out := renderBoxWithStyle(st, "Transcription", "hello world, this is a test of the boxing code", 30)
		lines := strings.Split(out, "\n")
		if len(lines) < 4 {
			t.Fatalf("expected >= 4 lines, got %d:\n%s", len(lines), out)
		}
		width := len([]rune(lines[0]))
		for i, l := range lines {
			if got := len([]rune(l)); got != width {
				t.Fatalf("line %d width %d, want %d: %q", i, got, width, l)
			}
		}
		if !strings.HasPrefix(lines[0], st.tl) || !strings.HasSuffix(lines[0], st.tr) {
			t.Fatalf("bad top border: %q", lines[0])
		}
		if !strings.HasPrefix(lines[len(lines)-1], st.bl) || !strings.HasSuffix(lines[len(lines)-1], st.br) {
			t.Fatalf("bad bottom border: %q", lines[len(lines)-1])
		}
		if !strings.Contains(out, "hello") {
			t.Fatalf("text missing:\n%s", out)
		}
	}
}

func TestWrapWords(t *testing.T) {
	lines := wrapWords("one two three four", 7)
	for _, l := range lines {
		if len([]rune(l)) > 7 {
			t.Fatalf("line too long: %q", l)
		}
	}
	if strings.Join(lines, " ") != "one two three four" {
		t.Fatalf("words lost: %q", lines)
	}
	// long word hard-split
	lines = wrapWords("abcdefghij", 4)
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %q", lines)
	}
}
