package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// boxWidth returns the box width: $COLUMNS when set, else 76, clamped to 40..100.
func boxWidth() int {
	w := 76
	if c := strings.TrimSpace(os.Getenv("COLUMNS")); c != "" {
		if n, err := strconv.Atoi(c); err == nil {
			w = n
		}
	}
	if w < 40 {
		w = 40
	}
	if w > 100 {
		w = 100
	}
	return w
}

// wrapWords splits text into lines of at most width runes on word boundaries.
// Long words are hard-split.
func wrapWords(text string, width int) []string {
	var lines []string
	for _, para := range strings.Split(text, "\n") {
		if strings.TrimSpace(para) == "" {
			lines = append(lines, "")
			continue
		}
		var cur []rune
		curLen := 0
		flush := func() {
			if curLen > 0 {
				lines = append(lines, string(cur))
				cur, curLen = nil, 0
			}
		}
		for _, word := range strings.Fields(para) {
			w := []rune(word)
			for len(w) > width {
				flush()
				lines = append(lines, string(w[:width]))
				w = w[width:]
			}
			extra := len(w)
			if curLen > 0 {
				extra++ // space
			}
			if curLen+extra > width {
				flush()
			}
			if curLen > 0 {
				cur = append(cur, ' ')
				curLen++
			}
			cur = append(cur, w...)
			curLen += len(w)
		}
		flush()
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

// boxStyle holds the border characters. Windows uses ASCII since legacy
// consoles (cmd.exe) often lack unicode line-drawing glyphs.
type boxStyle struct {
	tl, tr, bl, br, h, v, titleSep string
}

var (
	unicodeBoxStyle = boxStyle{"┌", "┐", "└", "┘", "─", "│", "─"}
	asciiBoxStyle   = boxStyle{"+", "+", "+", "+", "-", "|", "-"}
)

func currentBoxStyle() boxStyle {
	if runtime.GOOS == "windows" {
		return asciiBoxStyle
	}
	return unicodeBoxStyle
}

// renderBox wraps text in a titled box. Width is total columns.
func renderBox(title, text string, width int) string {
	return renderBoxWithStyle(currentBoxStyle(), title, text, width)
}

func renderBoxWithStyle(st boxStyle, title, text string, width int) string {
	inner := width - 4 // "│ " + " │"
	if inner < 10 {
		inner = 10
		width = inner + 4
	}
	lines := wrapWords(strings.TrimSpace(text), inner)

	var b strings.Builder
	// top: ┌─ title ───┐
	top := st.tl + st.titleSep + " " + title + " "
	if runeLen(top)+1 > width {
		top = st.tl + st.titleSep + " " + string([]rune(title)[:width-5]) + " "
	}
	b.WriteString(top + strings.Repeat(st.h, width-runeLen(top)-1) + st.tr + "\n")
	for _, l := range lines {
		r := []rune(l)
		b.WriteString(st.v + " " + string(r) + strings.Repeat(" ", inner-len(r)) + " " + st.v + "\n")
	}
	b.WriteString(st.bl + strings.Repeat(st.h, width-2) + st.br)
	return b.String()
}

func runeLen(s string) int { return len([]rune(s)) }

// printBoxed prints text to stdout inside a titled box.
func printBoxed(title, text string) {
	fmt.Println(renderBox(title, text, boxWidth()))
}
