package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeRec implements recognizer with canned output and call accounting.
type fakeRec struct {
	text  string
	err   error
	calls int
	last  int // len of last Decode input
	max   int // longest Decode input seen
}

func (f *fakeRec) Decode(samples []float32, sampleRate int) (string, error) {
	f.calls++
	f.last = len(samples)
	if len(samples) > f.max {
		f.max = len(samples)
	}
	return f.text, f.err
}
func (f *fakeRec) Close() {}

// fakeVAD emits fixed-size segments as audio is fed, emulating Silero
// segmentation closely enough for pipeline tests.
type fakeVAD struct {
	segLen int
	fed    int
	next   int
	segs   []vadSegment
	closed bool
}

func newFakeVAD(segLen int) *fakeVAD { return &fakeVAD{segLen: segLen} }

func (f *fakeVAD) AcceptWaveform(samples []float32) {
	f.fed += len(samples)
	for f.fed >= f.segLen {
		f.segs = append(f.segs, vadSegment{Start: f.next, N: f.segLen})
		f.next += f.segLen
		f.fed -= f.segLen
	}
}

func (f *fakeVAD) Flush() {
	if f.fed == 0 {
		return
	}
	f.segs = append(f.segs, vadSegment{Start: f.next, N: f.fed})
	f.next += f.fed
	f.fed = 0
}

func (f *fakeVAD) IsEmpty() bool { return len(f.segs) == 0 }
func (f *fakeVAD) Front() vadSegment {
	return f.segs[0]
}
func (f *fakeVAD) Pop()   { f.segs = f.segs[1:] }
func (f *fakeVAD) Close() { f.closed = true }

// testPipeline builds a pipeline with fake deps; audioFile is created (empty)
// so os.Stat passes while loadAudio is faked.
type testHarness struct {
	p          *pipeline
	rec        *fakeRec
	vad        *fakeVAD
	vadCreated int
	copied     string
	copyErr    error
	samples    []float32
	rate       int
	audioPath  string
}

func newHarness(t *testing.T, text string) *testHarness {
	t.Helper()
	dir := t.TempDir()
	audioPath := filepath.Join(dir, "in.flac")
	if err := os.WriteFile(audioPath, []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	h := &testHarness{
		rec:       &fakeRec{text: text},
		vad:       newFakeVAD(32000),
		samples:   make([]float32, 16000), // 1s at 16kHz
		rate:      16000,
		audioPath: audioPath,
	}
	h.p = &pipeline{
		vad:     defaultVadOptions(),
		threads: 2,
		newRec: func(m ModelFiles, threads int) (recognizer, error) {
			return h.rec, nil
		},
		newVAD: func(opts vadOptions) (vad, error) {
			h.vadCreated++
			return h.vad, nil
		},
		loadAudio: func(path string) ([]float32, int, error) {
			return h.samples, h.rate, nil
		},
		copyText: func(text string) error {
			h.copied = text
			return h.copyErr
		},
	}
	return h
}

func TestPipelineShortAudioSingleDecode(t *testing.T) {
	h := newHarness(t, "hello world")
	out := filepath.Join(t.TempDir(), "out.txt")

	got, err := h.p.run(h.audioPath, out, "text", ModelFiles{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got != out {
		t.Fatalf("out = %q, want %q", got, out)
	}
	if h.rec.calls != 1 {
		t.Fatalf("Decode calls = %d, want 1", h.rec.calls)
	}
	if h.rec.last != len(h.samples) {
		t.Fatalf("Decode got %d samples, want %d", h.rec.last, len(h.samples))
	}
	if h.vadCreated != 0 {
		t.Fatalf("VAD created %d times for short audio, want 0", h.vadCreated)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "hello world\n" {
		t.Fatalf("file = %q", body)
	}
	if h.copied != "hello world" {
		t.Fatalf("clipboard = %q", h.copied)
	}
}

func TestPipelineMarkdownOutput(t *testing.T) {
	h := newHarness(t, "note")
	out := filepath.Join(t.TempDir(), "notes.md")

	got, err := h.p.run(h.audioPath, out, "markdown", ModelFiles{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	body, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "# in\n") || !strings.Contains(string(body), "note") {
		t.Fatalf("markdown body = %q", body)
	}
}

func TestPipelineEmptyResultSkipsClipboard(t *testing.T) {
	h := newHarness(t, "")
	out := filepath.Join(t.TempDir(), "out.txt")

	got, err := h.p.run(h.audioPath, out, "text", ModelFiles{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if h.copied != "" {
		t.Fatalf("clipboard = %q, want untouched", h.copied)
	}
	if _, err := os.Stat(got); err != nil {
		t.Fatalf("empty result must still write transcript: %v", err)
	}
}

func TestPipelineClipboardFailureIsNonFatal(t *testing.T) {
	h := newHarness(t, "text")
	h.copyErr = errors.New("no clipboard utility")
	out := filepath.Join(t.TempDir(), "out.txt")

	if _, err := h.p.run(h.audioPath, out, "text", ModelFiles{}); err != nil {
		t.Fatalf("clipboard failure must not fail transcription: %v", err)
	}
}

func TestPipelineLongAudioSegmentsThroughVAD(t *testing.T) {
	h := newHarness(t, "seg")
	// 301s at 16kHz: over the single-shot cap.
	h.samples = make([]float32, (maxSingleShotSeconds+1)*16000)
	out := filepath.Join(t.TempDir(), "out.txt")

	got, err := h.p.run(h.audioPath, out, "text", ModelFiles{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if h.vadCreated != 1 {
		t.Fatalf("VAD created %d times, want 1", h.vadCreated)
	}
	if !h.vad.closed {
		t.Fatal("VAD was not closed")
	}
	if h.rec.calls < 2 {
		t.Fatalf("Decode calls = %d, want segmented (>1)", h.rec.calls)
	}
	body, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(body), "seg") < 2 {
		t.Fatalf("transcript should join segments: %q", body)
	}
}

func TestPipelineLongAudioResamplesBeforeVAD(t *testing.T) {
	h := newHarness(t, "seg")
	h.samples = make([]float32, 8000*(maxSingleShotSeconds+1)) // 301s at 8kHz
	h.rate = 8000
	out := filepath.Join(t.TempDir(), "out.txt")

	if _, err := h.p.run(h.audioPath, out, "text", ModelFiles{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if h.rec.calls < 2 {
		t.Fatalf("Decode calls = %d, want segmented", h.rec.calls)
	}
	// VAD works on the 16k timeline; segments are mapped back to 8kHz, so
	// each decoder input is at most half of an original-rate segment and the
	// full buffer is never handed over at once.
	if h.rec.max >= len(h.samples) {
		t.Fatalf("max Decode input %d should be < full %d", h.rec.max, len(h.samples))
	}
	if h.rec.max > 32000 {
		t.Fatalf("max Decode input %d exceeds one VAD segment at 8kHz (16000) + slack", h.rec.max)
	}
}

func TestPipelineValidationErrors(t *testing.T) {
	h := newHarness(t, "x")

	t.Run("bad format", func(t *testing.T) {
		if _, err := h.p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "html", ModelFiles{}); err == nil {
			t.Fatal("expected format error")
		}
	})
	t.Run("bad vad", func(t *testing.T) {
		p := *h.p
		p.vad.threshold = 0
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected vad error")
		}
	})
	t.Run("missing input", func(t *testing.T) {
		if _, err := h.p.run(filepath.Join(t.TempDir(), "nope.flac"), filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected input file error")
		}
	})
	t.Run("empty audio", func(t *testing.T) {
		p := *h.p
		p.loadAudio = func(string) ([]float32, int, error) { return nil, 16000, nil }
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected empty audio error")
		}
	})
	t.Run("recognizer error", func(t *testing.T) {
		p := *h.p
		p.newRec = func(ModelFiles, int) (recognizer, error) { return nil, errors.New("boom") }
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected recognizer error")
		}
	})
	t.Run("decode error", func(t *testing.T) {
		p := *h.p
		p.newRec = func(ModelFiles, int) (recognizer, error) {
			return &fakeRec{err: errors.New("decode failed")}, nil
		}
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected decode error")
		}
	})
	t.Run("vad error", func(t *testing.T) {
		h.samples = make([]float32, (maxSingleShotSeconds+1)*16000)
		p := *h.p
		p.newVAD = func(vadOptions) (vad, error) { return nil, errors.New("vad boom") }
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected vad error")
		}
	})
	t.Run("load error", func(t *testing.T) {
		p := *h.p
		p.loadAudio = func(string) ([]float32, int, error) { return nil, 0, errors.New("decode failed") }
		if _, err := p.run(h.audioPath, filepath.Join(t.TempDir(), "o.txt"), "text", ModelFiles{}); err == nil {
			t.Fatal("expected load error")
		}
	})
}

func TestPipelineThreadsAutoWhenZero(t *testing.T) {
	h := newHarness(t, "x")
	h.p.threads = 0
	out := filepath.Join(t.TempDir(), "out.txt")
	if _, err := h.p.run(h.audioPath, out, "text", ModelFiles{}); err != nil {
		t.Fatalf("run: %v", err)
	}
}
