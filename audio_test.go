package main

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

func sineSamples(n int) []int16 {
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(math.Sin(float64(i)*2*math.Pi*440/16000) * 10000)
	}
	return out
}

func TestFLACRoundTrip(t *testing.T) {
	src := sineSamples(16000)
	path := filepath.Join(t.TempDir(), "roundtrip.flac")
	if err := SaveFLAC(path, src); err != nil {
		t.Fatalf("SaveFLAC: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Size() == 0 {
		t.Fatal("empty flac file")
	}
	got, rate, err := DecodeFLACFile(path)
	if err != nil {
		t.Fatalf("DecodeFLACFile: %v", err)
	}
	if rate != targetSampleRate {
		t.Fatalf("rate = %d, want %d", rate, targetSampleRate)
	}
	if len(got) != len(src) {
		t.Fatalf("samples = %d, want %d", len(got), len(src))
	}
	want := Int16ToFloat32(src)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("sample %d differs: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestWAVRoundTrip(t *testing.T) {
	src := sineSamples(8000)
	path := filepath.Join(t.TempDir(), "roundtrip.wav")
	if err := SaveWAV(path, src); err != nil {
		t.Fatalf("SaveWAV: %v", err)
	}
	got, rate, err := DecodeWAVFile(path)
	if err != nil {
		t.Fatalf("DecodeWAVFile: %v", err)
	}
	if rate != targetSampleRate {
		t.Fatalf("rate = %d, want %d", rate, targetSampleRate)
	}
	if len(got) != len(src) {
		t.Fatalf("samples = %d, want %d", len(got), len(src))
	}
}

func TestNormalizeAudioFormat(t *testing.T) {
	if f, _ := NormalizeAudioFormat(""); f != FormatFLAC {
		t.Fatalf("empty -> %q, want flac", f)
	}
	if f, _ := NormalizeAudioFormat("wav"); f != FormatWAV {
		t.Fatalf("wav -> %q", f)
	}
	if _, err := NormalizeAudioFormat("mp3"); err == nil {
		t.Fatal("mp3 should be rejected")
	}
}

func TestApplyGain(t *testing.T) {
	s := []int16{1000, -1000, 10000}
	if clipped := ApplyGain(s, 2.0); clipped != 0 {
		t.Fatalf("clipped = %d, want 0", clipped)
	}
	if s[0] != 2000 || s[1] != -2000 || s[2] != 20000 {
		t.Fatalf("scaled = %v", s)
	}
	if clipped := ApplyGain(s, 2.0); clipped != 1 {
		t.Fatalf("clipped = %d, want 1 (40000 clamps)", clipped)
	}
	if s[2] != math.MaxInt16 {
		t.Fatalf("clamped = %d, want %d", s[2], math.MaxInt16)
	}
	if got := ApplyGain([]int16{1}, 1.0); got != 0 {
		t.Fatalf("gain 1.0 touched samples")
	}
}

func TestMeterHelpers(t *testing.T) {
	if got := dbFS(0); !math.IsInf(got, -1) {
		t.Fatalf("dbFS(0) = %v, want -inf", got)
	}
	if got := dbFS(32768); math.Abs(got) > 1e-9 {
		t.Fatalf("dbFS(32768) = %v, want 0", got)
	}
	if got := dbFS(16384); math.Abs(got+6.0206) > 1e-3 {
		t.Fatalf("dbFS(16384) = %v, want -6.02", got)
	}
	if bar := meterBar(0, 10); bar != "##########" {
		t.Fatalf("full bar = %q", bar)
	}
	if bar := meterBar(math.Inf(-1), 10); bar != "----------" {
		t.Fatalf("silent bar = %q", bar)
	}
	if bar := meterBar(-30, 10); bar != "#####-----" {
		t.Fatalf("half bar = %q", bar)
	}
}

func TestFormatElapsed(t *testing.T) {
	cases := map[string]string{
		"0s":      "0.0s",
		"12.34s":  "12.3s",
		"59.9s":   "59.9s",
		"65s":     "1:05",
		"3599s":   "59:59",
		"3723s":   "1:02:03",
		"7325.7s": "2:02:05",
	}
	for in, want := range cases {
		d, _ := time.ParseDuration(in)
		if got := formatElapsed(d); got != want {
			t.Fatalf("formatElapsed(%s) = %q, want %q", in, got, want)
		}
	}
}

func TestResampleLinear(t *testing.T) {
	// same rate: unchanged
	src := []float32{0.1, -0.2, 0.3}
	if got := ResampleLinear(src, 16000, 16000); len(got) != 3 || got[0] != 0.1 {
		t.Fatalf("identity resample = %v", got)
	}
	// constant signal stays constant through rate change
	constant := make([]float32, 1600)
	for i := range constant {
		constant[i] = 0.5
	}
	down := ResampleLinear(constant, 16000, 8000)
	if len(down) != 800 {
		t.Fatalf("16000->8000 len = %d, want 800", len(down))
	}
	for i, v := range down {
		if math.Abs(float64(v-0.5)) > 1e-6 {
			t.Fatalf("down[%d] = %v, want 0.5", i, v)
		}
	}
	up := ResampleLinear(constant, 8000, 16000)
	if len(up) != 3200 {
		t.Fatalf("8000->16000 len = %d, want 3200", len(up))
	}
	// ramp midpoint lands halfway
	ramp := []float32{0, 1}
	mid := ResampleLinear(ramp, 2, 4)
	if len(mid) != 4 || math.Abs(float64(mid[1]-0.5)) > 1e-6 {
		t.Fatalf("ramp resample = %v", mid)
	}
	if len(ResampleLinear(nil, 16000, 8000)) != 0 {
		t.Fatal("nil resample should be empty")
	}
}

func TestEnsureAudioPath(t *testing.T) {
	p, f, err := EnsureAudioPath("", FormatFLAC, false)
	if err != nil {
		t.Fatalf("default: %v", err)
	}
	if filepath.Ext(p) != ".flac" || f != FormatFLAC {
		t.Fatalf("default = %q (%s)", p, f)
	}

	p, f, err = EnsureAudioPath("out/note", FormatWAV, false)
	if err != nil {
		t.Fatalf("ext append: %v", err)
	}
	if filepath.Ext(p) != ".wav" || f != FormatWAV {
		t.Fatalf("appended = %q (%s)", p, f)
	}

	// An explicit extension wins over the default format.
	dir := t.TempDir()
	wavPath := filepath.Join(dir, "x.wav")
	p, f, err = EnsureAudioPath(wavPath, FormatFLAC, false)
	if err != nil {
		t.Fatalf("ext wins: %v", err)
	}
	if p != wavPath || f != FormatWAV {
		t.Fatalf("x.wav -> %q (%s), want wav", p, f)
	}
	if err := SaveAudio(p, f, sineSamples(1000)); err != nil {
		t.Fatalf("SaveAudio: %v", err)
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(raw) < 4 || string(raw[:4]) != "RIFF" {
		t.Fatalf("x.wav bytes not RIFF: % x", raw)
	}

	// An explicit --format that conflicts with the extension is an error.
	if _, _, err := EnsureAudioPath(wavPath, FormatFLAC, true); err == nil {
		t.Fatal("expected format/extension conflict error")
	}

	// A non-audio extension is rejected.
	if _, _, err := EnsureAudioPath(filepath.Join(dir, "x.mp3"), FormatFLAC, false); err == nil {
		t.Fatal("expected unsupported extension error")
	}
}

func TestRecoverPartialRecordings(t *testing.T) {
	dir := t.TempDir()
	src := sineSamples(16000)
	raw := make([]byte, len(src)*2)
	for i, s := range src {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(s))
	}
	partial := filepath.Join(dir, "2026-10-08_10-00-00.flac.partial.pcm")
	if err := os.WriteFile(partial, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := RecoverPartialRecordings(dir)
	if err != nil {
		t.Fatalf("RecoverPartialRecordings: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("recovered %d files, want 1", len(got))
	}
	if base := filepath.Base(got[0]); base != "2026-10-08_10-00-00.recovered.flac" {
		t.Fatalf("recovered path = %q", base)
	}
	if _, err := os.Stat(partial); !os.IsNotExist(err) {
		t.Fatalf("partial should be removed, err=%v", err)
	}
	samples, rate, err := DecodeFLACFile(got[0])
	if err != nil {
		t.Fatalf("DecodeFLACFile: %v", err)
	}
	if rate != targetSampleRate || len(samples) != len(src) {
		t.Fatalf("recovered audio rate=%d len=%d, want %d/%d", rate, len(samples), targetSampleRate, len(src))
	}
}

func TestRecoverPartialRecordingsEmpty(t *testing.T) {
	dir := t.TempDir()
	partial := filepath.Join(dir, "empty.wav.partial.pcm")
	if err := os.WriteFile(partial, []byte{1}, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := RecoverPartialRecordings(dir)
	if err != nil {
		t.Fatalf("RecoverPartialRecordings: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("recovered %d files, want 0", len(got))
	}
	if _, err := os.Stat(partial); !os.IsNotExist(err) {
		t.Fatalf("degenerate partial should be removed, err=%v", err)
	}
}

func TestWaitForStopReasons(t *testing.T) {
	enter := make(chan struct{}, 1)
	enter <- struct{}{}
	if got := waitForStop(enter, make(chan os.Signal), nil); got != stopEnter {
		t.Fatalf("enter -> %v, want stopEnter", got)
	}

	sig := make(chan os.Signal, 1)
	sig <- syscall.SIGINT
	if got := waitForStop(make(chan struct{}), sig, nil); got != stopSignal {
		t.Fatalf("signal -> %v, want stopSignal", got)
	}

	if got := waitForStop(make(chan struct{}), make(chan os.Signal), time.After(10*time.Millisecond)); got != stopDuration {
		t.Fatalf("timeout -> %v, want stopDuration", got)
	}

	errCh := make(chan error, 1)
	errCh <- io.ErrShortWrite
	reason, gotErr := waitForStopWithErrors(make(chan struct{}), make(chan os.Signal), nil, errCh)
	if reason != stopPartialError || gotErr != io.ErrShortWrite {
		t.Fatalf("partial error -> (%v, %v), want (%v, %v)", reason, gotErr, stopPartialError, io.ErrShortWrite)
	}
}

func TestStdinIsTerminalRejectsDevNull(t *testing.T) {
	original := os.Stdin
	stdin, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	os.Stdin = stdin
	defer func() {
		os.Stdin = original
		_ = stdin.Close()
	}()

	if stdinIsTerminal() {
		t.Fatal("/dev/null must not be treated as a terminal")
	}
}

type shortWriterAt struct {
	data []byte
}

func (w *shortWriterAt) WriteAt(p []byte, off int64) (int, error) {
	if off < 0 || int(off) > len(w.data) {
		return 0, io.ErrShortWrite
	}
	if int(off)+len(p) > len(w.data) {
		w.data = append(w.data, make([]byte, int(off)+len(p)-len(w.data))...)
	}
	n := len(p)
	if n > 3 {
		n = 3
	}
	copy(w.data[int(off):], p[:n])
	return n, nil
}

func TestWriteFullAtRetriesShortWrites(t *testing.T) {
	want := []byte("partial-pcm")
	writer := &shortWriterAt{}
	if err := writeFullAt(writer, want, 0); err != nil {
		t.Fatalf("writeFullAt: %v", err)
	}
	if string(writer.data) != string(want) {
		t.Fatalf("written = %q, want %q", writer.data, want)
	}
}

func TestWritePartialLoopReportsOpenError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "missing", "partial.pcm")
	raw := []byte{1, 2}
	stop := make(chan struct{})
	close(stop)
	errCh := make(chan error, 1)
	writePartialLoopWithErrors(&sync.Mutex{}, &raw, path, stop, errCh)
	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("expected partial writer error")
		}
	default:
		t.Fatal("partial writer error was not reported")
	}
}
