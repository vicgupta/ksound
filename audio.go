package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	flacpcm "github.com/tphakala/go-flac/pcm"
)

const targetSampleRate = 16000

// maxRecordingBytes caps in-memory capture (~6h at 16kHz s16 mono). Longer
// sessions should use --duration chunks: the periodic .partial.pcm file
// already provides crash safety, but the final samples slice still needs RAM.
const maxRecordingBytes = 700 * 1024 * 1024

// Capture stall detection: no new samples for captureStallAfter aborts the
// recording, or captureStopGrace after the device layer reports a stop
// (malgo fires its Stop callback on unplug/hot-plug).
const (
	captureStallAfter = 10 * time.Second
	captureStopGrace  = 2 * time.Second
	captureCheckEvery = time.Second
)

func timestampNow() string { return time.Now().Format("2006-01-02_15-04-05") }

// Audio formats supported for recording.
const (
	FormatFLAC = "flac"
	FormatWAV  = "wav"
)

// NormalizeAudioFormat validates the --format flag (flac default, wav option).
func NormalizeAudioFormat(f string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(f)) {
	case "", FormatFLAC:
		return FormatFLAC, nil
	case FormatWAV:
		return FormatWAV, nil
	default:
		return "", fmt.Errorf("unsupported --format %q (want flac or wav)", f)
	}
}

// DefaultAudioFile returns recordings/<timestamp>.<ext> for the given format.
// It is pure: creating the directory is the caller's job (see EnsureAudioPath).
func DefaultAudioFile(format string) (string, error) {
	f, err := NormalizeAudioFormat(format)
	if err != nil {
		return "", err
	}
	return filepath.Join("recordings", fmt.Sprintf("%s.%s", timestampNow(), f)), nil
}

// ResolveAudioPath resolves the output path and effective audio format without
// touching the filesystem. An explicit .flac/.wav extension wins over the
// default; if the user also passed --format explicitly (formatSet) and it
// conflicts with the extension, that is an error. Other extensions are
// rejected; a missing extension gets the format's extension.
func ResolveAudioPath(out, format string, formatSet bool) (string, string, error) {
	f, err := NormalizeAudioFormat(format)
	if err != nil {
		return "", "", err
	}
	if out == "" {
		p, err := DefaultAudioFile(f)
		return p, f, err
	}
	ext := strings.ToLower(filepath.Ext(out))
	switch ext {
	case ".flac", ".wav":
		eff := strings.TrimPrefix(ext, ".")
		if formatSet && eff != f {
			return "", "", fmt.Errorf("output extension %q conflicts with --format %s", ext, f)
		}
		return out, eff, nil
	case "":
		return out + "." + f, f, nil
	default:
		return "", "", fmt.Errorf("ksound records flac or wav only (got %q)", ext)
	}
}

// EnsureAudioPath resolves the output path (see ResolveAudioPath) and creates
// parent directories.
func EnsureAudioPath(out, format string, formatSet bool) (string, string, error) {
	p, f, err := ResolveAudioPath(out, format, formatSet)
	if err != nil {
		return "", "", err
	}
	if out == "" {
		if err := os.MkdirAll("recordings", 0o755); err != nil {
			return "", "", err
		}
		return p, f, nil
	}
	if err := mkdirFor(p); err != nil {
		return "", "", err
	}
	return p, f, nil
}

// mkdirFor creates the parent directory of path when it is not the cwd.
func mkdirFor(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0o755)
}

// recordingSpaceNeed estimates the disk bytes a recording of maxDur needs
// until the file is saved: the .partial.pcm copy (exact PCM size) plus the
// finished audio (full size for WAV, at most ~3/4 for FLAC), plus 1MB slack.
func recordingSpaceNeed(maxDur time.Duration, format string) int64 {
	pcm := int64(maxDur.Seconds() * targetSampleRate * 2)
	if format == FormatWAV {
		return 2*pcm + 1<<20
	}
	return pcm + pcm*3/4 + 1<<20
}

// evaluateRecordingSpace reports whether free bytes on the target volume can
// hold a recording of maxDur (0 = until stopped). It returns an error when a
// fixed-duration recording would certainly overflow, or a warning message
// when an open-ended recording could exhaust the disk.
func evaluateRecordingSpace(free int64, maxDur time.Duration, format string) (string, error) {
	if maxDur > 0 {
		need := recordingSpaceNeed(maxDur, format)
		if free < need {
			return "", fmt.Errorf("not enough disk space for a %s recording: need ~%s free, have %s",
				maxDur, humanBytes(need), humanBytes(free))
		}
		return "", nil
	}
	if free < maxRecordingBytes {
		return fmt.Sprintf("only %s free on the recording volume; a very long session can fill it (capture capped at %s)",
			humanBytes(free), humanBytes(maxRecordingBytes)), nil
	}
	return "", nil
}

// checkRecordingSpace probes the volume holding path before recording starts
// (see evaluateRecordingSpace). Platforms without a probe skip the check.
func checkRecordingSpace(path string, maxDur time.Duration, format string) error {
	free, err := freeDiskBytes(path)
	if err != nil {
		return nil
	}
	warnMsg, err := evaluateRecordingSpace(free, maxDur, format)
	if err != nil {
		return err
	}
	if warnMsg != "" {
		warnf("%s", warnMsg)
	}
	return nil
}

// humanBytes renders n as a compact binary unit string ("6.2 GB").
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

// InferAudioFormat returns flac/wav based on file extension.
func InferAudioFormat(path string) string {
	if strings.EqualFold(filepath.Ext(path), ".wav") {
		return FormatWAV
	}
	return FormatFLAC
}

// ListCaptureDevices returns names of available capture devices.
func ListCaptureDevices() ([]string, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	infos, err := ctx.Devices(malgo.Capture)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(infos))
	for _, d := range infos {
		marker := ""
		if d.IsDefault != 0 {
			marker = " (default)"
		}
		names = append(names, d.Name()+marker)
	}
	return names, nil
}

// stopReason identifies why recording stopped.
type stopReason int

const (
	stopEnter stopReason = iota
	stopSignal
	stopDuration
	stopCaptureError
)

// errCaptureStalled reports that the capture device stopped delivering audio
// for too long, typically because it was unplugged or claimed by another app.
var errCaptureStalled = errors.New("capture device stopped responding (unplugged or in use by another application?)")

// waitForStop blocks until one of the injected events fires and reports why.
// A nil channel blocks forever, so callers can leave duration disabled by
// passing a nil timeout.
func waitForStop(enter <-chan struct{}, sig <-chan os.Signal, timeout <-chan time.Time) stopReason {
	reason, _ := waitForStopWithErrors(enter, sig, timeout, nil)
	return reason
}

func waitForStopWithErrors(enter <-chan struct{}, sig <-chan os.Signal, timeout <-chan time.Time, errs <-chan error) (stopReason, error) {
	select {
	case <-enter:
		return stopEnter, nil
	case <-sig:
		return stopSignal, nil
	case <-timeout:
		return stopDuration, nil
	case err := <-errs:
		return stopCaptureError, err
	}
}

// monitorCaptureStall reports errCaptureStalled on out when no new audio has
// been captured for stallAfter, or for graceAfter after the device layer
// reported that the device stopped. It exits when done closes. The send on
// out is non-blocking: the first reported error wins.
func monitorCaptureStall(mu *sync.Mutex, raw *[]byte, deviceStopped *atomic.Bool, stallAfter, graceAfter, checkEvery time.Duration, done <-chan struct{}, out chan<- error) {
	ticker := time.NewTicker(checkEvery)
	defer ticker.Stop()
	lastLen := -1
	lastChange := time.Now()
	for {
		select {
		case <-done:
			return
		case now := <-ticker.C:
			mu.Lock()
			n := len(*raw)
			mu.Unlock()
			if n != lastLen {
				lastLen, lastChange = n, now
				continue
			}
			limit := stallAfter
			if deviceStopped != nil && deviceStopped.Load() {
				limit = graceAfter
			}
			if now.Sub(lastChange) >= limit {
				select {
				case out <- errCaptureStalled:
				default:
				}
				return
			}
		}
	}
}

// readEnter signals ch when a line is entered on stdin. EOF (for example when
// stdin is /dev/null or a closed pipe) is deliberately ignored so a
// non-interactive run is never stopped by an empty stdin.
func readEnter(ch chan<- struct{}) {
	if _, err := bufio.NewReader(os.Stdin).ReadString('\n'); err == nil {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// stdinIsTerminal reports whether stdin is attached to a terminal.
func stdinIsTerminal() bool {
	return isTerminal(os.Stdin)
}

// writeFullAt writes all bytes at offset, retrying legal short writes.
func writeFullAt(w io.WriterAt, data []byte, offset int64) error {
	for len(data) > 0 {
		n, err := w.WriteAt(data, offset)
		if n > 0 {
			data = data[n:]
			offset += int64(n)
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

// writePartialLoopWithErrors appends newly captured PCM bytes to path roughly
// once a second until stop is closed, then flushes the remainder. It never
// runs on the audio callback thread and reports disk errors through errCh.
func writePartialLoopWithErrors(mu *sync.Mutex, raw *[]byte, path string, stop <-chan struct{}, errCh chan<- error) {
	reportError := func(err error) {
		if errCh == nil {
			return
		}
		select {
		case errCh <- err:
		default:
		}
	}

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var f *os.File
	flushed := 0
	flush := func() error {
		mu.Lock()
		end := len(*raw)
		if end <= flushed {
			mu.Unlock()
			return nil
		}
		chunk := append([]byte(nil), (*raw)[flushed:end]...)
		start := flushed
		mu.Unlock()

		if f == nil {
			var err error
			f, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				return err
			}
		}
		if err := writeFullAt(f, chunk, int64(start)); err != nil {
			return err
		}
		if err := f.Sync(); err != nil {
			return err
		}
		flushed = end
		return nil
	}
	defer func() {
		if f != nil {
			if err := f.Close(); err != nil {
				reportError(err)
			}
		}
	}()

	for {
		select {
		case <-stop:
			if err := flush(); err != nil {
				reportError(err)
			}
			return
		case <-ticker.C:
			if err := flush(); err != nil {
				reportError(err)
				return
			}
		}
	}
}

// RecordUntilStop captures mono 16-bit audio at 16kHz. It stops on the first
// of: Enter on stdin, SIGINT/SIGTERM, or maxDur elapsing (when > 0). If
// partialPath is non-empty, captured audio is appended there periodically so
// a crash never loses the whole recording; the caller deletes it after a clean
// save. If deviceSubstr is non-empty, the first capture device whose name
// contains it (case-insensitive) is used; otherwise the default device is used.
func RecordUntilStop(deviceSubstr string, maxDur time.Duration, partialPath string) ([]int16, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("init audio context: %w", err)
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = targetSampleRate

	if deviceSubstr != "" {
		infos, err := ctx.Devices(malgo.Capture)
		if err != nil {
			return nil, fmt.Errorf("list devices: %w", err)
		}
		// selectedID is kept alive for the whole capture: DeviceID.Pointer()
		// must not point at a loop-local copy.
		var selectedID malgo.DeviceID
		matched := false
		matchedName := ""
		for i := range infos {
			if strings.Contains(strings.ToLower(infos[i].Name()), strings.ToLower(deviceSubstr)) {
				selectedID = infos[i].ID
				matchedName = infos[i].Name()
				matched = true
				break
			}
		}
		if !matched {
			return nil, fmt.Errorf("no capture device matching %q", deviceSubstr)
		}
		deviceConfig.Capture.DeviceID = selectedID.Pointer()
		infof("Using input device: %s", matchedName)
	}

	var mu sync.Mutex
	raw := make([]byte, 0, targetSampleRate*2*30) // ~30s prealloc
	overflow := false
	var deviceStopped atomic.Bool

	callbacks := malgo.DeviceCallbacks{
		Data: func(_, pSample []byte, framecount uint32) {
			if len(pSample) == 0 {
				return
			}
			mu.Lock()
			if len(raw) >= maxRecordingBytes {
				overflow = true
			} else {
				// Clamp the append so one over-long session cannot OOM.
				if len(raw)+len(pSample) > maxRecordingBytes {
					pSample = pSample[:maxRecordingBytes-len(raw)]
					overflow = true
				}
				raw = append(raw, pSample...)
			}
			mu.Unlock()
		},
		Stop: func() { deviceStopped.Store(true) },
	}

	device, err := malgo.InitDevice(ctx.Context, deviceConfig, callbacks)
	if err != nil {
		return nil, fmt.Errorf("init capture device (microphone permission granted?): %w", err)
	}
	defer device.Uninit()

	if err := device.Start(); err != nil {
		return nil, fmt.Errorf("start capture: %w", err)
	}
	defer device.Stop()

	// recordErrCh collects runtime capture failures (partial-file writes and
	// device stalls) for the wait loop below; buffer 2 so both sources can
	// report without blocking.
	recordErrCh := make(chan error, 2)

	var partialWG sync.WaitGroup
	stopPartial := make(chan struct{})
	if partialPath != "" {
		partialWG.Add(1)
		go func() {
			defer partialWG.Done()
			writePartialLoopWithErrors(&mu, &raw, partialPath, stopPartial, recordErrCh)
		}()
	}

	stopMonitor := make(chan struct{})
	var monitorWG sync.WaitGroup
	monitorWG.Add(1)
	go func() {
		defer monitorWG.Done()
		monitorCaptureStall(&mu, &raw, &deviceStopped,
			captureStallAfter, captureStopGrace, captureCheckEvery, stopMonitor, recordErrCh)
	}()

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	enterCh := make(chan struct{}, 1)
	// Only watch stdin when it is a terminal. Otherwise the reader goroutine
	// would block on ReadString forever (one leaked goroutine per run) and
	// EOF must never stop a non-interactive recording.
	interactive := stdinIsTerminal()
	if interactive {
		go readEnter(enterCh)
	} else {
		infof("stdin is not a terminal; press Ctrl+C or pass --duration to stop")
	}

	var timeoutCh <-chan time.Time
	if maxDur > 0 {
		timer := time.NewTimer(maxDur)
		defer timer.Stop()
		timeoutCh = timer.C
		infof("Recording... press Enter or Ctrl+C to stop early (max %s).", maxDur)
	} else {
		infof("Recording... press Enter to stop.")
	}

	stopMeter := startLevelMeter(&mu, &raw, time.Now())
	reason, captureErr := waitForStopWithErrors(enterCh, sigCh, timeoutCh, recordErrCh)
	stopMeter()

	if reason == stopSignal {
		// Restore default handling so a second Ctrl+C kills the process.
		signal.Reset(os.Interrupt, syscall.SIGTERM)
		fmt.Fprintln(os.Stderr, "\nInterrupted — saving recording...")
	}

	close(stopMonitor)
	monitorWG.Wait()
	close(stopPartial)
	partialWG.Wait()
	if captureErr == nil {
		select {
		case captureErr = <-recordErrCh:
		default:
		}
	}

	mu.Lock()
	defer mu.Unlock()

	if overflow {
		warnf("recording hit %dMB cap; audio truncated, use --duration chunks for very long sessions", maxRecordingBytes>>20)
	}
	n := len(raw) / 2
	samples := make([]int16, n)
	for i := 0; i < n; i++ {
		samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return samples, captureErr
}

// RecoverPartialRecordings converts leftover *.partial.pcm crash files in dir
// into <name>.recovered.flac (16kHz mono s16le), removes the .pcm files, and
// returns the recovered FLAC paths.
func RecoverPartialRecordings(dir string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.partial.pcm"))
	if err != nil {
		return nil, err
	}
	var recovered []string
	for _, m := range matches {
		raw, err := os.ReadFile(m)
		if err != nil {
			return recovered, err
		}
		if len(raw) < 2 {
			_ = os.Remove(m)
			continue
		}
		n := len(raw) / 2
		samples := make([]int16, n)
		for i := 0; i < n; i++ {
			samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		base := strings.TrimSuffix(m, ".partial.pcm")
		base = strings.TrimSuffix(base, filepath.Ext(base))
		out := base + ".recovered.flac"
		if err := SaveFLAC(out, samples); err != nil {
			return recovered, err
		}
		if err := os.Remove(m); err != nil {
			return recovered, err
		}
		recovered = append(recovered, out)
		infof("Recovered unsaved recording: %s", out)
	}
	return recovered, nil
}

// dbFS converts a linear int16 peak to dBFS (0 dB = full scale).
func dbFS(peak int) float64 {
	if peak <= 0 {
		return math.Inf(-1)
	}
	return 20 * math.Log10(float64(peak)/32768)
}

// meterBar renders a fixed-width level bar for db in [-60, 0].
func meterBar(db float64, width int) string {
	filled := int((db + 60) / 60 * float64(width))
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}
	return strings.Repeat("#", filled) + strings.Repeat("-", width-filled)
}

func formatDB(db float64) string {
	if math.IsInf(db, -1) {
		return "  -inf"
	}
	return fmt.Sprintf("%5.1f", db)
}

// formatElapsed renders durations for the meter: "12.3s" under a minute,
// "4:05" under an hour, "1:02:03" beyond.
func formatElapsed(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	s := int(d.Seconds())
	if s < 60 {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	h, m, sec := s/3600, (s%3600)/60, s%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, sec)
	}
	return fmt.Sprintf("%d:%02d", m, sec)
}

// startLevelMeter draws a live ASCII level meter with peak-hold on the
// terminal while recording. Levels are pre-gain (raw mic input), so the
// peak readout doubles as a gain-calibration aid: aim for peaks between
// -12 and -6 dB, raise --gain if peaks sit below -24 dB.
// It is disabled when stdout is not a terminal (piped output).
// The returned func stops the meter and moves to a fresh line.
func startLevelMeter(mu *sync.Mutex, raw *[]byte, t0 time.Time) (stop func()) {
	noop := func() {}
	fi, err := os.Stdout.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice == 0 {
		return noop
	}
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		peakHold := 0
		const windowBytes = targetSampleRate * 2 / 2 // trailing 0.5s
		const width = 30
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				mu.Lock()
				n := len(*raw)
				start := n - windowBytes
				if start < 0 {
					start = 0
				}
				snap := make([]byte, n-start)
				copy(snap, (*raw)[start:])
				mu.Unlock()

				peak := 0
				for i := 0; i+1 < len(snap); i += 2 {
					v := int(int16(binary.LittleEndian.Uint16(snap[i:])))
					if v < 0 {
						v = -v
					}
					if v > peak {
						peak = v
					}
				}
				if peak > peakHold {
					peakHold = peak
				}
				db, peakDB := dbFS(peak), dbFS(peakHold)
				fmt.Printf("\r[%s] %7s  %s dB  peak %s dB   ",
					meterBar(db, width), formatElapsed(time.Since(t0)), formatDB(db), formatDB(peakDB))
			}
		}
	}()
	return func() { close(done); wg.Wait(); fmt.Println() }
}

// SaveWAV writes mono 16-bit samples to path in chunks so hour-long
// recordings do not need a second full-size []int copy.
func SaveWAV(path string, samples []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := wav.NewEncoder(f, targetSampleRate, 16, 1, 1)
	defer enc.Close()

	const chunk = 64 * 1024
	format := &audio.Format{NumChannels: 1, SampleRate: targetSampleRate}
	data := make([]int, 0, min(len(samples), chunk))
	for start := 0; start < len(samples); start += chunk {
		end := start + chunk
		if end > len(samples) {
			end = len(samples)
		}
		data = data[:0]
		for _, s := range samples[start:end] {
			data = append(data, int(s))
		}
		buf := &audio.IntBuffer{Format: format, Data: data, SourceBitDepth: 16}
		if err := enc.Write(buf); err != nil {
			return fmt.Errorf("write wav: %w", err)
		}
	}
	return enc.Close()
}

// SaveFLAC writes mono 16-bit samples as FLAC (level 5) in chunks so
// hour-long recordings do not need a second full-size []byte copy.
func SaveFLAC(path string, samples []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc, err := flacpcm.NewEncoder(f, flacpcm.Config{
		SampleRate:       targetSampleRate,
		BitDepth:         16,
		Channels:         1,
		CompressionLevel: 5,
		TotalSamples:     uint64(len(samples)),
	})
	if err != nil {
		f.Close()
		return fmt.Errorf("init flac encoder: %w", err)
	}
	const chunk = 64 * 1024
	raw := make([]byte, 0, min(len(samples), chunk)*2)
	for start := 0; start < len(samples); start += chunk {
		end := start + chunk
		if end > len(samples) {
			end = len(samples)
		}
		raw = raw[:0]
		for _, s := range samples[start:end] {
			raw = append(raw, byte(s), byte(uint16(s)>>8))
		}
		if _, err := enc.Write(raw); err != nil {
			f.Close()
			return fmt.Errorf("write flac: %w", err)
		}
	}
	if err := enc.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// SaveAudio dispatches on format (flac default).
func SaveAudio(path, format string, samples []int16) error {
	if format == FormatWAV {
		return SaveWAV(path, samples)
	}
	return SaveFLAC(path, samples)
}

// ApplyGain multiplies PCM16 samples by gain (1.0 = unchanged).
// Values are rounded and clamped to int16 range; returns clipped count.
func ApplyGain(samples []int16, gain float64) int {
	if gain == 1.0 {
		return 0
	}
	clipped := 0
	for i, s := range samples {
		v := int(math.Round(float64(s) * gain))
		if v > math.MaxInt16 {
			v = math.MaxInt16
			clipped++
		} else if v < math.MinInt16 {
			v = math.MinInt16
			clipped++
		}
		samples[i] = int16(v)
	}
	return clipped
}

// Int16ToFloat32 normalizes PCM16 samples to [-1, 1] for the recognizer.
func Int16ToFloat32(samples []int16) []float32 {
	out := make([]float32, len(samples))
	for i, s := range samples {
		out[i] = float32(s) / 32768
	}
	return out
}

// DecodeFLACFile reads a FLAC file, mixes to mono float32, and returns the
// stream sample rate. Non-16-bit depths are scaled accordingly.
// Note: the whole stream is held in memory; hour-long files need ~700MB.
// Chunked streaming decode would be the next step if that matters.
func DecodeFLACFile(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	dec, err := flacpcm.NewDecoder(f)
	if err != nil {
		return nil, 0, fmt.Errorf("decode flac %s: %w", path, err)
	}
	info := dec.Info()
	if info.Channels < 1 || info.Channels > 8 {
		return nil, 0, fmt.Errorf("decode flac %s: unsupported channels %d", path, info.Channels)
	}
	raw, err := io.ReadAll(dec)
	if err != nil {
		return nil, 0, fmt.Errorf("decode flac %s: %w", path, err)
	}
	var bytesPS int
	var scale float32
	switch info.BitDepth {
	case 8:
		bytesPS, scale = 1, 128
	case 16:
		bytesPS, scale = 2, 32768
	case 24:
		bytesPS, scale = 3, 8388608
	case 32:
		bytesPS, scale = 4, 2147483648
	default:
		return nil, 0, fmt.Errorf("decode flac %s: unsupported bit depth %d", path, info.BitDepth)
	}
	frameSize := bytesPS * info.Channels
	if len(raw)%frameSize != 0 {
		return nil, 0, fmt.Errorf("decode flac %s: truncated stream", path)
	}
	frames := len(raw) / frameSize
	out := make([]float32, frames)
	for i := 0; i < frames; i++ {
		var acc float32
		for ch := 0; ch < info.Channels; ch++ {
			off := i*frameSize + ch*bytesPS
			var v float32
			switch info.BitDepth {
			case 8:
				v = float32(int8(raw[off])) / scale
			case 16:
				v = float32(int16(binary.LittleEndian.Uint16(raw[off:]))) / scale
			case 24:
				n := int32(raw[off]) | int32(raw[off+1])<<8 | int32(raw[off+2])<<16
				if n&(1<<23) != 0 {
					n |= ^int32(0xFFFFFF)
				}
				v = float32(n) / scale
			case 32:
				v = float32(int32(binary.LittleEndian.Uint32(raw[off:]))) / scale
			}
			acc += v
		}
		out[i] = acc / float32(info.Channels)
	}
	if len(out) == 0 {
		return nil, 0, fmt.Errorf("decode flac %s: empty stream", path)
	}
	return out, info.SampleRate, nil
}

// DecodeWAVFile reads a WAV file, mixes to mono float32, and returns the
// stream sample rate.
func DecodeWAVFile(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	if !dec.IsValidFile() {
		if err := dec.Err(); err != nil {
			return nil, 0, fmt.Errorf("decode wav %s: %w", path, err)
		}
		return nil, 0, fmt.Errorf("decode wav %s: invalid file", path)
	}
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decode wav %s: %w", path, err)
	}
	if buf == nil || len(buf.Data) == 0 {
		return nil, 0, fmt.Errorf("decode wav %s: empty stream", path)
	}
	ch := buf.Format.NumChannels
	if ch < 1 {
		return nil, 0, fmt.Errorf("decode wav %s: no channels", path)
	}
	depth := int(buf.SourceBitDepth)
	if depth == 0 {
		depth = 16
	}
	scale := float32(int64(1) << (depth - 1))
	frames := len(buf.Data) / ch
	out := make([]float32, frames)
	for i := 0; i < frames; i++ {
		var acc float32
		for c := 0; c < ch; c++ {
			acc += float32(buf.Data[i*ch+c]) / scale
		}
		out[i] = acc / float32(ch)
	}
	return out, buf.Format.SampleRate, nil
}

// ResampleLinear resamples mono float32 audio by linear interpolation.
// Returns src unchanged when rates match or are invalid. Linear interpolation
// is adequate for VAD scouting; the recognizer always receives the original
// rate (see transcribeLong) so ASR quality is unaffected.
func ResampleLinear(src []float32, fromRate, toRate int) []float32 {
	if fromRate <= 0 || toRate <= 0 || fromRate == toRate || len(src) == 0 {
		return src
	}
	n := int(int64(len(src)) * int64(toRate) / int64(fromRate))
	if n < 1 {
		n = 1
	}
	out := make([]float32, n)
	step := float64(fromRate) / float64(toRate)
	for i := range out {
		pos := float64(i) * step
		lo := int(pos)
		hi := lo + 1
		if hi >= len(src) {
			hi = len(src) - 1
		}
		frac := float32(pos - float64(lo))
		out[i] = src[lo]*(1-frac) + src[hi]*frac
	}
	return out
}

// LoadAudioForTranscribe decodes .wav or .flac into mono float32 samples.
func LoadAudioForTranscribe(path string) ([]float32, int, error) {
	if InferAudioFormat(path) == FormatWAV {
		return DecodeWAVFile(path)
	}
	return DecodeFLACFile(path)
}
