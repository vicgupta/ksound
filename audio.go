package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	flacpcm "github.com/tphakala/go-flac/pcm"
)

const targetSampleRate = 16000

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

// DefaultAudioFile returns recordings/<timestamp>.<ext> for the given format,
// creating the recordings dir.
func DefaultAudioFile(format string) (string, error) {
	if err := os.MkdirAll("recordings", 0o755); err != nil {
		return "", err
	}
	return filepath.Join("recordings", fmt.Sprintf("%s.%s", timestampNow(), format)), nil
}

// EnsureAudioPath resolves an explicit --audio-out/--output path: if empty a
// timestamped default is used; if the path has no audio extension the format's
// extension is appended; parent dirs are created.
func EnsureAudioPath(out, format string) (string, error) {
	if out == "" {
		return DefaultAudioFile(format)
	}
	ext := strings.ToLower(filepath.Ext(out))
	if ext != ".flac" && ext != ".wav" {
		out += "." + format
	}
	if dir := filepath.Dir(out); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
	}
	return out, nil
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

// RecordUntilEnter captures mono 16-bit audio at 16kHz until Enter is pressed.
// If deviceSubstr is non-empty, the first capture device whose name contains
// it (case-insensitive) is used; otherwise the default device is used.
func RecordUntilEnter(deviceSubstr string) ([]int16, error) {
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
		matched := false
		for _, d := range infos {
			if strings.Contains(strings.ToLower(d.Name()), strings.ToLower(deviceSubstr)) {
				// Keep a copy alive for the duration of capture.
				id := d.ID
				deviceConfig.Capture.DeviceID = id.Pointer()
				matched = true
				fmt.Printf("Using input device: %s\n", d.Name())
				break
			}
		}
		if !matched {
			return nil, fmt.Errorf("no capture device matching %q", deviceSubstr)
		}
	}

	var mu sync.Mutex
	raw := make([]byte, 0, targetSampleRate*2*30) // ~30s prealloc

	callbacks := malgo.DeviceCallbacks{
		Data: func(_, pSample []byte, framecount uint32) {
			if len(pSample) == 0 {
				return
			}
			mu.Lock()
			raw = append(raw, pSample...)
			mu.Unlock()
		},
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

	fmt.Println("Recording... press Enter to stop.")
	stopMeter := startLevelMeter(&mu, &raw, time.Now())
	if _, err := fmt.Scanln(); err != nil {
		// Scanln returns error on empty line in some cases; treat as stop.
		fmt.Fprintln(os.Stderr, "")
	}
	stopMeter()

	mu.Lock()
	defer mu.Unlock()

	n := len(raw) / 2
	samples := make([]int16, n)
	for i := 0; i < n; i++ {
		samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return samples, nil
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

// SaveWAV writes mono 16-bit samples to path.
func SaveWAV(path string, samples []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := wav.NewEncoder(f, targetSampleRate, 16, 1, 1)
	defer enc.Close()

	data := make([]int, len(samples))
	for i, s := range samples {
		data[i] = int(s)
	}
	buf := &audio.IntBuffer{
		Format:         &audio.Format{NumChannels: 1, SampleRate: targetSampleRate},
		Data:           data,
		SourceBitDepth: 16,
	}
	if err := enc.Write(buf); err != nil {
		return fmt.Errorf("write wav: %w", err)
	}
	return enc.Close()
}

// SaveFLAC writes mono 16-bit samples as FLAC (level 5).
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
	raw := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(s))
	}
	if _, err := enc.Write(raw); err != nil {
		f.Close()
		return fmt.Errorf("write flac: %w", err)
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
// Returns src unchanged when rates match.
func ResampleLinear(src []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate || len(src) == 0 {
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
