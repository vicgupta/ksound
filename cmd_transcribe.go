package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// defaultThreads scales with the machine instead of a fixed 2.
func defaultThreads() int {
	if n := runtime.NumCPU(); n > 0 {
		if n > 8 {
			return 8
		}
		return n
	}
	return 2
}

// validateGain checks --gain bounds (0 < gain <= 20).
func validateGain(gain float64) error {
	if gain <= 0 || gain > 20 {
		return fmt.Errorf("invalid --gain %v (want 0 < gain <= 20)", gain)
	}
	return nil
}

// writeTranscriptFile writes formatted to out, or stdout when out is "-".
func writeTranscriptFile(out, formatted string) error {
	if out == "-" {
		_, err := os.Stdout.WriteString(formatted)
		return err
	}
	return os.WriteFile(out, []byte(formatted), 0o644)
}

func resolveModel(encoder, decoder, joiner, tokens string) (ModelFiles, error) {
	if encoder != "" || decoder != "" || joiner != "" || tokens != "" {
		if encoder == "" || decoder == "" || joiner == "" || tokens == "" {
			return ModelFiles{}, fmt.Errorf("--encoder, --decoder, --joiner and --tokens must be given together")
		}
		return ModelFiles{Encoder: encoder, Decoder: decoder, Joiner: joiner, Tokens: tokens}, nil
	}
	return EnsureModel()
}

// modelLoadError builds a clear message for a failed sherpa initialisation,
// pointing at the cache directory the user can delete to force a re-download.
func modelLoadError(what string) error {
	if dir, err := ModelCacheDir(); err == nil {
		return fmt.Errorf("failed to %s (corrupt cache?); delete %s and retry", what, dir)
	}
	return fmt.Errorf("failed to %s (corrupt cache?)", what)
}

// maxSingleShotSeconds caps single-shot decoding: the Parakeet encoder
// crashes (ONNX broadcast error) on very long inputs, so longer audio is
// split into speech segments with Silero VAD and decoded piece by piece.
const maxSingleShotSeconds = 300

func normalizeTranscriptFormat(format string) (string, error) {
	if format == "" {
		return "text", nil
	}
	switch format {
	case "text", "markdown":
		return format, nil
	default:
		return "", fmt.Errorf("unsupported transcript format %q (choose text or markdown)", format)
	}
}

func defaultTranscriptPath(audioPath, format string) string {
	ext := ".txt"
	if format == "markdown" {
		ext = ".md"
	}
	return strings.TrimSuffix(audioPath, filepath.Ext(audioPath)) + ext
}

func transcriptOutputPath(audioPath, explicitPath, format string) string {
	if explicitPath != "" {
		return explicitPath
	}
	return defaultTranscriptPath(audioPath, format)
}

func formatTranscript(audioPath, transcript, format string, createdAt time.Time) (string, error) {
	format, err := normalizeTranscriptFormat(format)
	if err != nil {
		return "", err
	}
	if format == "text" {
		return transcript + "\n", nil
	}

	name := filepath.Base(audioPath)
	title := strings.TrimSuffix(name, filepath.Ext(name))
	source := strings.ReplaceAll(name, "`", "\\`")
	return fmt.Sprintf("# %s\n\nSource: `%s`\nTranscribed: %s\n\n%s\n",
		title, source, createdAt.Format("2006-01-02 15:04 MST"), transcript), nil
}

// vadOptions tunes Silero VAD segmentation for long audio.
type vadOptions struct {
	threshold  float32
	minSilence float32
	minSpeech  float32
	maxSpeech  float32
}

// defaultVadOptions matches the previously hardcoded values.
func defaultVadOptions() vadOptions {
	return vadOptions{threshold: 0.5, minSilence: 0.5, minSpeech: 0.25, maxSpeech: 20}
}

func (v vadOptions) validate() error {
	if v.threshold <= 0 || v.threshold >= 1 {
		return fmt.Errorf("invalid --vad-threshold %v (want 0 < t < 1)", v.threshold)
	}
	if v.minSilence < 0 || v.minSpeech < 0 || v.maxSpeech <= 0 {
		return fmt.Errorf("invalid VAD durations (min-silence=%v min-speech=%v max-speech=%v)",
			v.minSilence, v.minSpeech, v.maxSpeech)
	}
	return nil
}

func addVadFlags(cmd *cobra.Command, v *vadOptions) {
	cmd.Flags().Float32Var(&v.threshold, "vad-threshold", 0.5, "VAD speech threshold (0-1, higher = stricter)")
	cmd.Flags().Float32Var(&v.minSilence, "vad-min-silence", 0.5, "VAD minimum silence seconds to split")
	cmd.Flags().Float32Var(&v.minSpeech, "vad-min-speech", 0.25, "VAD minimum speech seconds to keep")
	cmd.Flags().Float32Var(&v.maxSpeech, "vad-max-speech", 20, "VAD maximum segment seconds")
}

// pipeline runs the transcription flow with swappable dependencies so tests
// can drive it without the native sherpa library or a model on disk.
type pipeline struct {
	vad       vadOptions
	threads   int
	newRec    func(m ModelFiles, threads int) (recognizer, error)
	newVAD    func(opts vadOptions) (vad, error)
	loadAudio func(path string) ([]float32, int, error)
	copyText  func(text string) error
}

// defaultPipeline wires the pipeline to the real sherpa implementations.
func defaultPipeline(vad vadOptions, threads int) *pipeline {
	return &pipeline{
		vad:       vad,
		threads:   threads,
		newRec:    newSherpaRecognizer,
		newVAD:    newSherpaVAD,
		loadAudio: LoadAudioForTranscribe,
		copyText:  copyToClipboard,
	}
}

func transcribeFile(audioPath, txtOut, outputFormat string, threads int, vad vadOptions, m ModelFiles) (string, error) {
	return defaultPipeline(vad, threads).run(audioPath, txtOut, outputFormat, m)
}

func (p *pipeline) run(audioPath, txtOut, outputFormat string, m ModelFiles) (string, error) {
	var err error
	outputFormat, err = normalizeTranscriptFormat(outputFormat)
	if err != nil {
		return "", err
	}
	threads := p.threads
	if threads <= 0 {
		threads = defaultThreads()
	}
	if err := p.vad.validate(); err != nil {
		return "", err
	}
	if _, err := os.Stat(audioPath); err != nil {
		return "", fmt.Errorf("input file: %w", err)
	}

	samples, sampleRate, err := p.loadAudio(audioPath)
	if err != nil {
		return "", err
	}
	if len(samples) == 0 {
		return "", fmt.Errorf("empty audio: %s", audioPath)
	}

	infof("Loading model (first run takes a while)...")
	rec, err := p.newRec(m, threads)
	if err != nil {
		return "", err
	}
	defer rec.Close()

	duration := float64(len(samples)) / float64(sampleRate)
	decodeStart := time.Now()
	var text string
	if duration > maxSingleShotSeconds {
		infof("Long audio (%.0fs): splitting on silence with VAD...", duration)
		text, err = p.transcribeLong(rec, samples, sampleRate)
		if err != nil {
			return "", err
		}
		text = strings.TrimSpace(text)
	} else {
		text, err = rec.Decode(samples, sampleRate)
		if err != nil {
			return "", err
		}
		text = strings.TrimSpace(text)
	}
	decodeWall := time.Since(decodeStart)
	infof("Decoded %.1fs of audio in %s with %d thread(s) (RTF %.2f)",
		duration, decodeWall.Round(time.Millisecond), threads, decodeWall.Seconds()/duration)

	createdAt := time.Now()
	formatted, err := formatTranscript(audioPath, text, outputFormat, createdAt)
	if err != nil {
		return "", err
	}
	out := transcriptOutputPath(audioPath, txtOut, outputFormat)
	if err := writeTranscriptFile(out, formatted); err != nil {
		return "", err
	}
	if out != "-" {
		infof("Transcription written to %s", out)
	}
	if text == "" {
		infof("(empty result — input may be silent or too short)")
		return out, nil
	}
	fmt.Printf("Transcription:\n%s\n", text)
	if err := p.copyText(text); err != nil {
		warnf("Could not copy transcript to clipboard: %v", err)
	} else {
		infof("Transcript copied to clipboard.")
	}
	return out, nil
}

// transcribeLong splits audio into ≤20s speech segments with Silero VAD
// (the same approach as sherpa-onnx-vad-with-offline-asr) and decodes
// each segment separately. Segment boundaries are found at 16kHz, then
// mapped back to the original sample rate so the recognizer always gets
// un-resampled audio.
func (p *pipeline) transcribeLong(rec recognizer, samples []float32, sampleRate int) (string, error) {
	vadSamples := samples
	if sampleRate != vadAnalysisRate {
		vadSamples = ResampleLinear(samples, sampleRate, vadAnalysisRate)
	}

	v, err := p.newVAD(p.vad)
	if err != nil {
		return "", err
	}
	defer v.Close()

	decodeSegment := func(start, end int) (string, error) {
		// Map 16kHz VAD indices back to the original rate.
		a := start * sampleRate / vadAnalysisRate
		b := end * sampleRate / vadAnalysisRate
		if a < 0 {
			a = 0
		}
		if b > len(samples) {
			b = len(samples)
		}
		if b-a < sampleRate/10 { // <0.1s: same cutoff as the reference
			return "", nil
		}
		return rec.Decode(samples[a:b], sampleRate)
	}

	var texts []string
	drain := func() error {
		for !v.IsEmpty() {
			seg := v.Front()
			v.Pop()
			t, err := decodeSegment(seg.Start, seg.Start+seg.N)
			if err != nil {
				return err
			}
			if t == "" {
				continue
			}
			texts = append(texts, t)
			infof("  [%d] %s …", len(texts), firstWords(t, 8))
		}
		return nil
	}

	i := 0
	for i < len(vadSamples) {
		if i+vadWindowSize <= len(vadSamples) {
			v.AcceptWaveform(vadSamples[i : i+vadWindowSize])
		} else {
			v.Flush()
		}
		i += vadWindowSize
		if err := drain(); err != nil {
			return "", err
		}
	}
	// In case the length was an exact multiple of the window size,
	// the tail Flush() above never ran.
	v.Flush()
	if err := drain(); err != nil {
		return "", err
	}

	return strings.Join(texts, " "), nil
}

func firstWords(s string, n int) string {
	w := strings.Fields(s)
	if len(w) > n {
		return strings.Join(w[:n], " ") + "…"
	}
	return s
}

func newTranscribeCmd() *cobra.Command {
	var output string
	var outputFormat string
	var threads int
	var encoder, decoder, joiner, tokens string
	vad := defaultVadOptions()

	cmd := &cobra.Command{
		Use:   "transcribe <file.flac|file.wav>",
		Short: "Transcribe a FLAC/WAV file locally",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			// Validate flags before resolveModel so a typo never
			// triggers a 630MB download.
			if _, err := normalizeTranscriptFormat(outputFormat); err != nil {
				return err
			}
			if err := vad.validate(); err != nil {
				return err
			}
			if threads < 0 {
				return fmt.Errorf("invalid --threads %d (want >= 0)", threads)
			}
			m, err := resolveModel(encoder, decoder, joiner, tokens)
			if err != nil {
				return err
			}
			_, err = transcribeFile(args[0], output, outputFormat, threads, vad, m)
			return err
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "output text path (default: <input>.txt; \"-\" = stdout)")
	cmd.Flags().StringVar(&outputFormat, "transcript-format", "text", "transcript format: text or markdown")
	cmd.Flags().IntVar(&threads, "threads", 0, "ONNX threads (0 = auto)")
	cmd.Flags().StringVar(&encoder, "encoder", "", "override: path to encoder.int8.onnx")
	cmd.Flags().StringVar(&decoder, "decoder", "", "override: path to decoder.int8.onnx")
	cmd.Flags().StringVar(&joiner, "joiner", "", "override: path to joiner.int8.onnx")
	cmd.Flags().StringVar(&tokens, "tokens", "", "override: path to tokens.txt")
	addVadFlags(cmd, &vad)
	return cmd
}
