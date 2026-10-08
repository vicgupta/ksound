package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/spf13/cobra"
)

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

func transcribeFile(audioPath, txtOut, outputFormat string, threads int, m ModelFiles) (string, error) {
	var err error
	outputFormat, err = normalizeTranscriptFormat(outputFormat)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(audioPath); err != nil {
		return "", fmt.Errorf("input file: %w", err)
	}

	config := sherpa.OfflineRecognizerConfig{}
	config.ModelConfig.Transducer.Encoder = m.Encoder
	config.ModelConfig.Transducer.Decoder = m.Decoder
	config.ModelConfig.Transducer.Joiner = m.Joiner
	config.ModelConfig.Tokens = m.Tokens
	config.ModelConfig.NumThreads = threads
	config.ModelConfig.Provider = "cpu"
	config.ModelConfig.Debug = 0
	config.ModelConfig.ModelType = "nemo_transducer"
	config.DecodingMethod = "greedy_search"

	samples, sampleRate, err := LoadAudioForTranscribe(audioPath)
	if err != nil {
		return "", err
	}
	if len(samples) == 0 {
		return "", fmt.Errorf("empty audio: %s", audioPath)
	}

	fmt.Println("Loading model (first run takes a while)...")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	if recognizer == nil {
		return "", modelLoadError("load the speech model")
	}
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	duration := float64(len(samples)) / float64(sampleRate)
	var text string
	if duration > maxSingleShotSeconds {
		fmt.Printf("Long audio (%.0fs): splitting on silence with VAD...\n", duration)
		text, err = transcribeLong(recognizer, samples, sampleRate)
		if err != nil {
			return "", err
		}
		text = strings.TrimSpace(text)
	} else {
		stream := sherpa.NewOfflineStream(recognizer)
		defer sherpa.DeleteOfflineStream(stream)

		stream.AcceptWaveform(sampleRate, samples)
		recognizer.Decode(stream)
		result := stream.GetResult()
		text = strings.TrimSpace(result.Text)
	}

	createdAt := time.Now()
	formatted, err := formatTranscript(audioPath, text, outputFormat, createdAt)
	if err != nil {
		return "", err
	}
	out := transcriptOutputPath(audioPath, txtOut, outputFormat)
	if err := os.WriteFile(out, []byte(formatted), 0o644); err != nil {
		return "", err
	}
	fmt.Printf("Transcription written to %s\n", out)
	if text == "" {
		fmt.Println("(empty result — input may be silent or too short)")
		return out, nil
	}
	fmt.Printf("Transcription:\n%s\n", text)
	if err := copyToClipboard(text); err != nil {
		fmt.Fprintf(os.Stderr, "Could not copy transcript to clipboard: %v\n", err)
	} else {
		fmt.Println("Transcript copied to clipboard.")
	}
	return out, nil
}

// transcribeLong splits audio into ≤20s speech segments with Silero VAD
// (the same approach as sherpa-onnx-vad-with-offline-asr) and decodes
// each segment separately. Segment boundaries are found at 16kHz, then
// mapped back to the original sample rate so the recognizer always gets
// un-resampled audio.
func transcribeLong(recognizer *sherpa.OfflineRecognizer, samples []float32, sampleRate int) (string, error) {
	const vadRate = 16000
	const windowSize = 512 // Silero training size for 16kHz

	vadSamples := samples
	if sampleRate != vadRate {
		vadSamples = ResampleLinear(samples, sampleRate, vadRate)
	}

	vadPath, err := EnsureVadModel()
	if err != nil {
		return "", err
	}
	vadConfig := sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              vadPath,
			Threshold:          0.5,
			MinSilenceDuration: 0.5,
			MinSpeechDuration:  0.25,
			WindowSize:         windowSize,
			MaxSpeechDuration:  20,
		},
		SampleRate: vadRate,
		NumThreads: 1,
		Provider:   "cpu",
	}
	vad := sherpa.NewVoiceActivityDetector(&vadConfig, 100)
	if vad == nil {
		return "", modelLoadError(fmt.Sprintf("create the VAD (model: %s)", vadPath))
	}
	defer sherpa.DeleteVoiceActivityDetector(vad)

	decodeSegment := func(start, end int) (string, error) {
		// Map 16kHz VAD indices back to the original rate.
		a := start * sampleRate / vadRate
		b := end * sampleRate / vadRate
		if a < 0 {
			a = 0
		}
		if b > len(samples) {
			b = len(samples)
		}
		if b-a < sampleRate/10 { // <0.1s: same cutoff as the reference
			return "", nil
		}
		seg := samples[a:b]
		stream := sherpa.NewOfflineStream(recognizer)
		defer sherpa.DeleteOfflineStream(stream)
		stream.AcceptWaveform(sampleRate, seg)
		recognizer.Decode(stream)
		return strings.TrimSpace(stream.GetResult().Text), nil
	}

	var texts []string
	drain := func() error {
		for !vad.IsEmpty() {
			seg := vad.Front()
			start, n := seg.Start, len(seg.Samples)
			vad.Pop()
			t, err := decodeSegment(start, start+n)
			if err != nil {
				return err
			}
			if t == "" {
				continue
			}
			texts = append(texts, t)
			fmt.Printf("  [%d] %s …\n", len(texts), firstWords(t, 8))
		}
		return nil
	}

	i := 0
	for i < len(vadSamples) {
		if i+windowSize <= len(vadSamples) {
			vad.AcceptWaveform(vadSamples[i : i+windowSize])
		} else {
			vad.Flush()
		}
		i += windowSize
		if err := drain(); err != nil {
			return "", err
		}
	}
	// In case the length was an exact multiple of the window size,
	// the tail Flush() above never ran.
	vad.Flush()
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

	cmd := &cobra.Command{
		Use:   "transcribe <file.flac|file.wav>",
		Short: "Transcribe a FLAC/WAV file locally",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := resolveModel(encoder, decoder, joiner, tokens)
			if err != nil {
				return err
			}
			_, err = transcribeFile(args[0], output, outputFormat, threads, m)
			return err
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "output text path (default: <input>.txt)")
	cmd.Flags().StringVar(&outputFormat, "transcript-format", "text", "transcript format: text or markdown")
	cmd.Flags().IntVar(&threads, "threads", 2, "ONNX threads")
	cmd.Flags().StringVar(&encoder, "encoder", "", "override: path to encoder.int8.onnx")
	cmd.Flags().StringVar(&decoder, "decoder", "", "override: path to decoder.int8.onnx")
	cmd.Flags().StringVar(&joiner, "joiner", "", "override: path to joiner.int8.onnx")
	cmd.Flags().StringVar(&tokens, "tokens", "", "override: path to tokens.txt")
	return cmd
}
