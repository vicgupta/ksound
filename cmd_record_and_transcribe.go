package main

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

type combinedOptions struct {
	audioOut                         string
	txtOut                           string
	device                           string
	format                           string
	transcriptFormat                 string
	gain                             float64
	duration                         time.Duration
	threads                          int
	keepAudio                        bool
	formatSet                        bool
	vad                              vadOptions
	encoder, decoder, joiner, tokens string
}

func runRecordAndTranscribe(o combinedOptions) error {
	// Validate flags before any side effects or model download.
	if err := validateGain(o.gain); err != nil {
		return err
	}
	if err := o.vad.validate(); err != nil {
		return err
	}
	if _, err := normalizeTranscriptFormat(o.transcriptFormat); err != nil {
		return err
	}
	if o.threads < 0 {
		return fmt.Errorf("invalid --threads %d (want >= 0)", o.threads)
	}
	recoverLeftovers()
	// Resolve (and, on first run, download) the model before recording so the
	// user is not left waiting after they have already spoken.
	m, err := resolveModel(o.encoder, o.decoder, o.joiner, o.tokens)
	if err != nil {
		return err
	}
	audioPath, err := recordToFile(o.audioOut, o.device, o.format, o.formatSet, o.gain, o.duration)
	if err != nil {
		return err
	}
	txtPath, err := transcribeFile(audioPath, o.txtOut, o.transcriptFormat, o.threads, o.vad, m)
	if err != nil {
		return err
	}
	if !o.keepAudio {
		if err := os.Remove(audioPath); err != nil {
			warnf("could not remove intermediate %s: %v", audioPath, err)
		} else {
			infof("(removed intermediate %s)", audioPath)
		}
		return nil
	}
	infof("Done.\n  audio: %s\n  text:  %s", audioPath, txtPath)
	return nil
}

func addCombinedFlags(cmd *cobra.Command, o *combinedOptions) {
	o.vad = defaultVadOptions()
	cmd.Flags().StringVar(&o.audioOut, "audio-out", "", "intermediate audio path (default recordings/YYYY-MM-DD_HH-MM-SS.flac)")
	cmd.Flags().StringVarP(&o.txtOut, "output", "o", "", "output text path (default: <audio>.txt; \"-\" = stdout)")
	cmd.Flags().StringVar(&o.transcriptFormat, "transcript-format", "text", "transcript format: text or markdown")
	cmd.Flags().StringVar(&o.device, "device", "", "capture device name substring (default: system default)")
	cmd.Flags().StringVar(&o.format, "format", FormatFLAC, "audio format: flac or wav")
	cmd.Flags().Float64Var(&o.gain, "gain", 1.0, "software gain multiplier, e.g. 2.0 = +6dB")
	cmd.Flags().DurationVar(&o.duration, "duration", 0, "max recording duration (e.g. 30s); 0 = until Enter/Ctrl+C")
	cmd.Flags().IntVar(&o.threads, "threads", 0, "ONNX threads (0 = auto)")
	cmd.Flags().BoolVar(&o.keepAudio, "keep-audio", true, "keep intermediate audio after transcription")
	cmd.Flags().StringVar(&o.encoder, "encoder", "", "override: path to encoder.int8.onnx")
	cmd.Flags().StringVar(&o.decoder, "decoder", "", "override: path to decoder.int8.onnx")
	cmd.Flags().StringVar(&o.joiner, "joiner", "", "override: path to joiner.int8.onnx")
	cmd.Flags().StringVar(&o.tokens, "tokens", "", "override: path to tokens.txt")
	addVadFlags(cmd, &o.vad)
}

func newRecordAndTranscribeCmd() *cobra.Command {
	var o combinedOptions

	cmd := &cobra.Command{
		Use:     "record-and-transcribe",
		Aliases: []string{"run", "default"},
		Short:   "Record until Enter, then transcribe (default flow, FLAC)",
		RunE: func(cmd *cobra.Command, args []string) error {
			o.formatSet = cmd.Flags().Changed("format")
			return runRecordAndTranscribe(o)
		},
	}
	addCombinedFlags(cmd, &o)
	return cmd
}
