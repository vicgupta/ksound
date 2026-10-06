package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

type combinedOptions struct {
	audioOut                         string
	txtOut                           string
	device                           string
	format                           string
	gain                             float64
	threads                          int
	keepAudio                        bool
	encoder, decoder, joiner, tokens string
}

func runRecordAndTranscribe(o combinedOptions) error {
	audioPath, err := recordToFile(o.audioOut, o.device, o.format, o.gain)
	if err != nil {
		return err
	}
	m, err := resolveModel(o.encoder, o.decoder, o.joiner, o.tokens)
	if err != nil {
		return err
	}
	txtPath, err := transcribeFile(audioPath, o.txtOut, o.threads, m)
	if err != nil {
		return err
	}
	if !o.keepAudio {
		_ = os.Remove(audioPath)
		fmt.Printf("(removed intermediate %s)\n", audioPath)
		return nil
	}
	fmt.Printf("Done.\n  audio: %s\n  text:  %s\n", audioPath, txtPath)
	return nil
}

func addCombinedFlags(cmd *cobra.Command, o *combinedOptions) {
	cmd.Flags().StringVar(&o.audioOut, "audio-out", "", "intermediate audio path (default recordings/YYYY-MM-DD_HH-MM-SS.flac)")
	cmd.Flags().StringVar(&o.audioOut, "wav-out", "", "alias of --audio-out")
	_ = cmd.Flags().MarkDeprecated("wav-out", "use --audio-out instead")
	cmd.Flags().StringVarP(&o.txtOut, "output", "o", "", "output text path (default: <audio>.txt)")
	cmd.Flags().StringVar(&o.device, "device", "", "capture device name substring (default: system default)")
	cmd.Flags().StringVar(&o.format, "format", FormatFLAC, "audio format: flac (default) or wav")
	cmd.Flags().Float64Var(&o.gain, "gain", 1.0, "software gain multiplier, e.g. 2.0 = +6dB (default 1.0)")
	cmd.Flags().IntVar(&o.threads, "threads", 2, "ONNX threads")
	cmd.Flags().BoolVar(&o.keepAudio, "keep-audio", true, "keep intermediate audio after transcription")
	cmd.Flags().BoolVar(&o.keepAudio, "keep-wav", true, "alias of --keep-audio")
	_ = cmd.Flags().MarkDeprecated("keep-wav", "use --keep-audio instead")
	cmd.Flags().StringVar(&o.encoder, "encoder", "", "override: path to encoder.int8.onnx")
	cmd.Flags().StringVar(&o.decoder, "decoder", "", "override: path to decoder.int8.onnx")
	cmd.Flags().StringVar(&o.joiner, "joiner", "", "override: path to joiner.int8.onnx")
	cmd.Flags().StringVar(&o.tokens, "tokens", "", "override: path to tokens.txt")
}

func newRecordAndTranscribeCmd() *cobra.Command {
	var o combinedOptions

	cmd := &cobra.Command{
		Use:     "record-and-transcribe",
		Aliases: []string{"run", "default"},
		Short:   "Record until Enter, then transcribe (default flow, FLAC)",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecordAndTranscribe(o)
		},
	}
	addCombinedFlags(cmd, &o)
	return cmd
}
