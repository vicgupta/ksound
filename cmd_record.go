package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

func recordToFile(audioOut, device, format string, gain float64) (string, error) {
	format, err := NormalizeAudioFormat(format)
	if err != nil {
		return "", err
	}
	if gain <= 0 || gain > 20 {
		return "", fmt.Errorf("invalid --gain %v (want 0 < gain <= 20)", gain)
	}
	samples, err := RecordUntilEnter(device)
	if err != nil {
		return "", err
	}
	if len(samples) == 0 {
		return "", fmt.Errorf("no audio captured")
	}
	if clipped := ApplyGain(samples, gain); clipped > 0 {
		pct := 100 * float64(clipped) / float64(len(samples))
		fmt.Printf("Warning: %.1f%% of samples clipped at gain %.2f — lower --gain\n", pct, gain)
	}
	out, err := EnsureAudioPath(audioOut, format)
	if err != nil {
		return "", err
	}
	if err := SaveAudio(out, format, samples); err != nil {
		return "", err
	}
	dur := float64(len(samples)) / float64(targetSampleRate)
	fmt.Printf("Saved %s (%.1fs, %d samples)\n", out, dur, len(samples))
	return out, nil
}

func newRecordCmd() *cobra.Command {
	var output string
	var device string
	var format string
	var gain float64

	cmd := &cobra.Command{
		Use:   "record",
		Short: "Record from microphone until Enter, save as FLAC (default) or WAV",
		RunE: func(cmd *cobra.Command, args []string) error {
			out, err := recordToFile(output, device, format, gain)
			if err != nil {
				return err
			}
			fmt.Printf("Next: ksound transcribe %s\n", out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "output audio path (default recordings/YYYY-MM-DD_HH-MM-SS.flac)")
	cmd.Flags().StringVar(&device, "device", "", "capture device name substring (default: system default)")
	cmd.Flags().StringVar(&format, "format", FormatFLAC, "audio format: flac (default) or wav")
	cmd.Flags().Float64Var(&gain, "gain", 1.0, "software gain multiplier, e.g. 2.0 = +6dB (default 1.0)")
	return cmd
}
