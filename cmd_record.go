package main

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

func recordToFile(audioOut, device, format string, formatSet bool, gain float64, maxDur time.Duration) (string, error) {
	if gain <= 0 || gain > 20 {
		return "", fmt.Errorf("invalid --gain %v (want 0 < gain <= 20)", gain)
	}
	out, effFormat, err := EnsureAudioPath(audioOut, format, formatSet)
	if err != nil {
		return "", err
	}
	partial := out + ".partial.pcm"
	samples, recordErr := RecordUntilStop(device, maxDur, partial)
	if len(samples) == 0 {
		if recordErr != nil {
			return "", recordErr
		}
		return "", fmt.Errorf("no audio captured")
	}
	if recordErr != nil {
		fmt.Fprintf(os.Stderr, "warning: periodic recovery write failed: %v; saving captured audio normally\n", recordErr)
	}
	if clipped := ApplyGain(samples, gain); clipped > 0 {
		pct := 100 * float64(clipped) / float64(len(samples))
		fmt.Printf("Warning: %.1f%% of samples clipped at gain %.2f — lower --gain\n", pct, gain)
	}
	if err := SaveAudio(out, effFormat, samples); err != nil {
		return "", err
	}
	if err := os.Remove(partial); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "warning: could not remove partial recording %s: %v\n", partial, err)
	}
	dur := float64(len(samples)) / float64(targetSampleRate)
	fmt.Printf("Saved %s (%.1fs, %d samples)\n", out, dur, len(samples))
	return out, nil
}

// recoverLeftovers converts crash-left *.partial.pcm files in recordings/ to
// FLAC so an interrupted recording is never lost.
func recoverLeftovers() {
	if _, err := RecoverPartialRecordings("recordings"); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not recover partial recordings: %v\n", err)
	}
}

func newRecordCmd() *cobra.Command {
	var output string
	var device string
	var format string
	var gain float64
	var duration time.Duration

	cmd := &cobra.Command{
		Use:   "record",
		Short: "Record from microphone until Enter, save as FLAC (default) or WAV",
		RunE: func(cmd *cobra.Command, args []string) error {
			recoverLeftovers()
			out, err := recordToFile(output, device, format, cmd.Flags().Changed("format"), gain, duration)
			if err != nil {
				return err
			}
			fmt.Printf("Next: ksound transcribe %s\n", out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "output audio path (default recordings/YYYY-MM-DD_HH-MM-SS.flac)")
	cmd.Flags().StringVar(&device, "device", "", "capture device name substring (default: system default)")
	cmd.Flags().StringVar(&format, "format", FormatFLAC, "audio format: flac or wav")
	cmd.Flags().Float64Var(&gain, "gain", 1.0, "software gain multiplier, e.g. 2.0 = +6dB")
	cmd.Flags().DurationVar(&duration, "duration", 0, "max recording duration (e.g. 30s); 0 = until Enter/Ctrl+C")
	return cmd
}
