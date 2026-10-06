package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ksound",
	Short: "Record audio and transcribe it locally (Parakeet via sherpa-onnx)",
	Long: `ksound records from the default microphone and transcribes offline
using NVIDIA Parakeet TDT 0.6b v2 (int8) via sherpa-onnx. English, plain text.
Runs on macOS, Linux and Windows (mic permission required on first run).

Default (no subcommand): record until Enter, then transcribe.
  ksound
  ksound -o note.txt
  ksound --format wav --device "MacBook" --keep-audio=false`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runRecordAndTranscribe(defaultCombinedOptions)
	},
}

var defaultCombinedOptions combinedOptions

func init() {
	addCombinedFlags(rootCmd, &defaultCombinedOptions)
}

func main() {
	rootCmd.AddCommand(newRecordAndTranscribeCmd())
	rootCmd.AddCommand(newRecordCmd())
	rootCmd.AddCommand(newTranscribeCmd())
	rootCmd.AddCommand(newDevicesCmd())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
