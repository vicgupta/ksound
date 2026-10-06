package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

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

func transcribeFile(audioPath, txtOut string, threads int, m ModelFiles) (string, error) {
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

	fmt.Println("Loading model (first run takes a while)...")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)
	recognizer.Decode(stream)
	result := stream.GetResult()
	text := strings.TrimSpace(result.Text)

	out := txtOut
	if out == "" {
		ext := filepath.Ext(audioPath)
		out = strings.TrimSuffix(audioPath, ext) + ".txt"
	}
	if err := os.WriteFile(out, []byte(text+"\n"), 0o644); err != nil {
		return "", err
	}
	fmt.Printf("Transcription written to %s\n", out)
	if text == "" {
		fmt.Println("(empty result — input may be silent or too short)")
		return out, nil
	}
	printBoxed("Transcription", text)
	return out, nil
}

func newTranscribeCmd() *cobra.Command {
	var output string
	var threads int
	var encoder, decoder, joiner, tokens string

	cmd := &cobra.Command{
		Use:   "transcribe <file.flac|file.wav>",
		Short: "Transcribe a FLAC/WAV file locally, write plain-text .txt",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := resolveModel(encoder, decoder, joiner, tokens)
			if err != nil {
				return err
			}
			_, err = transcribeFile(args[0], output, threads, m)
			return err
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "output text path (default: <input>.txt)")
	cmd.Flags().IntVar(&threads, "threads", 2, "ONNX threads")
	cmd.Flags().StringVar(&encoder, "encoder", "", "override: path to encoder.int8.onnx")
	cmd.Flags().StringVar(&decoder, "decoder", "", "override: path to decoder.int8.onnx")
	cmd.Flags().StringVar(&joiner, "joiner", "", "override: path to joiner.int8.onnx")
	cmd.Flags().StringVar(&tokens, "tokens", "", "override: path to tokens.txt")
	return cmd
}
