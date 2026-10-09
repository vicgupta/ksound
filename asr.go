package main

import (
	"fmt"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// recognizer decodes mono float32 samples at sampleRate into text. It is the
// seam between the transcription pipeline and the native sherpa-onnx library,
// so the pipeline can be tested with a fake.
type recognizer interface {
	Decode(samples []float32, sampleRate int) (string, error)
	Close()
}

// vadSegment is one detected speech chunk, indexed in samples at the VAD
// analysis rate (16kHz).
type vadSegment struct {
	Start int
	N     int
}

// vad splits mono float32 audio into speech segments. Another seam for tests.
type vad interface {
	AcceptWaveform(samples []float32)
	Flush()
	IsEmpty() bool
	Front() vadSegment
	Pop()
	Close()
}

// --- real sherpa-onnx implementations ---

type sherpaRecognizer struct{ r *sherpa.OfflineRecognizer }

// newSherpaRecognizer builds an offline Parakeet recognizer from ModelFiles.
// Returns modelLoadError when native init fails (corrupt cache, bad paths).
func newSherpaRecognizer(m ModelFiles, threads int) (recognizer, error) {
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

	r := sherpa.NewOfflineRecognizer(&config)
	if r == nil {
		return nil, modelLoadError("load the speech model")
	}
	return &sherpaRecognizer{r: r}, nil
}

func (s *sherpaRecognizer) Decode(samples []float32, sampleRate int) (string, error) {
	stream := sherpa.NewOfflineStream(s.r)
	if stream == nil {
		return "", fmt.Errorf("create decode stream failed")
	}
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(sampleRate, samples)
	s.r.Decode(stream)
	return strings.TrimSpace(stream.GetResult().Text), nil
}

func (s *sherpaRecognizer) Close() { sherpa.DeleteOfflineRecognizer(s.r) }

type sherpaVAD struct {
	v    *sherpa.VoiceActivityDetector
	path string
}

// newSherpaVAD downloads/loads Silero VAD and builds a detector tuned to opts.
func newSherpaVAD(opts vadOptions) (vad, error) {
	vadPath, err := EnsureVadModel()
	if err != nil {
		return nil, err
	}
	cfg := sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              vadPath,
			Threshold:          opts.threshold,
			MinSilenceDuration: opts.minSilence,
			MinSpeechDuration:  opts.minSpeech,
			WindowSize:         vadWindowSize,
			MaxSpeechDuration:  opts.maxSpeech,
		},
		SampleRate: vadAnalysisRate,
		NumThreads: 1,
		Provider:   "cpu",
	}
	v := sherpa.NewVoiceActivityDetector(&cfg, 100)
	if v == nil {
		return nil, modelLoadError(fmt.Sprintf("create the VAD (model: %s)", vadPath))
	}
	return &sherpaVAD{v: v, path: vadPath}, nil
}

func (s *sherpaVAD) AcceptWaveform(samples []float32) { s.v.AcceptWaveform(samples) }
func (s *sherpaVAD) Flush()                           { s.v.Flush() }
func (s *sherpaVAD) IsEmpty() bool                    { return s.v.IsEmpty() }
func (s *sherpaVAD) Pop()                             { s.v.Pop() }
func (s *sherpaVAD) Close()                           { sherpa.DeleteVoiceActivityDetector(s.v) }

func (s *sherpaVAD) Front() vadSegment {
	seg := s.v.Front()
	return vadSegment{Start: seg.Start, N: len(seg.Samples)}
}

const (
	// vadAnalysisRate is the sample rate Silero VAD is trained for.
	vadAnalysisRate = 16000
	// vadWindowSize is the Silero inference window at vadAnalysisRate.
	vadWindowSize = 512
)
