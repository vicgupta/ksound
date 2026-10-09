# ksound

`ksound` is a cross-platform command-line tool for recording microphone audio and transcribing it locally with NVIDIA Parakeet TDT 0.6b v2 (int8) through sherpa-onnx. Audio is processed on your machine; the first transcription downloads the model files.

Supports macOS, Linux, and Windows. English speech recognition runs on CPU. Microphone access may need to be granted by your operating system.

## Build

Requires Go 1.27.1 or newer.

```sh
go build -o ksound .
./ksound --help
```

Run the test and static-analysis checks with:

```sh
go test ./...
go vet ./...
```

## Quick start

Record, then transcribe:

```sh
./ksound
```

Press Enter or Ctrl+C to stop. The default flow saves audio as `recordings/<timestamp>.flac`, writes a plain-text transcript beside it, prints the transcript, and copies it to the system clipboard when a clipboard utility is available.

On first transcription, ksound downloads the Parakeet model (about 630 MB) before recording starts. It is cached under the operating system's user cache directory in `ksound/models`; an incomplete model cache is detected and downloaded again.

## Commands and examples

Record audio without transcribing it:

```sh
./ksound record --duration 30s --device "MacBook" -o recordings/demo.wav
```

Transcribe an existing FLAC or WAV file:

```sh
./ksound transcribe recordings/demo.wav
```

Write a Markdown transcript:

```sh
./ksound transcribe recordings/demo.wav --transcript-format markdown -o notes.md
```

Run the combined flow with a fixed duration and explicit output paths:

```sh
./ksound record-and-transcribe \
  --duration 2m \
  --audio-out recordings/meeting.wav \
  --output meeting.md \
  --transcript-format markdown
```

List available microphone devices:

```sh
./ksound list-devices
./ksound list-devices --json
```

Write the transcript to stdout instead of a file:

```sh
./ksound transcribe recordings/demo.wav -o -
```

Tune VAD segmentation for noisy audio and set threads explicitly:

```sh
./ksound transcribe long.wav --vad-threshold 0.7 --vad-min-silence 0.3 --threads 4
```

Use `./ksound <command> --help` to see all options.

## Recording behavior

- Recording stops on Enter, Ctrl+C/SIGINT, SIGTERM, or when `--duration` elapses. For example: `--duration 90s` or `--duration 5m`.
- If stdin is not an interactive terminal, EOF does not stop recording; pass `--duration` or stop it with Ctrl+C/SIGTERM.
- Audio is recorded as mono, 16 kHz, 16-bit PCM and saved as FLAC by default. WAV is also supported.
- A `.flac` or `.wav` output extension selects the format. If `--format` is explicitly supplied and disagrees with the extension, ksound reports an error. If there is no extension, ksound appends the selected format's extension. Other extensions are rejected.
- `--gain` applies software gain (default `1.0`); values above `1.0` can clip and produce a warning.
- During recording, ksound periodically writes a recovery copy. On the next startup, it scans `recordings/` for leftover `*.partial.pcm` files and converts them to `*.recovered.flac`.

## Transcription and models

- Transcript output defaults to plain text (`.txt`). Use `--transcript-format markdown` for a Markdown file (`.md`) with the audio source and transcription time.
- Completed transcript text is printed to the terminal and copied to the clipboard when a supported system utility is available. Clipboard support is best-effort; transcription still succeeds if no utility is installed. Linux utilities checked are `wl-copy`, `xclip`, and `xsel`.
- Audio is transcribed locally. The Parakeet model is downloaded once and cached in the user cache directory.
- For audio longer than five minutes, ksound downloads the Silero VAD model (about 2 MB) on first use and splits speech into shorter segments for transcription. Segmentation is tunable via `--vad-threshold`, `--vad-min-silence`, `--vad-min-speech`, and `--vad-max-speech`.
- The model tarball is cached next to the model directory, so an interrupted download resumes (HTTP Range) instead of starting over, and the archive is verified against `KSOUND_MODEL_SHA256` when set (otherwise a trust-on-first-use `.sha256` sidecar pinned after the first successful download). Downloads retry up to 3 times, validate sizes, reject truncated streams, and all progress goes to stderr so `-o -` piping works. Optional per-file SHA256 verification via `KSOUND_ENCODER_SHA256`, `KSOUND_DECODER_SHA256`, `KSOUND_JOINER_SHA256`, `KSOUND_TOKENS_SHA256`, and `KSOUND_VAD_SHA256`.
- `--threads 0` (default) picks a sensible value from `runtime.NumCPU` (capped at 8).
- If model initialization fails because the cache is corrupt, the error message points to the cache directory to remove before retrying.
- `./ksound --version` prints the build version (bake in with `go build -ldflags "-X main.version=1.2.3"`).

## Development

The repository uses Go modules. The main checks are:

```sh
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

CI runs these on Linux, macOS, and Windows (with a coverage gate on Linux).
Pushing a tag matching `v*` triggers the release workflow, which builds
bundled binaries for Linux (amd64), macOS (arm64/amd64), and Windows (amd64)
— each archive contains the shared sherpa-onnx libraries next to the binary —
and publishes them with a `SHA256SUMS` file.
