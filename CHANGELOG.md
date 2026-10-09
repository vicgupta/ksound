# Changelog

## [Unreleased]

## [0.2.0] - 2026-10-08

### Added
- Add recording duration limits and recovery of interrupted recordings.
- Add Markdown transcript output with `--transcript-format markdown`.
- Add a README with build instructions, usage examples, and model-cache details.
- Add `--version`, `transcribe -o -` (stdout), `list-devices --json`, and VAD tuning flags (`--vad-threshold`, `--vad-min-silence`, `--vad-min-speech`, `--vad-max-speech`).
- Add optional SHA256 verification via `KSOUND_*_SHA256` env vars, download retries, and truncated-stream detection.
- Add CI workflow (vet, test, race, build) and golangci-lint config.
- Add tests for gain/VAD/threads validation, transcript writing, retry, checksums, and stale-dir expiry.
- Add recognizer/VAD interfaces with a swappable transcription pipeline, fake-backed pipeline tests, and a cobra test harness.
- Add resumable model downloads: the tarball is cached, resumed via HTTP Range, and pinned by SHA256 (`KSOUND_MODEL_SHA256`, falling back to a trust-on-first-use `.sha256` sidecar).
- Add an ubuntu/macos/windows CI matrix with gofmt, golangci-lint, and coverage gates.
- Add a tag-triggered release workflow (native cgo builds with bundled sherpa-onnx libs, rpath rewrite, `SHA256SUMS`) and Dependabot for Go modules and GitHub Actions.
- Detect capture-device stalls (unplug/hot-plug): a recording whose microphone stops delivering audio aborts with a clear error instead of hanging.
- Report transcription speed (wall time and RTF) after every decode.
- Check free disk space before recording: fixed-duration recordings fail fast when they cannot fit; open-ended recordings warn when space is low.

### Changed
- Download and validate speech models atomically; automatically replace incomplete caches.
- Resolve FLAC/WAV output format from the file extension and reject conflicting explicit flags.
- Download the speech model before recording in the combined flow.
- Send all progress/status to stderr; only transcripts and device lists go to stdout.
- Default `--threads` to auto (`runtime.NumCPU`, capped at 8).
- Replace hand-rolled terminal detection with `golang.org/x/term`; unify process checks per-OS.
- Write WAV/FLAC in 64k-sample chunks to bound memory; cap in-RAM capture with a truncation warning.
- Fix capture device ID lifetime and only watch Enter when stdin is a terminal.
- Split pure `ResolveAudioPath` from side-effecting `EnsureAudioPath`.
- Remove unused boxed-output code and deprecated `--wav-out`/`--keep-wav` aliases.
- Validate CLI flags before any download, recording, or other side effect.
- Fall back to the next clipboard utility when one exists but fails at runtime (e.g. xclip without an X display).

## [0.1.0] - 2026-10-07

### Added
- Split long audio into speech segments with Silero VAD for transcription.
- Copy completed transcripts to the system clipboard when supported.

### Changed
- Display the full transcript directly in the terminal instead of a width-limited box.
