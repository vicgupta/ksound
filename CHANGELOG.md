# Changelog

## [Unreleased]

### Added
- Add recording duration limits and recovery of interrupted recordings.
- Add Markdown transcript output with `--transcript-format markdown`.
- Add a README with build instructions, usage examples, and model-cache details.

### Changed
- Download and validate speech models atomically; automatically replace incomplete caches.
- Resolve FLAC/WAV output format from the file extension and reject conflicting explicit flags.
- Download the speech model before recording in the combined flow.

## [0.1.0] - 2026-10-07

### Added
- Split long audio into speech segments with Silero VAD for transcription.
- Copy completed transcripts to the system clipboard when supported.

### Changed
- Display the full transcript directly in the terminal instead of a width-limited box.
