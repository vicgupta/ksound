package main

import (
	"archive/tar"
	"compress/bzip2"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	modelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
	modelDir = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
	// Silero VAD (~0.6MB; upstream shrank it from ~2MB) used to split
	// long recordings into speech segments, since the Parakeet encoder
	// cannot decode very long audio in one shot.
	vadModelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

	// completeMarker is written last, once every model file is present and
	// validated. A directory without it is treated as an incomplete cache.
	completeMarker = ".complete"

	// modelArchiveName is the release tarball cached next to the model dir
	// so a failed/interrupted download can resume instead of restarting.
	modelArchiveName = modelDir + ".tar.bz2"

	// Minimum plausible file sizes, used to detect truncated downloads.
	minEncoderBytes = int64(500 * 1024 * 1024)
	minDecoderBytes = int64(1 * 1024 * 1024)
	minJoinerBytes  = int64(1 * 1024 * 1024)
	minTokensBytes  = int64(1 * 1024)
	minVadBytes     = int64(100 * 1024)

	// staleTempDirMaxAge bounds PID-reuse risk: a tmp dir older than this is
	// removed even if its PID appears live.
	staleTempDirMaxAge = 24 * time.Hour

	// downloadMaxAttempts caps transient-error retries for model fetches.
	downloadMaxAttempts = 3
)

// ModelFiles holds resolved ONNX model paths.
type ModelFiles struct {
	Encoder string
	Decoder string
	Joiner  string
	Tokens  string
}

// ModelCacheDir returns ~/.cache/ksound/models/<modelDir>.
func ModelCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "ksound", "models", modelDir)
	return dir, nil
}

// modelFilesFor returns the expected model paths inside dir.
func modelFilesFor(dir string) ModelFiles {
	return ModelFiles{
		Encoder: filepath.Join(dir, "encoder.int8.onnx"),
		Decoder: filepath.Join(dir, "decoder.int8.onnx"),
		Joiner:  filepath.Join(dir, "joiner.int8.onnx"),
		Tokens:  filepath.Join(dir, "tokens.txt"),
	}
}

// parakeetMinSizes maps each required Parakeet file to its minimum sane size.
func parakeetMinSizes() map[string]int64 {
	return map[string]int64{
		"encoder.int8.onnx": minEncoderBytes,
		"decoder.int8.onnx": minDecoderBytes,
		"joiner.int8.onnx":  minJoinerBytes,
		"tokens.txt":        minTokensBytes,
	}
}

// validateModelFiles checks that every file in minSizes exists in dir and is
// at least the given size. Unlike validateModelDir it does not require the
// .complete marker, so a freshly extracted download can be validated before
// the marker is written.
func validateModelFiles(dir string, minSizes map[string]int64) error {
	for name, min := range minSizes {
		st, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("missing %s: %w", name, err)
		}
		if st.IsDir() {
			return fmt.Errorf("%s is a directory", name)
		}
		if st.Size() <= min {
			return fmt.Errorf("%s too small: %d bytes must be greater than %d", name, st.Size(), min)
		}
	}
	return nil
}

// validateModelDir reports why dir is not a complete model directory: the
// .complete marker must be present and every file in minSizes must exist and
// be at least the given size.
func validateModelDir(dir string, minSizes map[string]int64) error {
	if st, err := os.Stat(filepath.Join(dir, completeMarker)); err != nil || st.IsDir() {
		return fmt.Errorf("missing %s marker", completeMarker)
	}
	return validateModelFiles(dir, minSizes)
}

// prepareModelTempDir validates extracted files and writes the completion
// marker only after every required file passes validation.
func prepareModelTempDir(dir string, minSizes map[string]int64) error {
	if err := validateModelFiles(dir, minSizes); err != nil {
		return err
	}
	marker, err := os.OpenFile(filepath.Join(dir, completeMarker), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := marker.Write([]byte("complete\n")); err != nil {
		_ = marker.Close()
		return err
	}
	if err := marker.Close(); err != nil {
		return err
	}
	return validateModelDir(dir, minSizes)
}

// EnsureModel downloads and extracts the Parakeet model on first run. The
// download lands in a temporary sibling directory and is renamed into place
// only once every file has been extracted and validated, so an interrupted
// download can never leave a corrupt cache behind.
func EnsureModel() (ModelFiles, error) {
	dir, err := ModelCacheDir()
	if err != nil {
		return ModelFiles{}, err
	}
	removeStaleTempDirs(dir)

	m := modelFilesFor(dir)
	if validateModelDir(dir, parakeetMinSizes()) == nil {
		return m, nil
	}
	if _, statErr := os.Stat(dir); statErr == nil {
		infof("Model cache incomplete, re-downloading...")
		if err := os.RemoveAll(dir); err != nil {
			return ModelFiles{}, err
		}
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return ModelFiles{}, err
	}

	tmpDir := fmt.Sprintf("%s.tmp-%d", dir, os.Getpid())
	if err := os.RemoveAll(tmpDir); err != nil {
		return ModelFiles{}, err
	}
	archive := filepath.Join(filepath.Dir(dir), modelArchiveName)
	infof("Downloading Parakeet model (~630MB, resumable)...")
	if err := ensureModelArchive(modelURL, archive); err != nil {
		return ModelFiles{}, err
	}
	infof("Extracting model archive...")
	if err := extractModelArchive(archive, tmpDir); err != nil {
		_ = os.RemoveAll(tmpDir)
		// Keep the verified archive so a retry skips the download.
		return ModelFiles{}, err
	}
	if err := prepareModelTempDir(tmpDir, parakeetMinSizes()); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, fmt.Errorf("model download incomplete: %w", err)
	}
	if err := verifyModelChecksums(tmpDir); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, err
	}
	if err := os.Rename(tmpDir, dir); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, err
	}
	// Cache is complete; free the archive. The .sha256 sidecar stays behind
	// as the integrity pin for any future re-download.
	_ = os.Remove(archive)
	infof("Model ready.")
	return m, nil
}

// removeStaleTempDirs deletes leftover <dir>.tmp-* directories from a previous
// interrupted download without removing a directory owned by a live process.
// Directories older than staleTempDirMaxAge are removed regardless, since PIDs
// can be recycled by the OS.
func removeStaleTempDirs(dir string) {
	parent := filepath.Dir(dir)
	prefix := filepath.Base(dir) + ".tmp-"
	entries, err := os.ReadDir(parent)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), prefix) {
			continue
		}
		full := filepath.Join(parent, e.Name())
		pid, err := strconv.Atoi(strings.TrimPrefix(e.Name(), prefix))
		if err == nil && pid > 0 && processIsRunning(pid) {
			if st, statErr := os.Stat(full); statErr == nil {
				if time.Since(st.ModTime()) < staleTempDirMaxAge {
					continue
				}
			} else {
				continue
			}
		}
		_ = os.RemoveAll(full)
	}
}

// withRetry runs fn up to attempts times with linear backoff for transient
// network failures.
func withRetry(attempts int, fn func() error) error {
	var err error
	for i := 1; i <= attempts; i++ {
		if err = fn(); err == nil {
			return nil
		}
		if i < attempts {
			infof("download attempt %d/%d failed: %v; retrying...", i, attempts, err)
			time.Sleep(time.Duration(i) * 200 * time.Millisecond)
		}
	}
	return err
}

// verifyModelChecksums optionally verifies extracted model files against
// hex-encoded SHA256 hashes from the environment:
// KSOUND_ENCODER_SHA256, KSOUND_DECODER_SHA256, KSOUND_JOINER_SHA256,
// KSOUND_TOKENS_SHA256. Unset variables are skipped.
func verifyModelChecksums(dir string) error {
	envFor := map[string]string{
		"encoder.int8.onnx": "KSOUND_ENCODER_SHA256",
		"decoder.int8.onnx": "KSOUND_DECODER_SHA256",
		"joiner.int8.onnx":  "KSOUND_JOINER_SHA256",
		"tokens.txt":        "KSOUND_TOKENS_SHA256",
	}
	for name, env := range envFor {
		want := strings.TrimSpace(os.Getenv(env))
		if want == "" {
			continue
		}
		if err := verifyFileSHA256(filepath.Join(dir, name), want); err != nil {
			return fmt.Errorf("%s checksum: %w", name, err)
		}
	}
	return nil
}

// fileSHA256 returns the hex SHA256 of the file at path.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// verifyFileSHA256 checks path against a hex-encoded SHA256.
func verifyFileSHA256(path, wantHex string) error {
	want := strings.TrimSpace(wantHex)
	if b, err := hex.DecodeString(want); err != nil || len(b) != sha256.Size {
		return fmt.Errorf("invalid SHA256 %q", wantHex)
	}
	got, err := fileSHA256(path)
	if err != nil {
		return err
	}
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("mismatch (got %s)", got)
	}
	return nil
}

// resolveModelSHA256 returns the expected archive hash: the explicit
// KSOUND_MODEL_SHA256 pin if set, otherwise the trust-on-first-use sidecar
// written after a previous verified download.
func resolveModelSHA256(archive string) string {
	if h := strings.TrimSpace(os.Getenv("KSOUND_MODEL_SHA256")); h != "" {
		return h
	}
	if b, err := os.ReadFile(archive + ".sha256"); err == nil {
		return strings.TrimSpace(string(b))
	}
	return ""
}

// EnsureVadModel downloads the Silero VAD model on first use and returns
// its path. Used to split long recordings into transcribable segments.
func EnsureVadModel() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return ensureVadModel(filepath.Join(base, "ksound", "models"), vadModelURL)
}

// ensureVadModel downloads the Silero VAD model into dir if absent, writing to
// a temporary file first so a partial download never lands at the final path.
func ensureVadModel(dir, url string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "silero_vad.onnx")
	if st, err := os.Stat(path); err == nil {
		if !st.IsDir() && st.Size() > minVadBytes {
			if want := strings.TrimSpace(os.Getenv("KSOUND_VAD_SHA256")); want != "" {
				if err := verifyFileSHA256(path, want); err != nil {
					return "", fmt.Errorf("existing VAD model checksum: %w", err)
				}
			}
			return path, nil
		}
		if err := os.Remove(path); err != nil {
			return "", fmt.Errorf("remove incomplete VAD model %s: %w", path, err)
		}
	} else if !os.IsNotExist(err) {
		return "", err
	}
	tmp := path + ".tmp"
	_ = os.Remove(tmp)
	infof("Downloading Silero VAD model to %s ...", path)
	if err := withRetry(downloadMaxAttempts, func() error { return downloadFile(url, tmp) }); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	st, err := os.Stat(tmp)
	if err != nil || st.IsDir() || st.Size() <= minVadBytes {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("VAD model download incomplete in %s", dir)
	}
	if want := strings.TrimSpace(os.Getenv("KSOUND_VAD_SHA256")); want != "" {
		if err := verifyFileSHA256(tmp, want); err != nil {
			_ = os.Remove(tmp)
			return "", fmt.Errorf("VAD model checksum: %w", err)
		}
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	infof("VAD model ready.")
	return path, nil
}

// newDownloadClient returns an HTTP client with connect/handshake/header
// timeouts but no overall timeout, since model files are large.
func newDownloadClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
			ExpectContinueTimeout: 10 * time.Second,
		},
	}
}

// progressReader prints download progress to stderr every ~5%.
type progressReader struct {
	r       io.Reader
	total   int64
	base    int64 // bytes already on disk before this transfer
	read    int64
	lastPct int
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.read += int64(n)
	if p.total > 0 {
		if pct := int((p.base + p.read) * 100 / p.total); pct >= p.lastPct+5 {
			p.lastPct = pct
			fmt.Fprintf(os.Stderr, "  downloading... %d%%\n", pct)
		}
	}
	return n, err
}

// errRangeRejected reports that the server refused our resume range; the
// caller should discard the partial file and start over.
var errRangeRejected = errors.New("server rejected resume range")

// resumeDownload fetches url into dest, appending to a partial file when the
// server honours HTTP Range. It errors when the result is shorter than the
// advertised length, so a truncated transfer is never mistaken for a complete
// archive.
func resumeDownload(url, dest string) error {
	var offset int64
	if st, err := os.Stat(dest); err == nil && !st.IsDir() {
		offset = st.Size()
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := newDownloadClient().Do(req)
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	defer resp.Body.Close()

	appendMode := false
	var want int64 // expected total size afterwards; <=0 means unknown
	switch resp.StatusCode {
	case http.StatusPartialContent:
		appendMode = true
		want = offset + resp.ContentLength
	case http.StatusOK:
		// Server ignored Range (or we asked from 0): full body, restart.
		offset = 0
		want = resp.ContentLength
	case http.StatusRequestedRangeNotSatisfiable:
		return errRangeRejected
	default:
		return fmt.Errorf("download model: HTTP %s", resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(dest, flags, 0o644)
	if err != nil {
		return err
	}
	pr := &progressReader{r: resp.Body, total: want, base: offset}
	if _, err := io.Copy(f, pr); err != nil {
		_ = f.Close()
		return fmt.Errorf("download model: %w", err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	if want > 0 {
		st, err := os.Stat(dest)
		if err != nil {
			return err
		}
		if st.Size() < want {
			return fmt.Errorf("download model: truncated (%d of %d bytes)", st.Size(), want)
		}
	}
	return nil
}

// ensureModelArchive downloads the model tarball to archive with resume
// support. The archive is verified against KSOUND_MODEL_SHA256 when set,
// otherwise against a trust-on-first-use .sha256 sidecar recorded after a
// previous complete download. On mismatch the archive is discarded and the
// download is retried from scratch.
func ensureModelArchive(url, archive string) error {
	// Fast path: a fully verified archive needs no HTTP at all.
	if st, err := os.Stat(archive); err == nil && !st.IsDir() && st.Size() > 0 {
		if expected := resolveModelSHA256(archive); expected != "" {
			if verifyFileSHA256(archive, expected) == nil {
				return nil
			}
			infof("Cached model archive failed checksum, re-downloading...")
			if err := os.Remove(archive); err != nil {
				return err
			}
		}
	}

	var lastErr error
	for attempt := 1; attempt <= downloadMaxAttempts; attempt++ {
		if attempt > 1 {
			time.Sleep(time.Duration(attempt) * 200 * time.Millisecond)
		}
		if err := resumeDownload(url, archive); err != nil {
			if errors.Is(err, errRangeRejected) {
				// Cannot confirm the existing copy; start clean.
				_ = os.Remove(archive)
				err = fmt.Errorf("resume rejected, restarting download")
			}
			lastErr = err
			infof("download attempt %d/%d failed: %v; retrying...", attempt, downloadMaxAttempts, err)
			continue
		}
		got, err := fileSHA256(archive)
		if err != nil {
			lastErr = err
			continue
		}
		if expected := resolveModelSHA256(archive); expected != "" {
			if !strings.EqualFold(expected, got) {
				_ = os.Remove(archive)
				lastErr = fmt.Errorf("model archive checksum mismatch (got %s)", got)
				infof("download attempt %d/%d failed: %v; retrying...", attempt, downloadMaxAttempts, lastErr)
				continue
			}
			return nil
		}
		// Trust on first use: record the hash of this verified-complete
		// archive so future re-downloads are pinned.
		if err := os.WriteFile(archive+".sha256", []byte(got+"\n"), 0o644); err != nil {
			infof("warning: could not record model checksum: %v", err)
		}
		return nil
	}
	return lastErr
}

// extractModelArchive unpacks the verified bzip2 tar archive into destDir,
// keeping only the model files ksound needs.
func extractModelArchive(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open model archive: %w", err)
	}
	defer f.Close()
	return extractTar(bzip2.NewReader(f), destDir)
}

// extractTar unpacks a tar stream of model files into destDir, flattening
// the archive's top-level directory and skipping everything ksound does not
// need. All four required files must be present.
func extractTar(r io.Reader, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	tr := tar.NewReader(r)
	extracted := map[string]bool{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("extract model: %w", err)
		}
		// Flatten: archive contains top-level dir; we want files directly in destDir.
		name := filepath.Base(hdr.Name)
		if hdr.FileInfo().IsDir() || name == "." || name == "/" {
			continue
		}
		// Only keep files we need (skip test_wavs to save time/space).
		switch name {
		case "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt":
		default:
			continue
		}
		out := filepath.Join(destDir, name)
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			if closeErr := f.Close(); closeErr != nil {
				return fmt.Errorf("extract %s: %w", name, errors.Join(err, closeErr))
			}
			return fmt.Errorf("extract %s: %w", name, err)
		}
		if err := f.Close(); err != nil {
			return err
		}
		extracted[name] = true
		infof("  extracted %s", name)
	}
	for _, want := range []string{"encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"} {
		if !extracted[want] {
			return fmt.Errorf("extract model: missing %s in archive", want)
		}
	}
	return nil
}

// downloadFile streams url into dest, returning any error. The caller is
// responsible for cleaning up dest on error.
func downloadFile(url, dest string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := newDownloadClient().Do(req)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: HTTP %s", resp.Status)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	pr := &progressReader{r: resp.Body, total: resp.ContentLength}
	if _, err := io.Copy(f, pr); err != nil {
		if closeErr := f.Close(); closeErr != nil {
			return fmt.Errorf("download: %w", errors.Join(err, closeErr))
		}
		return fmt.Errorf("download: %w", err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	if pr.total > 0 && pr.read < pr.total {
		return fmt.Errorf("download: truncated (%d of %d bytes)", pr.read, pr.total)
	}
	return nil
}
