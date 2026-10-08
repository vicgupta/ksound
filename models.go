package main

import (
	"archive/tar"
	"compress/bzip2"
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
	// Silero VAD (~2MB) used to split long recordings into speech
	// segments, since the Parakeet encoder cannot decode very long
	// audio in one shot.
	vadModelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

	// completeMarker is written last, once every model file is present and
	// validated. A directory without it is treated as an incomplete cache.
	completeMarker = ".complete"

	// Minimum plausible file sizes, used to detect truncated downloads.
	minEncoderBytes = int64(500 * 1024 * 1024)
	minDecoderBytes = int64(1 * 1024 * 1024)
	minJoinerBytes  = int64(1 * 1024 * 1024)
	minTokensBytes  = int64(1 * 1024)
	minVadBytes     = int64(1 * 1024 * 1024)
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
		fmt.Fprintln(os.Stderr, "Model cache incomplete, re-downloading...")
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
	fmt.Printf("Downloading Parakeet model (~630MB) to %s ...\n", dir)
	if err := downloadAndExtract(modelURL, tmpDir); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, err
	}
	if err := prepareModelTempDir(tmpDir, parakeetMinSizes()); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, fmt.Errorf("model download incomplete: %w", err)
	}
	if err := os.Rename(tmpDir, dir); err != nil {
		_ = os.RemoveAll(tmpDir)
		return ModelFiles{}, err
	}
	fmt.Println("Model ready.")
	return m, nil
}

// removeStaleTempDirs deletes leftover <dir>.tmp-* directories from a previous
// interrupted download without removing a directory owned by a live process.
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
		pid, err := strconv.Atoi(strings.TrimPrefix(e.Name(), prefix))
		if err == nil && pid > 0 && processIsRunning(pid) {
			continue
		}
		_ = os.RemoveAll(filepath.Join(parent, e.Name()))
	}
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
	fmt.Printf("Downloading Silero VAD model to %s ...\n", path)
	if err := downloadFile(url, tmp); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	st, err := os.Stat(tmp)
	if err != nil || st.IsDir() || st.Size() <= minVadBytes {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("VAD model download incomplete in %s", dir)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	fmt.Println("VAD model ready.")
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
	read    int64
	lastPct int
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.read += int64(n)
	if p.total > 0 {
		if pct := int(p.read * 100 / p.total); pct >= p.lastPct+5 {
			p.lastPct = pct
			fmt.Fprintf(os.Stderr, "  downloading... %d%%\n", pct)
		}
	}
	return n, err
}

func downloadAndExtract(url, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	resp, err := newDownloadClient().Do(req)
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download model: HTTP %s", resp.Status)
	}

	var body io.Reader = resp.Body
	if resp.ContentLength > 0 {
		body = &progressReader{r: resp.Body, total: resp.ContentLength}
	}
	tr := tar.NewReader(bzip2.NewReader(body))
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
		fmt.Printf("  extracted %s\n", name)
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
	var body io.Reader = resp.Body
	if resp.ContentLength > 0 {
		body = &progressReader{r: resp.Body, total: resp.ContentLength}
	}
	if _, err := io.Copy(f, body); err != nil {
		if closeErr := f.Close(); closeErr != nil {
			return fmt.Errorf("download: %w", errors.Join(err, closeErr))
		}
		return fmt.Errorf("download: %w", err)
	}
	return f.Close()
}
