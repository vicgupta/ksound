package main

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func writeSizedFile(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestValidateModelDir(t *testing.T) {
	minSizes := map[string]int64{"encoder.onnx": 100, "tokens.txt": 10}

	t.Run("ok", func(t *testing.T) {
		dir := t.TempDir()
		writeSizedFile(t, filepath.Join(dir, completeMarker), 1)
		writeSizedFile(t, filepath.Join(dir, "encoder.onnx"), 101)
		writeSizedFile(t, filepath.Join(dir, "tokens.txt"), 11)
		if err := validateModelDir(dir, minSizes); err != nil {
			t.Fatalf("validateModelDir: %v", err)
		}
	})

	t.Run("missing marker", func(t *testing.T) {
		dir := t.TempDir()
		writeSizedFile(t, filepath.Join(dir, "encoder.onnx"), 101)
		writeSizedFile(t, filepath.Join(dir, "tokens.txt"), 11)
		if err := validateModelDir(dir, minSizes); err == nil {
			t.Fatal("expected missing marker error")
		}
	})

	t.Run("truncated file", func(t *testing.T) {
		dir := t.TempDir()
		writeSizedFile(t, filepath.Join(dir, completeMarker), 1)
		writeSizedFile(t, filepath.Join(dir, "encoder.onnx"), 50)
		writeSizedFile(t, filepath.Join(dir, "tokens.txt"), 11)
		if err := validateModelDir(dir, minSizes); err == nil {
			t.Fatal("expected truncated file error")
		}
	})

	t.Run("missing file", func(t *testing.T) {
		dir := t.TempDir()
		writeSizedFile(t, filepath.Join(dir, completeMarker), 1)
		writeSizedFile(t, filepath.Join(dir, "encoder.onnx"), 101)
		if err := validateModelDir(dir, minSizes); err == nil {
			t.Fatal("expected missing file error")
		}
	})
}

func TestPrepareModelTempDirWritesMarkerLast(t *testing.T) {
	dir := t.TempDir()
	minSizes := map[string]int64{"encoder.onnx": 100, "tokens.txt": 10}
	writeSizedFile(t, filepath.Join(dir, "encoder.onnx"), 101)
	writeSizedFile(t, filepath.Join(dir, "tokens.txt"), 11)

	if err := prepareModelTempDir(dir, minSizes); err != nil {
		t.Fatalf("prepareModelTempDir: %v", err)
	}
	if err := validateModelDir(dir, minSizes); err != nil {
		t.Fatalf("prepared directory is not complete: %v", err)
	}
}

func TestRemoveStaleTempDirs(t *testing.T) {
	parent := t.TempDir()
	modelPath := filepath.Join(parent, "model")
	stale := modelPath + ".tmp-2147483647"
	live := modelPath + ".tmp-" + strconv.Itoa(os.Getpid())
	other := modelPath + ".tmp-file"
	for _, path := range []string{stale, live} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(other, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}

	removeStaleTempDirs(modelPath)
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale directory should be removed, err=%v", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("live download directory should remain: %v", err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatalf("non-directory temp-named file should remain: %v", err)
	}
}

func TestEnsureVadModelDownload(t *testing.T) {
	body := make([]byte, minVadBytes+1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	path, err := ensureVadModel(dir, srv.URL)
	if err != nil {
		t.Fatalf("ensureVadModel: %v", err)
	}
	if path != filepath.Join(dir, "silero_vad.onnx") {
		t.Fatalf("path = %q", path)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Size() != int64(len(body)) {
		t.Fatalf("size = %d, want %d", st.Size(), len(body))
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file should be gone, err=%v", err)
	}
}

func TestEnsureVadModelHTTPErrorLeavesNoFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	dir := t.TempDir()
	writeSizedFile(t, filepath.Join(dir, "silero_vad.onnx"), int(minVadBytes-1))
	if _, err := ensureVadModel(dir, srv.URL); err == nil {
		t.Fatal("expected HTTP error")
	}
	assertNoVadFiles(t, dir)
}

func TestEnsureVadModelTruncatedLeavesNoFile(t *testing.T) {
	body := make([]byte, minVadBytes+1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		_, _ = w.Write(body[:100])
	}))
	defer srv.Close()

	dir := t.TempDir()
	if _, err := ensureVadModel(dir, srv.URL); err == nil {
		t.Fatal("expected truncated download error")
	}
	assertNoVadFiles(t, dir)
}

func assertNoVadFiles(t *testing.T, dir string) {
	t.Helper()
	for _, name := range []string{"silero_vad.onnx", "silero_vad.onnx.tmp"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatalf("%s should not exist, err=%v", name, err)
		}
	}
}

func TestWithRetrySucceedsAfterFailure(t *testing.T) {
	calls := 0
	err := withRetry(3, func() error {
		calls++
		if calls < 2 {
			return os.ErrInvalid
		}
		return nil
	})
	if err != nil || calls != 2 {
		t.Fatalf("withRetry err=%v calls=%d, want nil/2", err, calls)
	}
}

func TestWithRetryExhausts(t *testing.T) {
	calls := 0
	err := withRetry(2, func() error {
		calls++
		return os.ErrInvalid
	})
	if err == nil || calls != 2 {
		t.Fatalf("withRetry err=%v calls=%d, want err/2", err, calls)
	}
}

func TestVerifyFileSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.bin")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	// echo -n abc | sha256sum
	const want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if err := verifyFileSHA256(path, want); err != nil {
		t.Fatalf("valid hash: %v", err)
	}
	if err := verifyFileSHA256(path, "00"+want[2:]); err == nil {
		t.Fatal("wrong hash should fail")
	}
	if err := verifyFileSHA256(path, "not-hex"); err == nil {
		t.Fatal("invalid hex should fail")
	}
}

func TestRemoveStaleTempDirsAgeExpiry(t *testing.T) {
	parent := t.TempDir()
	modelPath := filepath.Join(parent, "model")
	// Live-PID dir that is very old must still be collected (PID reuse guard).
	old := modelPath + ".tmp-" + strconv.Itoa(os.Getpid())
	if err := os.MkdirAll(old, 0o755); err != nil {
		t.Fatal(err)
	}
	oldTime := staleTempDirMaxAge + time.Hour
	if err := os.Chtimes(old, time.Now().Add(-oldTime), time.Now().Add(-oldTime)); err != nil {
		t.Fatal(err)
	}
	removeStaleTempDirs(modelPath)
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("old live-PID dir should be removed, err=%v", err)
	}
}

// rangeServer serves body with HTTP Range support: no Range header gets the
// full body with200, a valid Range gets206 with the remainder, anything else
// gets416. hits counts requests.
func rangeServer(t *testing.T, body []byte, hits *int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*hits++
		rng := r.Header.Get("Range")
		if rng == "" {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
			return
		}
		var start int
		if _, err := fmt.Sscanf(rng, "bytes=%d-", &start); err != nil || start < 0 || start >= len(body) {
			http.Error(w, "range not satisfiable", http.StatusRequestedRangeNotSatisfiable)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
		w.Header().Set("Content-Length", strconv.Itoa(len(body)-start))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(body[start:])
	}))
	t.Cleanup(srv.Close)
	return srv
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestResumeDownloadAppendsOn206(t *testing.T) {
	t.Setenv("KSOUND_MODEL_SHA256", "")
	body := []byte("0123456789abcdefghij")
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), "a.bin")
	if err := os.WriteFile(dest, body[:7], 0o644); err != nil {
		t.Fatal(err)
	}
	if err := resumeDownload(srv.URL, dest); err != nil {
		t.Fatalf("resumeDownload: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Fatalf("content = %q, want %q", got, body)
	}
}

func TestResumeDownloadRestartsWhenServerIgnoresRange(t *testing.T) {
	t.Setenv("KSOUND_MODEL_SHA256", "")
	body := []byte("full body content")
	hits := 0
	// Server never honours Range (always200 with the full body).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	dest := filepath.Join(t.TempDir(), "a.bin")
	if err := os.WriteFile(dest, []byte("stale-partial"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := resumeDownload(srv.URL, dest); err != nil {
		t.Fatalf("resumeDownload: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Fatalf("content = %q, want %q", got, body)
	}
}

func TestResumeDownload416ReportsRejectedRange(t *testing.T) {
	t.Setenv("KSOUND_MODEL_SHA256", "")
	body := []byte("complete")
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), "a.bin")
	if err := os.WriteFile(dest, body, 0o644); err != nil { // offset == len: unsatisfiable
		t.Fatal(err)
	}
	if err := resumeDownload(srv.URL, dest); !errors.Is(err, errRangeRejected) {
		t.Fatalf("err = %v, want errRangeRejected", err)
	}
}

func TestResumeDownloadTruncatedBodyErrors(t *testing.T) {
	t.Setenv("KSOUND_MODEL_SHA256", "")
	body := []byte("0123456789")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body[:4]) // fewer bytes than announced
	}))
	t.Cleanup(srv.Close)
	dest := filepath.Join(t.TempDir(), "a.bin")
	if err := resumeDownload(srv.URL, dest); err == nil {
		t.Fatal("want truncation error, got nil")
	}
}

func TestEnsureModelArchiveTOFUSidecar(t *testing.T) {
	t.Setenv("KSOUND_MODEL_SHA256", "")
	body := []byte("model tarball bytes")
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), modelArchiveName)
	if err := ensureModelArchive(srv.URL, dest); err != nil {
		t.Fatalf("ensureModelArchive: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Fatalf("content = %q, want %q", got, body)
	}
	sidecar, err := os.ReadFile(dest + ".sha256")
	if err != nil {
		t.Fatalf("sidecar: %v", err)
	}
	if strings.TrimSpace(string(sidecar)) != sha256Hex(body) {
		t.Fatalf("sidecar = %q, want %q", sidecar, sha256Hex(body))
	}
}

func TestEnsureModelArchivePinnedSkipsDownload(t *testing.T) {
	body := []byte("pinned bytes")
	t.Setenv("KSOUND_MODEL_SHA256", sha256Hex(body))
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), modelArchiveName)
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureModelArchive(srv.URL, dest); err != nil {
		t.Fatalf("ensureModelArchive: %v", err)
	}
	if hits != 0 {
		t.Fatalf("hits = %d, want 0 (no HTTP when pinned archive verifies)", hits)
	}
}

func TestEnsureModelArchivePinnedMismatchRetriesAndFails(t *testing.T) {
	body := []byte("actual bytes")
	t.Setenv("KSOUND_MODEL_SHA256", sha256Hex([]byte("different bytes")))
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), modelArchiveName)
	if err := ensureModelArchive(srv.URL, dest); err == nil {
		t.Fatal("want checksum error, got nil")
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Fatalf("archive should be removed after mismatch, err=%v", err)
	}
	if hits < 2 {
		t.Fatalf("hits = %d, want retries", hits)
	}
}

func TestEnsureModelArchiveSidecarPinsRediscovery(t *testing.T) {
	body := []byte("rediscovered bytes")
	t.Setenv("KSOUND_MODEL_SHA256", "")
	hits := 0
	srv := rangeServer(t, body, &hits)
	dest := filepath.Join(t.TempDir(), modelArchiveName)
	if err := os.WriteFile(dest+".sha256", []byte(sha256Hex(body)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureModelArchive(srv.URL, dest); err != nil {
		t.Fatalf("ensureModelArchive: %v", err)
	}
	if hits != 1 {
		t.Fatalf("hits = %d, want 1", hits)
	}
	if _, err := os.Stat(dest); err != nil {
		t.Fatalf("archive: %v", err)
	}
}

// buildModelTar returns an uncompressed tar containing files (names are
// archive paths, e.g. "top/encoder.int8.onnx").
func buildModelTar(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for name, data := range files {
		hdr := &tar.Header{Name: name, Mode: 0o644, Size: int64(len(data))}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestExtractTar(t *testing.T) {
	data := buildModelTar(t, map[string]string{
		"top/encoder.int8.onnx": "enc",
		"top/decoder.int8.onnx": "dec",
		"top/joiner.int8.onnx":  "joi",
		"top/tokens.txt":        "tok",
		"top/test_wavs/x.wav":   "skip",
	})
	out := filepath.Join(t.TempDir(), "out")
	if err := extractTar(bytes.NewReader(data), out); err != nil {
		t.Fatalf("extractTar: %v", err)
	}
	for name, want := range map[string]string{
		"encoder.int8.onnx": "enc", "decoder.int8.onnx": "dec",
		"joiner.int8.onnx": "joi", "tokens.txt": "tok",
	} {
		got, err := os.ReadFile(filepath.Join(out, name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if string(got) != want {
			t.Fatalf("%s = %q, want %q", name, got, want)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "x.wav")); !os.IsNotExist(err) {
		t.Fatalf("test_wavs file should be skipped, err=%v", err)
	}
}

func TestExtractTarMissingFileFails(t *testing.T) {
	data := buildModelTar(t, map[string]string{"top/encoder.int8.onnx": "enc"})
	err := extractTar(bytes.NewReader(data), filepath.Join(t.TempDir(), "out"))
	if err == nil {
		t.Fatal("want missing-file error, got nil")
	}
	if !strings.Contains(err.Error(), "missing") {
		t.Fatalf("err = %v, want mention of missing file", err)
	}
}

func TestExtractModelArchiveRejectsNonBzip2(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "a.tar.bz2")
	if err := os.WriteFile(archive, []byte("not a bzip2 stream"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := extractModelArchive(archive, filepath.Join(dir, "out")); err == nil {
		t.Fatal("want error for non-bzip2 archive, got nil")
	}
}
