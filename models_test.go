package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
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
