package main

import (
	"archive/tar"
	"compress/bzip2"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	modelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
	modelDir = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
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

// EnsureModel downloads and extracts the Parakeet model on first run.
func EnsureModel() (ModelFiles, error) {
	dir, err := ModelCacheDir()
	if err != nil {
		return ModelFiles{}, err
	}
	m := ModelFiles{
		Encoder: filepath.Join(dir, "encoder.int8.onnx"),
		Decoder: filepath.Join(dir, "decoder.int8.onnx"),
		Joiner:  filepath.Join(dir, "joiner.int8.onnx"),
		Tokens:  filepath.Join(dir, "tokens.txt"),
	}
	if allExist(m) {
		return m, nil
	}
	fmt.Printf("Downloading Parakeet model (~630MB) to %s ...\n", dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ModelFiles{}, err
	}
	if err := downloadAndExtract(modelURL, dir); err != nil {
		return ModelFiles{}, err
	}
	if !allExist(m) {
		return ModelFiles{}, fmt.Errorf("model extraction incomplete in %s", dir)
	}
	fmt.Println("Model ready.")
	return m, nil
}

func allExist(m ModelFiles) bool {
	for _, p := range []string{m.Encoder, m.Decoder, m.Joiner, m.Tokens} {
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			return false
		}
	}
	return true
}

func downloadAndExtract(url, destDir string) error {
	resp, err := http.Get(url) //nolint:gosec,noctx
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download model: HTTP %s", resp.Status)
	}

	br := bzip2.NewReader(resp.Body)
	tr := tar.NewReader(br)
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
			f.Close()
			return err
		}
		f.Close()
		fmt.Printf("  extracted %s\n", name)
	}
	return nil
}
