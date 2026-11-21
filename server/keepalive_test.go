package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestExpiresAt(t *testing.T) {
	s := New(Config{})
	soon := time.Now().Add(time.Minute)
	if got := s.expiresAt(&loaded{expires: soon}); !got.Equal(soon) {
		t.Errorf("idle: %v, want %v", got, soon)
	}
	// answering: when it would unload if the answers finished now
	if got := s.expiresAt(&loaded{active: 1}); got.Sub(time.Now()) < 4*time.Minute || got.Sub(time.Now()) > 6*time.Minute {
		t.Errorf("busy: %v", got)
	}
	// kept forever: a real (far-future) time, never "", which Ollama's client can't parse
	if got := s.expiresAt(&loaded{}); got.Year() < 2200 {
		t.Errorf("forever: %v", got)
	}
}

func TestUnloadWhenIdleConfig(t *testing.T) {
	if s := New(Config{}); s.cfg.KeepAlive != 5*time.Minute {
		t.Errorf("default: %v", s.cfg.KeepAlive)
	}
	if s := New(Config{UnloadWhenIdle: true, KeepAlive: time.Hour}); s.cfg.KeepAlive != 0 {
		t.Errorf("UnloadWhenIdle: %v", s.cfg.KeepAlive)
	}
}

// TestKeepAlive loads and unloads a real model: GGUF=/path/to/qwen3-0.6b.gguf go test ./server -run KeepAlive
func TestKeepAlive(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	dir := t.TempDir()
	if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "test.gguf")
	loadedName := func(s *Server) string {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.cur == nil {
			return ""
		}
		return s.cur.name
	}

	t.Run("UnloadWhenIdle", func(t *testing.T) {
		s := New(Config{ModelsDir: dir, UnloadWhenIdle: true})
		defer s.Close()
		l, _, err := s.acquire(model)
		if err != nil {
			t.Fatal(err)
		}
		s.release(l, nil)
		if n := loadedName(s); n != "" {
			t.Errorf("still loaded: %s", n)
		}
	})

	t.Run("ZeroWhileBusy", func(t *testing.T) {
		s := New(Config{ModelsDir: dir})
		defer s.Close()
		a, _, _ := s.acquire(model)
		b, _, err := s.acquire(model)
		if err != nil || a != b {
			t.Fatal(err)
		}
		zero := time.Duration(0)
		s.release(a, &zero) // keep_alive 0 while b still answers
		if loadedName(s) == "" {
			t.Fatal("unloaded while answering")
		}
		s.release(b, nil) // b has no keep_alive of its own
		if n := loadedName(s); n != "" {
			t.Errorf("keep_alive 0 was lost: %s still loaded", n)
		}
	})

	t.Run("Timer", func(t *testing.T) {
		s := New(Config{ModelsDir: dir})
		defer s.Close()
		ts := newTestServer(t, s)
		code, out := call(t, ts, "POST", "/api/generate", `{"model":"`+model+`","keep_alive":"300ms"}`)
		if code != 200 || out["done_reason"] != "load" {
			t.Fatalf("preload: %d %v", code, out)
		}
		_, ps := call(t, ts, "GET", "/api/ps", "")
		running := ps["models"].([]any)
		if len(running) != 1 {
			t.Fatalf("ps: %v", ps)
		}
		exp, err := time.Parse(time.RFC3339Nano, running[0].(map[string]any)["expires_at"].(string))
		if err != nil || time.Until(exp) > time.Second {
			t.Errorf("expires_at: %v %v", exp, err)
		}
		time.Sleep(600 * time.Millisecond)
		if n := loadedName(s); n != "" {
			t.Errorf("still loaded after keep_alive: %s", n)
		}
		code, out = call(t, ts, "POST", "/api/generate", `{"model":"`+model+`","keep_alive":-1}`)
		if code != 200 {
			t.Fatalf("load forever: %d %v", code, out)
		}
		_, ps = call(t, ts, "GET", "/api/ps", "")
		exp, err = time.Parse(time.RFC3339Nano, ps["models"].([]any)[0].(map[string]any)["expires_at"].(string))
		if err != nil || exp.Year() < 2200 {
			t.Errorf("forever expires_at: %v %v", exp, err)
		}
		code, out = call(t, ts, "POST", "/api/generate", `{"model":"`+model+`","keep_alive":0}`)
		if code != 200 || out["done_reason"] != "unload" || loadedName(s) != "" {
			t.Errorf("unload: %d %v loaded=%q", code, out, loadedName(s))
		}
	})
}

func newTestServer(t *testing.T, s *Server) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	return ts
}
