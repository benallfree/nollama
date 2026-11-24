package server

import (
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"sync"

	"github.com/benallfree/nollama/catalog"
)

// What the server lets clients do, set in Config. The zero Config is the CLI server's behavior: no API key, model
// management on, any model, loopback hosts and origins only. Embedders usually want ReadOnly and Models as well.

// errForbidden is a request the server's settings refuse.
type errForbidden struct{ error }

// errUnauthorized is a request without the API key.
var errUnauthorized = errors.New("missing or wrong API key: send Authorization: Bearer <key>")

// errBusy is a client with too many requests at once.
type errBusy struct{ error }

// guard checks every request before it reaches the API: the Host header (DNS rebinding), the Origin header (web
// pages calling a local server), the API key, the request body's size and how many requests the client has open.
func (s *Server) guard(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fail := func(err error) {
			if strings.HasPrefix(r.URL.Path, "/v1/") {
				openaiError(w, err)
			} else {
				ollamaError(w, err)
			}
		}
		if !s.hostAllowed(r.Host) {
			fail(errForbidden{fmt.Errorf("host %q isn't allowed (set AllowedHosts / --allowed-hosts to serve it by that name)", r.Host)})
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" {
			if !s.originAllowed(origin) {
				fail(errForbidden{fmt.Errorf("origin %q isn't allowed (set AllowedOrigins / --allowed-origins)", origin)})
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Add("Vary", "Origin")
			if r.Method == http.MethodOptions {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, HEAD, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		health := (r.Method == http.MethodGet || r.Method == http.MethodHead) && r.URL.Path == "/"
		if !health && !s.keyOK(r) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="nollama"`)
			fail(errUnauthorized)
			return
		}
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes)
		}
		if s.cfg.PerClient > 0 {
			client := clientOf(r)
			if !s.clients.enter(client, s.cfg.PerClient) {
				fail(errBusy{fmt.Errorf("too many requests at once from %s (the limit is %d)", client, s.cfg.PerClient)})
				return
			}
			defer s.clients.leave(client)
		}
		h.ServeHTTP(w, r)
	})
}

// keyOK reports whether r carries the API key (always true when there is none). Both OpenAI's header
// (Authorization: Bearer) and X-API-Key work; the comparison takes the same time whatever the key sent.
func (s *Server) keyOK(r *http.Request) bool {
	if s.cfg.APIKey == "" {
		return true
	}
	got := r.Header.Get("X-API-Key")
	if a := r.Header.Get("Authorization"); a != "" {
		if scheme, tok, ok := strings.Cut(a, " "); ok && strings.EqualFold(scheme, "Bearer") {
			got = strings.TrimSpace(tok)
		}
	}
	want := sha256.Sum256([]byte(s.cfg.APIKey))
	have := sha256.Sum256([]byte(got))
	return subtle.ConstantTimeCompare(want[:], have[:]) == 1
}

// hostAllowed checks the Host header against DNS rebinding: a web page whose own domain is made to point at
// 127.0.0.1 sends its domain as Host. IP addresses and localhost names always pass, as does anything once an API key
// is set (a rebinding page can't know it), and names in AllowedHosts ("*" allows any).
func (s *Server) hostAllowed(hostport string) bool {
	if s.cfg.APIKey != "" || hostport == "" {
		return true
	}
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.ToLower(strings.Trim(host, "[]"))
	if net.ParseIP(host) != nil || isLocalName(host) {
		return true
	}
	for _, a := range s.cfg.AllowedHosts {
		a = strings.ToLower(a)
		if a == "*" || a == host || (strings.HasPrefix(a, ".") && strings.HasSuffix(host, a)) {
			return true
		}
	}
	return false
}

// originAllowed checks a browser's Origin header. Pages served from this machine (localhost, 127.0.0.1, [::1], any
// port) pass; others only when listed in AllowedOrigins ("*" allows any). Without this, any web page you visit
// could send requests to a server on your machine.
func (s *Server) originAllowed(origin string) bool {
	for _, a := range s.cfg.AllowedOrigins {
		if a == "*" || strings.EqualFold(strings.TrimRight(a, "/"), origin) {
			return true
		}
	}
	u, err := url.Parse(origin)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	h := strings.ToLower(u.Hostname())
	if isLocalName(h) {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

func isLocalName(h string) bool {
	return h == "localhost" || strings.HasSuffix(h, ".localhost")
}

// clientOf names a request's client by its IP address. Behind a reverse proxy every request comes from the proxy,
// so set the limit there instead.
func clientOf(r *http.Request) string {
	if h, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return h
	}
	return r.RemoteAddr
}

// clientCounts counts the requests each client has open.
type clientCounts struct {
	mu sync.Mutex
	n  map[string]int
}

func (c *clientCounts) enter(client string, limit int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.n == nil {
		c.n = map[string]int{}
	}
	if c.n[client] >= limit {
		return false
	}
	c.n[client]++
	return true
}

func (c *clientCounts) leave(client string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.n[client]--; c.n[client] <= 0 {
		delete(c.n, client)
	}
}

// resolve turns the model a request names into a source, applying the server's settings: the allowlist, and paths
// only inside the models folder unless AllowPaths is set. Every handler that takes a model name goes through it.
func (s *Server) resolve(name string) (catalog.Source, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return catalog.Source{}, badRequest{errors.New("model is required")}
	}
	if !s.modelAllowed(name) {
		return catalog.Source{}, errForbidden{fmt.Errorf("model %q isn't one this server serves", name)}
	}
	src, err := catalog.Resolve(name, s.cfg.ModelsDir)
	if err != nil {
		return src, badRequest{err}
	}
	if catalog.IsPath(catalog.Normalize(name)) && !s.cfg.AllowPaths && !within(s.cfg.ModelsDir, src.Path) {
		return catalog.Source{}, errForbidden{fmt.Errorf("model %q: this server doesn't load files by path; put the .gguf in its models folder and use its name", name)}
	}
	return src, nil
}

// modelAllowed reports whether the allowlist (Config.Models) lets clients use name. Names compare as nollama spells
// them, so "qwen3:1.7b" matches "qwen3-1.7b".
func (s *Server) modelAllowed(name string) bool {
	if len(s.cfg.Models) == 0 {
		return true
	}
	n := catalog.Normalize(name)
	for _, m := range s.cfg.Models {
		if catalog.Normalize(m) == n {
			return true
		}
	}
	return false
}

// within reports whether path p is inside dir, after cleaning both (so "dir/../x" is not).
func within(dir, p string) bool {
	d, err1 := filepath.Abs(dir)
	f, err2 := filepath.Abs(p)
	if err1 != nil || err2 != nil {
		return false
	}
	rel, err := filepath.Rel(d, f)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

// manage refuses a model-management endpoint (pull, delete, create, copy, push) that the settings turn off.
func (s *Server) manage(what string, off bool, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.ReadOnly || off {
			ollamaError(w, errForbidden{fmt.Errorf("%s is turned off on this server", what)})
			return
		}
		next(w, r)
	}
}
