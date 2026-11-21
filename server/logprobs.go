package server

import (
	"errors"
	"fmt"

	"github.com/benallfree/nollama/engine"
)

// maxTopLogprobs is the most alternatives a request may ask for per token (OpenAI's chat limit).
const maxTopLogprobs = 20

func checkTopLogprobs(n, limit int) error {
	if n < 0 || n > limit {
		return badRequest{fmt.Errorf("top_logprobs must be between 0 and %d", limit)}
	}
	return nil
}

var errTopWithoutLogprobs = badRequest{errors.New("top_logprobs needs logprobs: true")}

// lpToken is a chosen token's log-probability, its bytes, where they sit in the generated output, and the most
// likely tokens at its position. Bytes are copied out of the model while it is held, so the token can be written
// after the model is released.
type lpToken struct {
	bytes      []byte
	logprob    float64
	top        []lpAlt
	start, end int // byte offsets in the generated output
}

type lpAlt struct {
	bytes   []byte
	logprob float64
}

func newLPToken(m *engine.Model, lp engine.TokenLogprob, start int) lpToken {
	b := m.TokenBytes(lp.Token)
	t := lpToken{bytes: b, logprob: lp.Logprob.Logprob, start: start, end: start + len(b)}
	for _, alt := range lp.Top {
		t.top = append(t.top, lpAlt{m.TokenBytes(alt.Token), alt.Logprob})
	}
	return t
}

// lpStream pairs the logprobs the engine reports as it picks tokens with the text it streams, which can lag behind
// (text that might start a stop string is held back). A token goes out with the piece that completes its bytes.
type lpStream struct {
	m       *engine.Model // set by Server.generate once the model is loaded
	q       []lpToken
	total   int // bytes of every token reported so far
	emitted int // bytes of text streamed so far
}

func (s *lpStream) add(lp engine.TokenLogprob) {
	t := newLPToken(s.m, lp, s.total)
	s.q = append(s.q, t)
	s.total = t.end
}

// take returns the tokens whose bytes are all in the text streamed once piece is.
func (s *lpStream) take(piece string) []lpToken {
	s.emitted += len(piece)
	i := 0
	for i < len(s.q) && s.q[i].end <= s.emitted {
		i++
	}
	out := s.q[:i:i]
	s.q = s.q[i:]
	return out
}

// rest returns, once generation is over, the tokens left that put at least one byte into the final text of n bytes
// (one cut by a stop string, or an unfinished UTF-8 character).
func (s *lpStream) rest(n int) []lpToken {
	out := []lpToken{}
	for _, t := range s.q {
		if t.start < n {
			out = append(out, t)
		}
	}
	s.q = nil
	return out
}

// lpTokens is a finished answer's logprobs (engine.Result.Logprobs) with their bytes and offsets.
func lpTokens(m *engine.Model, lps []engine.TokenLogprob) []lpToken {
	out := make([]lpToken, len(lps))
	off := 0
	for i, lp := range lps {
		out[i] = newLPToken(m, lp, off)
		off = out[i].end
	}
	return out
}

// lpEntry is one token in OpenAI's chat logprobs and in Ollama's: the token as text, its log-probability, its raw
// bytes (a token can be part of a UTF-8 character) and the most likely tokens at its position.
type lpEntry struct {
	Token       string     `json:"token"`
	Logprob     float64    `json:"logprob"`
	Bytes       []int      `json:"bytes"`
	TopLogprobs []lpEntryN `json:"top_logprobs"`
}

type lpEntryN struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

func ints(b []byte) []int {
	out := make([]int, len(b))
	for i, c := range b {
		out[i] = int(c)
	}
	return out
}

func lpEntries(toks []lpToken) []lpEntry {
	out := make([]lpEntry, len(toks))
	for i, t := range toks {
		e := lpEntry{Token: string(t.bytes), Logprob: t.logprob, Bytes: ints(t.bytes), TopLogprobs: make([]lpEntryN, len(t.top))}
		for j, alt := range t.top {
			e.TopLogprobs[j] = lpEntryN{Token: string(alt.bytes), Logprob: alt.logprob, Bytes: ints(alt.bytes)}
		}
		out[i] = e
	}
	return out
}

// legacyLogprobs is the logprobs of OpenAI's (legacy) completions: parallel lists, offsets into the completion.
type legacyLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float64            `json:"token_logprobs"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

func legacyOf(toks []lpToken) legacyLogprobs {
	l := legacyLogprobs{Tokens: []string{}, TokenLogprobs: []float64{}, TopLogprobs: []map[string]float64{}, TextOffset: []int{}}
	for _, t := range toks {
		l.Tokens = append(l.Tokens, string(t.bytes))
		l.TokenLogprobs = append(l.TokenLogprobs, t.logprob)
		top := map[string]float64{}
		for _, alt := range t.top {
			if _, dup := top[string(alt.bytes)]; !dup { // two tokens can share a text; the likelier comes first
				top[string(alt.bytes)] = alt.logprob
			}
		}
		l.TopLogprobs = append(l.TopLogprobs, top)
		l.TextOffset = append(l.TextOffset, t.start)
	}
	return l
}
