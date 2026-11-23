package engine

import "errors"

// The integer model's public API: load, tokenize, and a session that feeds one token at a time and returns the
// next-token distribution as cumulative integer counts.

// IntCountsTotal bounds the distribution's total: every token gets a count of at least 1 and the counts sum to at
// most IntCountsTotal.
const IntCountsTotal = intTotal

// BOS is the token a context starts with.
func (m *IntModel) BOS() int { return m.bos }

// VocabSize is the number of tokens (the length of a distribution).
func (m *IntModel) VocabSize() int { return m.V }

// MaxContext is the longest context a session can hold (the rope tables' length).
func (m *IntModel) MaxContext() int { return len(m.ropeC) / (m.hd / 2) }

// Tokenize splits text into tokens with the model's tokenizer (special tokens such as <|endoftext|> matched whole).
func (m *IntModel) Tokenize(text string) []int { return m.tok.encode(text) }

// TokenBytes returns the bytes a token stands for.
func (m *IntModel) TokenBytes(id int) []byte { return m.tok.tokenBytes(id) }

// ByteToken returns the token that stands for exactly one byte (every byte has one).
func (m *IntModel) ByteToken(b byte) int { return m.tok.vocab[string(m.tok.byteEnc[b])] }

// IntSession is one context: a KV cache and a position, on the model's device. Not safe for concurrent use.
type IntSession struct {
	s      *intState    // the CPU
	g      *intGPUState // or the GPU
	pos    int
	maxlen int
}

// NewSession starts a context that can hold up to maxContext tokens (at most MaxContext).
func (m *IntModel) NewSession(maxContext int) (*IntSession, error) {
	n := min(max(maxContext, 1), m.MaxContext())
	if m.gpu != nil {
		g, err := m.gpu.newState(m, n)
		if err != nil {
			return nil, err
		}
		return &IntSession{g: g, maxlen: n}, nil
	}
	return &IntSession{s: m.newState(n), maxlen: n}, nil
}

// Close frees the session's GPU buffers (CPU sessions need nothing).
func (s *IntSession) Close() {
	if s.g != nil {
		s.g.release()
	}
}

// Reset empties the context.
func (s *IntSession) Reset() {
	s.pos = 0
	if s.s != nil {
		s.s.reset()
	}
}

// Pos is the number of tokens fed so far.
func (s *IntSession) Pos() int { return s.pos }

// IntCount is one position's counts for a known next token t: the coder's interval [Low, High) out of Total, where
// High = cum[t], Low = cum[t-1] (0 for t = 0) and Total = cum[VocabSize-1] (see Feed).
type IntCount struct{ Total, High, Low int64 }

// FeedBatch adds toks to the context and returns, for each, the counts of next[i] (the token that follows it):
// the numbers Feed's tables give, computed many positions at a time when the device can (a GPU), so compression,
// which knows every token in advance, needn't go one token at a time.
func (s *IntSession) FeedBatch(toks, next []int) ([]IntCount, error) {
	if len(next) != len(toks) {
		return nil, errors.New("intmodel: FeedBatch needs one next token per token")
	}
	if s.pos+len(toks) > s.maxlen {
		return nil, errors.New("intmodel: context full")
	}
	out := make([]IntCount, len(toks))
	if s.g != nil {
		if err := s.g.feedBatch(toks, next, s.pos, out); err != nil {
			return nil, err
		}
		s.pos += len(toks)
		return out, nil
	}
	for i, t := range toks {
		cum, err := s.Feed(t)
		if err != nil {
			return nil, err
		}
		out[i] = countOf(cum, next[i])
	}
	return out, nil
}

func countOf(cum []int64, t int) IntCount {
	c := IntCount{Total: cum[len(cum)-1], High: cum[t]}
	if t > 0 {
		c.Low = cum[t-1]
	}
	return c
}

// Feed adds a token to the context and returns the next token's distribution as cumulative counts: token t has
// count cum[t] - cum[t-1] (cum[-1] = 0), and cum[VocabSize-1] is the total. The slice is reused by the next Feed.
func (s *IntSession) Feed(tok int) ([]int64, error) {
	if s.pos >= s.maxlen {
		return nil, errors.New("intmodel: context full")
	}
	s.pos++
	if s.g != nil {
		if err := s.g.step(tok, s.pos-1); err != nil {
			return nil, err
		}
		return s.g.cum, nil
	}
	s.s.step(tok)
	return s.s.cum, nil
}
