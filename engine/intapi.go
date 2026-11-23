package engine

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

// IntSession is one context: a KV cache and a position. Not safe for concurrent use.
type IntSession struct{ s *intState }

// NewSession starts a context that can hold up to maxContext tokens (at most MaxContext).
func (m *IntModel) NewSession(maxContext int) *IntSession {
	return &IntSession{s: m.newState(min(max(maxContext, 1), m.MaxContext()))}
}

// Reset empties the context.
func (s *IntSession) Reset() { s.s.reset() }

// Pos is the number of tokens fed so far.
func (s *IntSession) Pos() int { return s.s.pos }

// Feed adds a token to the context and returns the next token's distribution as cumulative counts: token t has
// count cum[t] - cum[t-1] (cum[-1] = 0), and cum[VocabSize-1] is the total. The slice is reused by the next Feed.
func (s *IntSession) Feed(tok int) []int64 {
	s.s.step(tok)
	return s.s.cum
}
