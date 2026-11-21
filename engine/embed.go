package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// EmbedOptions controls Embed.
type EmbedOptions struct {
	// Dimensions keeps only the first n values of each vector (then normalizes again), for models trained to allow it
	// (Qwen3-Embedding: Matryoshka, 32 to 1024). 0 keeps them all.
	Dimensions int
	// MaxTokens is the most tokens one text may have (including the appended end-of-text token); 0 or more than the
	// model's context means the model's context.
	MaxTokens int
	// Truncate cuts a longer text to MaxTokens (keeping its start) instead of failing.
	Truncate bool
}

// Embedding is one text's vector.
type Embedding struct {
	Vector []float32 // unit length (L2)
	Tokens int       // tokens read, including the end-of-text token the model's tokenizer appends
}

// EmbedQuery formats a search query with the task instruction Qwen3-Embedding expects on queries (documents get none):
// "Instruct: <task>\nQuery:<query>". An empty task uses the model card's web search instruction.
func EmbedQuery(task, query string) string {
	if task == "" {
		task = "Given a web search query, retrieve relevant passages that answer the query"
	}
	return "Instruct: " + task + "\nQuery:" + query
}

// Embed returns one unit-length vector per text: the model's final hidden state (after the output norm) at the
// last token, with the end-of-text token appended when the model's tokenizer asks for it (Qwen3-Embedding does), as
// the model card's reference code does. Texts run one after another; each reads all of its tokens, so the prompt
// cache isn't used. A query for retrieval should go through EmbedQuery first.
func (m *Model) Embed(ctx context.Context, texts []string, o EmbedOptions) ([]Embedding, error) {
	ids := make([][]int, len(texts))
	for i, t := range texts {
		ids[i] = m.Tokenize(t)
	}
	return m.EmbedTokens(ctx, ids, o)
}

// EmbedTokens is Embed on pre-tokenized texts (without the appended end-of-text token, which it adds).
func (m *Model) EmbedTokens(ctx context.Context, texts [][]int, o EmbedOptions) (out []Embedding, err error) {
	m.res.call(func() { out, err = m.embedTokens(ctx, texts, o) })
	return out, err
}

func (m *Model) embedTokens(ctx context.Context, texts [][]int, o EmbedOptions) ([]Embedding, error) {
	if o.Dimensions < 0 || o.Dimensions > m.hp.dim {
		return nil, fmt.Errorf("engine: dimensions %d out of range (1 to %d)", o.Dimensions, m.hp.dim)
	}
	seqs := make([][]int, len(texts))
	longest := 1
	for i, ids := range texts {
		var err error
		if seqs[i], err = m.embedSeq(ids, o.MaxTokens, o.Truncate); err != nil {
			return nil, fmt.Errorf("engine: text %d: %w", i, err)
		}
		longest = max(longest, len(seqs[i]))
	}
	out := make([]Embedding, len(seqs))
	if len(seqs) == 0 {
		return out, nil
	}
	s, err := m.getState(longest, min(longest, prefillBatch))
	if err != nil {
		return nil, err
	}
	defer func() {
		s.hidden = nil
		m.putState(s)
	}()
	for i, ids := range seqs {
		v := make([]float32, m.hp.dim)
		s.hidden = v
		for pos := 0; pos < len(ids); pos += prefillBatch {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			m.forward(s, ids[pos:min(pos+prefillBatch, len(ids))], pos)
		}
		if o.Dimensions > 0 {
			v = v[:o.Dimensions:o.Dimensions]
		}
		normalize(v)
		out[i] = Embedding{Vector: v, Tokens: len(ids)}
	}
	return out, nil
}

// embedSeq is the tokens Embed reads for one text: the end-of-text token appended when the tokenizer asks for it,
// and, when truncate is set, cut to limit tokens (0: the model's context) keeping that token last.
func (m *Model) embedSeq(ids []int, limit int, truncate bool) ([]int, error) {
	if limit <= 0 || limit > m.hp.ctx {
		limit = m.hp.ctx
	}
	if m.addEOS {
		ids = append(ids[:len(ids):len(ids)], m.eosID)
	}
	if len(ids) == 0 {
		return nil, errors.New("empty text")
	}
	if len(ids) > limit {
		if !truncate {
			return nil, fmt.Errorf("%d tokens, more than the limit of %d", len(ids), limit)
		}
		if m.addEOS { // keep the end-of-text token last: it's the one pooled
			ids = append(ids[:limit-1:limit-1], m.eosID)
		} else {
			ids = ids[:limit]
		}
	}
	return ids, nil
}

// normalize scales v to unit length (L2); a zero vector stays zero.
func normalize(v []float32) {
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	if ss == 0 {
		return
	}
	inv := 1 / math.Sqrt(ss)
	for i := range v {
		v[i] = float32(float64(v[i]) * inv)
	}
}

// Cosine returns the cosine similarity of a and b (their dot product when both are unit length, as Embed's are).
func Cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}
