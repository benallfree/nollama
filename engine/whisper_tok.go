package engine

import (
	"fmt"
	"strings"
	"unicode"
)

// whisperTokenizer is Whisper's tiktoken-style byte-level BPE: a token's id is its merge rank, and the special tokens
// (end of text, start of transcript, languages, task, timestamps) follow the ranked tokens.
type whisperTokenizer struct {
	tokens [][]byte       // ranked tokens, id order
	ids    map[string]int // token bytes -> id
	nVocab int            // all tokens, specials included

	eot, sot, translate, transcribe, solm, prev, nospeech, notimestamps, timestampBegin int
	langBegin, nLangs                                                                   int
}

// whisperLanguages are Whisper's language codes in token order (yue only in models with 100 languages: large-v3).
var whisperLanguages = []string{
	"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
	"vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
	"cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
	"kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
	"yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
	"ha", "ba", "jw", "su", "yue",
}

// newWhisperTokenizer builds the tokenizer from the ranked tokens and the model's total vocabulary size, which says
// which special-token layout it has: 51864 English-only, 51865 multilingual (99 languages), 51866 large-v3 (100).
func newWhisperTokenizer(tokens [][]byte, nVocab int) (*whisperTokenizer, error) {
	t := &whisperTokenizer{tokens: tokens, ids: make(map[string]int, len(tokens)), nVocab: nVocab}
	for i, b := range tokens {
		if _, dup := t.ids[string(b)]; !dup && len(b) > 0 {
			t.ids[string(b)] = i
		}
	}
	multilingual := nVocab >= 51865
	t.eot, t.sot = 50256, 50257
	if multilingual {
		t.eot, t.sot = 50257, 50258
	}
	t.langBegin = t.sot + 1
	t.nLangs = 0
	if multilingual {
		t.nLangs = 99 + (nVocab - 51865)
		if t.nLangs > len(whisperLanguages) {
			return nil, fmt.Errorf("engine: unknown whisper vocabulary size %d", nVocab)
		}
	}
	t.translate = t.langBegin + t.nLangs
	if !multilingual {
		t.translate = 50357
	}
	t.transcribe = t.translate + 1
	t.solm = t.translate + 2
	t.prev = t.translate + 3
	t.nospeech = t.translate + 4
	t.notimestamps = t.translate + 5
	t.timestampBegin = t.translate + 6
	if t.timestampBegin+1501 != nVocab { // 1501 timestamps, 0.00 to 30.00 s
		return nil, fmt.Errorf("engine: whisper vocabulary size %d doesn't match its special tokens", nVocab)
	}
	return t, nil
}

// lang returns the token for a language code, or -1.
func (t *whisperTokenizer) lang(code string) int {
	for i := 0; i < t.nLangs; i++ {
		if whisperLanguages[i] == code {
			return t.langBegin + i
		}
	}
	return -1
}

func (t *whisperTokenizer) isTimestamp(id int) bool { return id >= t.timestampBegin }

// isText is a ranked token, as opposed to a special one.
func (t *whisperTokenizer) isText(id int) bool { return id >= 0 && id < len(t.tokens) && id < t.eot }

// decode joins the text tokens' bytes, skipping special tokens.
func (t *whisperTokenizer) decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		if t.isText(id) {
			b = append(b, t.tokens[id]...)
		}
	}
	return string(b)
}

// tokenString is a token's text, with special tokens spelled as Whisper names them (<|en|>, <|0.00|>).
func (t *whisperTokenizer) tokenString(id int) string {
	switch {
	case t.isText(id):
		return string(t.tokens[id])
	case id == t.eot:
		return "<|endoftext|>"
	case id == t.sot:
		return "<|startoftranscript|>"
	case id >= t.langBegin && id < t.langBegin+t.nLangs:
		return "<|" + whisperLanguages[id-t.langBegin] + "|>"
	case id == t.translate:
		return "<|translate|>"
	case id == t.transcribe:
		return "<|transcribe|>"
	case id == t.solm:
		return "<|startoflm|>"
	case id == t.prev:
		return "<|startofprev|>"
	case id == t.nospeech:
		return "<|nospeech|>"
	case id == t.notimestamps:
		return "<|notimestamps|>"
	case t.isTimestamp(id) && id < t.nVocab:
		return fmt.Sprintf("<|%.2f|>", float64(id-t.timestampBegin)*0.02)
	}
	return fmt.Sprintf("<|%d|>", id)
}

// encode tokenizes plain text (no special tokens): GPT-2 pre-tokenization, then tiktoken's byte-pair merging, where
// the pair whose joined bytes have the lowest rank merges first.
func (t *whisperTokenizer) encode(s string) []int {
	var out []int
	for _, w := range gpt2PreTokenize(s) {
		out = t.bpe([]byte(w), out)
	}
	return out
}

func (t *whisperTokenizer) bpe(w []byte, out []int) []int {
	if id, ok := t.ids[string(w)]; ok {
		return append(out, id)
	}
	parts := make([][]byte, len(w))
	for i := range w {
		parts[i] = w[i : i+1]
	}
	for len(parts) > 1 {
		best, bi := -1, -1
		for i := 0; i+1 < len(parts); i++ {
			j := string(parts[i]) + string(parts[i+1])
			if r, ok := t.ids[j]; ok && (best < 0 || r < best) {
				best, bi = r, i
			}
		}
		if bi < 0 {
			break
		}
		parts[bi] = w[offsetOf(w, parts[bi]) : offsetOf(w, parts[bi+1])+len(parts[bi+1])]
		parts = append(parts[:bi+1], parts[bi+2:]...)
	}
	for _, p := range parts {
		if id, ok := t.ids[string(p)]; ok {
			out = append(out, id)
		}
	}
	return out
}

// offsetOf is the index of sub (a subslice of w) in w.
func offsetOf(w, sub []byte) int { return cap(w) - cap(sub) }

// gpt2PreTokenize splits s the way GPT-2's (and Whisper's) regex does:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func gpt2PreTokenize(s string) []string {
	rs := []rune(s)
	n := len(rs)
	isL, isN, isS := unicode.IsLetter, unicode.IsNumber, unicode.IsSpace
	var out []string
	for i := 0; i < n; {
		c := rs[i]
		end := -1
		if c == '\'' && i+1 < n { // contractions, case-sensitive in GPT-2
			c1 := rs[i+1]
			var c2 rune = -1
			if i+2 < n {
				c2 = rs[i+2]
			}
			switch {
			case c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd':
				end = i + 2
			case (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'):
				end = i + 3
			}
		}
		if end < 0 {
			j := i
			if c == ' ' && i+1 < n && !isS(rs[i+1]) {
				j = i + 1
			}
			switch r := rs[j]; {
			case isL(r):
				for j < n && isL(rs[j]) {
					j++
				}
				end = j
			case isN(r):
				for j < n && isN(rs[j]) {
					j++
				}
				end = j
			case !isS(r):
				for j < n && !isS(rs[j]) && !isL(rs[j]) && !isN(rs[j]) {
					j++
				}
				end = j
			}
		}
		if end < 0 { // whitespace: \s+(?!\S) leaves the last space for the next word, else \s+
			k := i
			for k < n && isS(rs[k]) {
				k++
			}
			end = k
			if k < n && k-i >= 2 {
				end = k - 1
			}
		}
		out = append(out, string(rs[i:end]))
		i = end
	}
	return out
}

// promptString renders token ids for debugging.
func (t *whisperTokenizer) promptString(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(t.tokenString(id))
	}
	return b.String()
}
