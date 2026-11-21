package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// GenerateOptions controls sampling.
type GenerateOptions struct {
	MaxTokens   int     // tokens to generate (default 256)
	Temperature float64 // 0 = greedy
	TopP        float64 // 0 or 1 = off
	TopK        int     // 0 = off
	Seed        uint64
	Stop        []string // stop strings (not included in the output)
	IgnoreEOS   bool     // keep generating past end-of-turn tokens (benchmarks)
	// Constraint, when set, limits the output to text it accepts (a JSON schema, a grammar): each candidate token is
	// tried by calling it with the output so far plus that token's bytes, best-scoring first. End-of-turn is allowed
	// only once it says FitComplete or FitDone, and FitDone ends the output after that token.
	Constraint func(text string) Fit
	// CacheScope separates prompt cache entries: a prompt only reuses the cached start of prompts with the same scope.
	// A host serving several apps passes each app's id, so one app's timing can't tell it what another app sent.
	CacheScope string
	// Thread keeps this conversation's state (every token read and written) after the call, and continues from it
	// when the next call's prompt starts the same way (threads.go). "" for none.
	Thread string
	// ThreadDir, when set with Thread, also keeps the state on disk there (half precision), within ThreadDisk bytes
	// for the folder (<= 0: no limit), least recently used first.
	ThreadDir  string
	ThreadDisk int64
	// Logprobs records each generated token's log-probability in Result.Logprobs, with the TopLogprobs most likely
	// tokens at each step; tokens that only held a stop string are left out. OnLogprob, when set, is also called with
	// each one as it is chosen, before its text reaches onToken (which may hold text back while it could start a stop
	// string), and for every chosen token, stop string or not.
	Logprobs    bool
	TopLogprobs int
	OnLogprob   func(TokenLogprob)
}

// Fit is how a Constraint judges the output so far.
type Fit int

const (
	FitNo       Fit = iota // not a prefix of anything allowed
	FitPrefix              // a valid start; more must follow
	FitComplete            // complete, though more could follow (a number)
	FitDone                // complete, and nothing can follow
	// FitFree: nothing limits the next token here; it is sampled as with no constraint (end-of-turn allowed). A
	// constraint that only starts partway (a tool call after free text) says this until it begins.
	FitFree
)

// Result reports what Generate produced.
type Result struct {
	Text         string
	Tokens       []int
	PromptTokens int
	CachedTokens int // of PromptTokens, how many came from the prompt cache instead of being read again
	OutputTokens int
	Prefill      time.Duration  // prompt processing, up to the first sampled token
	Decode       time.Duration  // remaining tokens
	StopReason   string         // "eos", "stop", "length" (MaxTokens reached), "callback", "canceled"
	Logprobs     []TokenLogprob // one per token of Text, when GenerateOptions.Logprobs
}

// Message is one chat turn.
type Message struct {
	Role      string // system, user, assistant, tool
	Content   string
	ToolCalls []ToolCall // calls an assistant turn made
}

// ToolCall is a call an assistant turn made, as the chat template writes it back.
type ToolCall struct {
	Name      string
	Arguments string // a JSON object
}

// ChatPrompt applies Qwen3's ChatML template with thinking turned off.
func ChatPrompt(messages []Message) string { return ChatPromptTools(messages, nil) }

// ChatPromptTools is ChatPrompt with tools the model may call: each a JSON tool definition as the client sent it
// ({"type": "function", "function": {"name": ..., "parameters": ...}}), listed in the system turn the way Qwen3's
// template does. Assistant turns write their calls back in <tool_call> tags, and tool turns become <tool_response>
// blocks in a user turn.
func ChatPromptTools(messages []Message, tools []json.RawMessage) string {
	var b strings.Builder
	if len(tools) > 0 {
		b.WriteString("<|im_start|>system\n")
		if len(messages) > 0 && messages[0].Role == "system" {
			b.WriteString(messages[0].Content)
			b.WriteString("\n\n")
			messages = messages[1:]
		}
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
			"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, t := range tools {
			b.WriteString("\n")
			b.WriteString(PyJSON(t))
		}
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within " +
			"<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
			"</tool_call><|im_end|>\n")
	}
	for i, m := range messages {
		if m.Role == "tool" {
			if i == 0 || messages[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n")
			b.WriteString(m.Content)
			b.WriteString("\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
			continue
		}
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		if m.Role == "assistant" {
			// the empty thinking block every answer starts with (below), so an earlier answer reads exactly as the model
			// wrote it and a thread's state covers it
			b.WriteString("<think>\n\n</think>\n\n")
		}
		b.WriteString(m.Content)
		for j, c := range m.ToolCalls {
			if j > 0 || m.Content != "" {
				b.WriteString("\n")
			}
			args := c.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			b.WriteString("<tool_call>\n{\"name\": \"")
			b.WriteString(c.Name)
			b.WriteString("\", \"arguments\": ")
			b.WriteString(args)
			b.WriteString("}\n</tool_call>")
		}
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n<think>\n\n</think>\n\n")
	return b.String()
}

// ChatPrompt is the method form of the package function.
func (m *Model) ChatPrompt(messages []Message) string { return ChatPrompt(messages) }

// ChatPromptTools is the method form of the package function.
func (m *Model) ChatPromptTools(messages []Message, tools []json.RawMessage) string {
	return ChatPromptTools(messages, tools)
}

// PyJSON rewrites JSON text the way Python's json.dumps writes it (and so the chat template's tojson): ", " and ": "
// between items, keys in their order, non-ASCII kept. Text that isn't JSON comes back as it is.
func PyJSON(raw []byte) string {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var b strings.Builder
	type level struct {
		object bool
		n      int // items written
	}
	var stack []level
	str := func(s string) {
		var sb bytes.Buffer
		enc := json.NewEncoder(&sb)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(s)
		b.Write(bytes.TrimRight(sb.Bytes(), "\n"))
	}
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return string(raw)
		}
		if d, ok := tok.(json.Delim); ok && (d == '}' || d == ']') {
			stack = stack[:len(stack)-1]
			b.WriteRune(rune(d))
			continue
		}
		if len(stack) > 0 {
			top := &stack[len(stack)-1]
			if top.object {
				if top.n%2 == 1 {
					b.WriteString(": ") // a value after its key
				} else if top.n > 0 {
					b.WriteString(", ")
				}
			} else if top.n > 0 {
				b.WriteString(", ")
			}
			top.n++
		}
		switch v := tok.(type) {
		case json.Delim:
			b.WriteRune(rune(v))
			stack = append(stack, level{object: v == '{'})
		case string:
			str(v)
		case json.Number:
			b.WriteString(v.String())
		case bool:
			b.WriteString(strconv.FormatBool(v))
		case nil:
			b.WriteString("null")
		}
	}
	return b.String()
}

const prefillBatch = 512

// Generate runs the prompt (raw text; the caller applies the chat template)
// and streams generated text pieces to onToken (which may be nil). Returning
// false from onToken stops generation.
func (m *Model) Generate(ctx context.Context, prompt string, o GenerateOptions, onToken func(piece string) bool) (Result, error) {
	ids := m.Tokenize(prompt)
	return m.GenerateTokens(ctx, ids, o, onToken)
}

// GenerateTokens is Generate on pre-tokenized input.
func (m *Model) GenerateTokens(ctx context.Context, ids []int, o GenerateOptions, onToken func(piece string) bool) (res Result, err error) {
	m.res.call(func() { res, err = m.generateTokens(ctx, ids, o, onToken) })
	return res, err
}

func (m *Model) generateTokens(ctx context.Context, ids []int, o GenerateOptions, onToken func(piece string) bool) (res Result, err error) {
	res = Result{PromptTokens: len(ids)}
	if len(ids) == 0 {
		return res, errors.New("engine: empty prompt")
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 256
	}
	maxCtx := len(ids) + o.MaxTokens
	if err := checkContext(len(ids), o.MaxTokens, m.hp.ctx); err != nil { // a *ContextError: never cut the prompt
		return res, err
	}
	if m.respond != nil {
		return m.generateScripted(ctx, ids, o, onToken)
	}
	s, err := m.getState(maxCtx, min(len(ids), prefillBatch))
	if err != nil {
		return res, err
	}
	defer m.putState(s)
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))

	start := time.Now()
	from := 0
	var te *prefixEntry
	tn := 0
	if o.Thread != "" { // this conversation's own state: covers the earlier answers too
		te, tn = m.lookupThread(o.CacheScope, o.Thread, o.ThreadDir, ids)
	}
	if e, n := m.lookupPrefix(o.CacheScope, ids); e != nil && n > tn { // the start of this prompt was read before
		te, tn = e, n
	}
	if te != nil {
		m.restorePrefix(s, te, tn)
		from, res.CachedTokens = tn, tn
	}
	for pos := from; pos < len(ids); pos += prefillBatch {
		if err := ctx.Err(); err != nil {
			res.StopReason = "canceled"
			return res, err
		}
		m.forward(s, ids[pos:min(pos+prefillBatch, len(ids))], pos)
	}
	m.storePrefix(o.CacheScope, s, ids, "")
	fed := 0 // generated tokens whose keys and values are in s
	if o.Thread != "" {
		defer func() { // runs before the state goes back (deferred earlier)
			m.storeThread(o, s, append(append([]int(nil), ids...), res.Tokens[:fed]...))
		}()
	}

	var text strings.Builder
	var pending []byte // bytes not yet forming complete UTF-8
	emitted := 0       // bytes of text already passed to onToken
	maxStop := 0
	for _, st := range o.Stop {
		maxStop = max(maxStop, len(st))
	}
	pos := len(ids)
	var decodeStart time.Time
	finish := func(reason string) {
		res.StopReason = reason
		res.Text = text.String()
		if len(res.Logprobs) > 0 { // keep the tokens that put at least one byte into Text (a stop string drops the rest)
			off, keep := 0, 0
			for _, t := range res.Tokens {
				if off >= len(res.Text) {
					break
				}
				off += len(m.tok.tokenBytes(t))
				keep++
			}
			res.Logprobs = res.Logprobs[:min(keep, len(res.Logprobs))]
		}
		if !decodeStart.IsZero() {
			res.Decode = time.Since(decodeStart)
		}
	}
	var raw []byte // every generated byte, for the constraint
	for {
		var tok int
		done := false
		cur := FitFree
		if o.Constraint != nil {
			cur = o.Constraint(string(raw))
		}
		if cur != FitFree {
			var ok bool
			tok, done, ok = m.sampleConstrained(s.logits, o, rng, raw, cur)
			if !ok {
				finish("stop") // nothing allowed can follow
				break
			}
		} else {
			tok = m.sample(s.logits, o, rng)
		}
		if res.OutputTokens == 0 {
			res.Prefill = time.Since(start)
			decodeStart = time.Now()
		}
		if m.tok.eos[tok] && !o.IgnoreEOS {
			finish("eos")
			break
		}
		if o.Logprobs || o.OnLogprob != nil {
			lp := logprobOf(s.logits, tok, o.TopLogprobs)
			if o.Logprobs {
				res.Logprobs = append(res.Logprobs, lp)
			}
			if o.OnLogprob != nil {
				o.OnLogprob(lp)
			}
		}
		res.Tokens = append(res.Tokens, tok)
		res.OutputTokens++
		tb := m.tok.tokenBytes(tok)
		raw = append(raw, tb...)
		pending = append(pending, tb...)
		// move the complete UTF-8 prefix into text
		k := 0
		for k < len(pending) {
			r, size := utf8.DecodeRune(pending[k:])
			if r == utf8.RuneError && size <= 1 && !utf8.FullRune(pending[k:]) {
				break
			}
			k += size
		}
		text.Write(pending[:k])
		pending = pending[k:]

		stopped := false
		full := text.String()
		if maxStop > 0 {
			for _, st := range o.Stop {
				if st == "" {
					continue
				}
				if i := strings.Index(full[max(0, emitted-len(st)):], st); i >= 0 {
					cut := max(0, emitted-len(st)) + i
					text.Reset()
					text.WriteString(full[:cut])
					full = full[:cut]
					stopped = true
					break
				}
			}
		}
		// hold back text that could be the start of a stop string
		safe := len(full)
		if !stopped && maxStop > 0 {
			safe = max(emitted, len(full)-(maxStop-1))
		}
		if safe > emitted && onToken != nil {
			if !onToken(full[emitted:safe]) {
				emitted = safe
				finish("callback")
				return res, nil
			}
		}
		emitted = max(emitted, safe)
		if done && !stopped {
			if onToken != nil && emitted < len(full) {
				onToken(full[emitted:])
			}
			finish("stop")
			return res, nil
		}
		if stopped {
			finish("stop")
			return res, nil
		}
		if res.OutputTokens >= o.MaxTokens {
			if onToken != nil && emitted < len(full) {
				onToken(full[emitted:])
			}
			finish("length")
			return res, nil
		}
		if err := ctx.Err(); err != nil {
			finish("canceled")
			return res, err
		}
		m.forward(s, []int{tok}, pos)
		pos++
		fed++
	}
	if onToken != nil {
		full := text.String()
		if emitted < len(full) {
			onToken(full[emitted:])
		}
	}
	return res, nil
}

func (m *Model) sample(logits []float32, o GenerateOptions, rng *rand.Rand) int {
	if o.Temperature <= 0 || o.TopK == 1 {
		best, bv := 0, float32(math.Inf(-1))
		for i, v := range logits {
			if v > bv {
				best, bv = i, v
			}
		}
		return best
	}
	type cand struct {
		id int
		p  float32
	}
	k := o.TopK
	if k <= 0 || k > len(logits) {
		k = len(logits)
	}
	// Candidates more than 30/temperature below the best have probability
	// < e^-30 relative to it; dropping them avoids sorting the whole vocabulary.
	mxl := float32(math.Inf(-1))
	for _, v := range logits {
		mxl = max(mxl, v)
	}
	cut := mxl - float32(30*o.Temperature)
	cs := make([]cand, 0, 256)
	for i, v := range logits {
		if v >= cut {
			cs = append(cs, cand{i, v})
		}
	}
	k = min(k, len(cs))
	sort.Slice(cs, func(a, b int) bool { return cs[a].p > cs[b].p })
	cs = cs[:k]
	mx := cs[0].p
	var sum float64
	for i := range cs {
		e := math.Exp(float64(cs[i].p-mx) / o.Temperature)
		cs[i].p = float32(e)
		sum += e
	}
	for i := range cs {
		cs[i].p = float32(float64(cs[i].p) / sum)
	}
	if o.TopP > 0 && o.TopP < 1 {
		var c float64
		for i := range cs {
			c += float64(cs[i].p)
			if c >= o.TopP {
				cs = cs[:i+1]
				break
			}
		}
		sum = c
	} else {
		sum = 1
	}
	r := rng.Float64() * sum
	var c float64
	for _, x := range cs {
		c += float64(x.p)
		if r < c {
			return x.id
		}
	}
	return cs[len(cs)-1].id
}

// sampleConstrained picks the next token among those o.Constraint accepts after out: the best-scoring one when greedy,
// else a sample from the top accepted ones. cur is the constraint's verdict on out. done says the output is finished
// after it; ok is false when no token fits.
func (m *Model) sampleConstrained(logits []float32, o GenerateOptions, rng *rand.Rand, out []byte, cur Fit) (tok int, done, ok bool) {
	type cand struct {
		id int
		v  float32
	}
	prefix := string(out)
	try := func(id int) (Fit, bool) {
		if m.tok.eos[id] {
			return FitDone, cur == FitComplete || cur == FitDone
		}
		if m.tok.isControl(id) {
			return FitNo, false // <|im_start|> and the like; user-defined ones such as <tool_call> are text the constraint judges
		}
		f := o.Constraint(prefix + string(m.tok.tokenBytes(id)))
		return f, f != FitNo
	}
	// the best candidates first; widen to the whole vocabulary only if none of them fits
	for _, width := range []float32{30, float32(math.Inf(1))} {
		mx := float32(math.Inf(-1))
		for _, v := range logits {
			mx = max(mx, v)
		}
		cut := mx - width
		cs := make([]cand, 0, 512)
		for i, v := range logits {
			if v >= cut {
				cs = append(cs, cand{i, v})
			}
		}
		sort.Slice(cs, func(a, b int) bool { return cs[a].v > cs[b].v })
		greedy := o.Temperature <= 0 || o.TopK == 1
		k := o.TopK
		if k <= 0 {
			k = 40
		}
		var fits []cand
		var fitOf []Fit
		for _, c := range cs {
			f, good := try(c.id)
			if !good {
				continue
			}
			if greedy {
				return c.id, f == FitDone && !m.tok.eos[c.id], true
			}
			fits, fitOf = append(fits, c), append(fitOf, f)
			if len(fits) >= k {
				break
			}
		}
		if len(fits) == 0 {
			continue
		}
		var sum float64
		ps := make([]float64, len(fits))
		for i, c := range fits {
			ps[i] = math.Exp(float64(c.v-fits[0].v) / o.Temperature)
			sum += ps[i]
		}
		r := rng.Float64() * sum
		for i := range fits {
			r -= ps[i]
			if r < 0 || i == len(fits)-1 {
				return fits[i].id, fitOf[i] == FitDone && !m.tok.eos[fits[i].id], true
			}
		}
	}
	return 0, false, false
}

// WarmResult says what Warm kept.
type WarmResult struct {
	Tokens int   // the prompt's tokens
	Kept   int   // how many of them the prompt cache holds (0: no room, or the cache is off)
	Bytes  int64 // the memory they take
}

// Warm reads prompt (raw text, like Generate's) into the prompt cache ahead of the calls that start with it, and pins
// the entry under pin so it isn't evicted until Unpin(pin). Calls with the same scope whose prompts start the same way
// then skip reading that part.
func (m *Model) Warm(ctx context.Context, prompt, scope, pin string) (res WarmResult, err error) {
	m.res.call(func() { res, err = m.warm(ctx, prompt, scope, pin) })
	return res, err
}

func (m *Model) warm(ctx context.Context, prompt, scope, pin string) (WarmResult, error) {
	ids := m.Tokenize(prompt)
	res := WarmResult{Tokens: len(ids)}
	if m.prefix.budget <= 0 || len(ids) < prefixMin {
		return res, nil
	}
	if err := checkContext(len(ids), 1, m.hp.ctx); err != nil {
		return res, err
	}
	s, err := m.getState(len(ids)+1, min(len(ids), prefillBatch))
	if err != nil {
		return res, err
	}
	defer m.putState(s)
	from := 0
	if e, n := m.lookupPrefix(scope, append(ids, 0)); e != nil { // all of ids may come from the cache
		m.restorePrefix(s, e, n)
		from = n
	}
	for pos := from; pos < len(ids); pos += prefillBatch {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		m.forward(s, ids[pos:min(pos+prefillBatch, len(ids))], pos)
	}
	res.Kept = m.storePrefix(scope, s, ids, pin)
	res.Bytes = int64(res.Kept) * m.tokenBytes()
	return res, nil
}

// CountTokens is how many tokens text is for this model.
func (m *Model) CountTokens(text string) int { return len(m.Tokenize(text)) }
