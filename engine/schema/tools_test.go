package schema

import (
	"encoding/json"
	"testing"

	"github.com/benallfree/nollama/engine"
)

var testTools = []Tool{
	{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}},"required":["city"]}`)},
	{Name: "get_time", Parameters: json.RawMessage(`{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}`)},
	{Name: "ping"}, // no parameters: any object
}

func TestToolsFit(t *testing.T) {
	auto, err := CompileTools(testTools, false)
	if err != nil {
		t.Fatal(err)
	}
	req, err := CompileTools(testTools, true)
	if err != nil {
		t.Fatal(err)
	}
	named, err := CompileTools(testTools[1:2], true)
	if err != nil {
		t.Fatal(err)
	}
	const call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	cases := []struct {
		tools *Tools
		text  string
		want  engine.Fit
	}{
		{auto, ``, engine.FitFree},
		{auto, `The weather is`, engine.FitFree},
		{auto, `Let me check. <tool_`, engine.FitFree}, // not a call until the tag is whole
		{auto, `Let me check.` + "\n<tool_call>", engine.FitPrefix},
		{auto, "<tool_call>\n{\"name\": \"get_", engine.FitPrefix},
		{auto, "<tool_call>\n{\"name\": \"get_x", engine.FitNo},                                // no such tool
		{auto, "<tool_call>\n{\"arguments\": {}", engine.FitNo},                                // the name comes first
		{auto, "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"city\"", engine.FitNo}, // get_time takes a zone
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\"", engine.FitPrefix},
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}", engine.FitNo}, // city is required
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"k\"", engine.FitNo},
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}", engine.FitPrefix},
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool", engine.FitPrefix},
		{auto, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, ", engine.FitNo},
		{auto, call, engine.FitComplete},
		{auto, call + "\n", engine.FitComplete},
		{auto, call + "\nThanks", engine.FitNo}, // after a call, only more calls
		{auto, call + "\n<tool_call>\n{\"name\": \"ping\", \"arguments\": {\"anything\": [1, 2]}}\n</tool_call>", engine.FitComplete},
		{auto, call + "\n<tool_call>\n{\"name\": \"ping\", \"arguments\": 3", engine.FitNo}, // arguments are an object
		{req, ``, engine.FitPrefix},
		{req, `<tool`, engine.FitPrefix},
		{req, `Sure`, engine.FitNo}, // a call from the first token
		{req, "\n<tool_call>\n{\"name\": \"ping\"", engine.FitPrefix},
		{req, call, engine.FitComplete},
		{named, "<tool_call>\n{\"name\": \"get_weather\"", engine.FitNo}, // only get_time
		{named, "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"UTC\"}}\n</tool_call>", engine.FitComplete},
	}
	for _, c := range cases {
		if got := c.tools.Fit(c.text); got != c.want {
			t.Errorf("%q: got %d, want %d", c.text, got, c.want)
		}
	}
}

func TestCompileToolsErrors(t *testing.T) {
	for _, tools := range [][]Tool{
		nil,
		{{Name: ""}},
		{{Name: "a"}, {Name: "a"}},
		{{Name: "a", Parameters: json.RawMessage(`"object"`)}},
	} {
		if _, err := CompileTools(tools, false); err == nil {
			t.Errorf("%v: want an error", tools)
		}
	}
}

func TestSplitToolCalls(t *testing.T) {
	content, calls := SplitToolCalls("Checking both.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n" +
		"<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"cut")
	if content != "Checking both." || len(calls) != 2 {
		t.Fatalf("got %q %+v", content, calls)
	}
	if calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city": "Paris"}` || calls[1].Name != "ping" {
		t.Errorf("calls: %+v", calls)
	}
	if content, calls := SplitToolCalls("just text"); content != "just text" || calls != nil {
		t.Errorf("no calls: %q %v", content, calls)
	}
}
