package engine

import (
	"encoding/json"
	"testing"
)

// TestChatPromptTools checks the prompt against what Qwen3's chat template (in its GGUF) writes for the same
// conversation, except that every assistant turn keeps the empty thinking block (see ChatPrompt).
func TestChatPromptTools(t *testing.T) {
	tools := []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"get_weather","description":"Weather now <in> a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`)}
	got := ChatPromptTools([]Message{
		{Role: "system", Content: "Be brief."},
		{Role: "user", Content: "Weather in Paris and Rome?"},
		{Role: "assistant", ToolCalls: []ToolCall{{Name: "get_weather", Arguments: `{"city": "Paris"}`}, {Name: "get_weather", Arguments: `{"city": "Rome"}`}}},
		{Role: "tool", Content: "18C"},
		{Role: "tool", Content: "21C"},
	}, tools)
	want := "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
		`{"type": "function", "function": {"name": "get_weather", "description": "Weather now <in> a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}` +
		"\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
		"<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" +
		"<|im_start|>user\nWeather in Paris and Rome?<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n" +
		"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n" +
		"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call><|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n21C\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
	// without tools, the plain template
	if ChatPromptTools([]Message{{Role: "user", Content: "hi"}}, nil) != "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" {
		t.Error("no tools: wrong prompt")
	}
}

func TestPyJSON(t *testing.T) {
	for in, want := range map[string]string{
		`{"b":1,"a":[true,null,"é<>"],"c":{}}`: `{"b": 1, "a": [true, null, "é<>"], "c": {}}`,
		`[]`:                                   `[]`,
		`1.50`:                                 `1.50`,
		`not json`:                             `not json`,
	} {
		if got := PyJSON([]byte(in)); got != want {
			t.Errorf("%s: got %s, want %s", in, got, want)
		}
	}
}
