package main

import (
	"testing"
	"time"
)

func TestParseKeepAlive(t *testing.T) {
	for in, want := range map[string]time.Duration{"300": 300 * time.Second, "5m": 5 * time.Minute, "-1": -time.Second, "0": 0, "1h30m": 90 * time.Minute} {
		if d, err := parseKeepAlive(in); err != nil || d != want {
			t.Errorf("%s: %v %v, want %v", in, d, err, want)
		}
	}
	if _, err := parseKeepAlive("soon"); err == nil {
		t.Error("soon: want an error")
	}
}
