package main

import "testing"

func TestLoopback(t *testing.T) {
	for addr, want := range map[string]bool{"127.0.0.1:11434": true, "[::1]:11434": true, "localhost:11434": true,
		"127.0.0.2:1": true, "0.0.0.0:11434": false, ":11434": false, "[::]:11434": false, "192.168.1.5:11434": false,
		"example.com:80": false, "nonsense": false} {
		if got := loopback(addr); got != want {
			t.Errorf("loopback(%q) = %v, want %v", addr, got, want)
		}
	}
}
