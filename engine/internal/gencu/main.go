// Command gencu writes gpu_int.cu from gpu_int.metal: the integer models' kernels are written once, in Metal's
// language, and this turns them into CUDA (with gpu_int_prelude.cuh supplying Metal's types and simdgroup functions).
// It rewrites kernel signatures (builtin thread indices become locals, constant references become values, address
// spaces go), threadgroup arrays (__shared__) and the few Metal calls the prelude doesn't cover.
//
//	go run ./internal/gencu gpu_int.metal gpu_int.cu
package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: gencu gpu_int.metal gpu_int.cu")
		os.Exit(2)
	}
	src, err := os.ReadFile(os.Args[1])
	if err != nil {
		fail(err)
	}
	out, err := translate(string(src))
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(os.Args[2], []byte(out), 0o644); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gencu:", err)
	os.Exit(1)
}

var (
	kernelRe = regexp.MustCompile(`kernel void (\w+)\(`)
	sharedRe = regexp.MustCompile(`(?m)^(\s*)threadgroup (\w+) (\w+(\[[^\]]+\])*);`)
	charRe   = regexp.MustCompile(`\b(const )?char \*`)
	addrRe   = regexp.MustCompile(`\b(?:device|constant) ((?:const |atomic_int |\w+ ?[*&]|\w+ \w+ ?[*&]|\w+))`)
	builtins = map[string]map[string]string{ // attribute -> type -> expression
		"thread_index_in_threadgroup":  {"uint": "threadIdx.x"},
		"threadgroup_position_in_grid": {"uint": "blockIdx.x", "uint2": "make_uint2(blockIdx.x, blockIdx.y)", "uint3": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)"},
		"thread_position_in_grid": {"uint": "blockIdx.x * blockDim.x + threadIdx.x",
			"uint2": "make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)"},
		"simdgroup_index_in_threadgroup": {"uint": "threadIdx.x / 32"},
		"thread_index_in_simdgroup":      {"uint": "threadIdx.x % 32"},
		"threads_per_threadgroup":        {"uint": "blockDim.x"},
		"threadgroups_per_grid":          {"uint": "gridDim.x"},
	}
	paramRe = regexp.MustCompile(`^(.*?)\s*\[\[(\w+)(\(\d+\))?\]\]$`)
)

func translate(s string) (string, error) {
	var b strings.Builder
	b.WriteString("// Generated from gpu_int.metal by internal/gencu (go generate ./engine). Do not edit.\n\n")
	b.WriteString("#include \"gpu_int_prelude.cuh\"\n\n")
	s = strings.Replace(s, "#include <metal_stdlib>\nusing namespace metal;\n", "", 1)
	// the prelude has shx (Metal's goes through as_type)
	s = regexp.MustCompile(`(?m)^static inline long shx\(.*\n`).ReplaceAllString(s, "")
	s = strings.ReplaceAll(s, "static inline", "static __device__ __forceinline__")
	s = strings.ReplaceAll(s, "atomic_store_explicit(ovf, 1, memory_order_relaxed)", "atomicExch(ovf, 1)")
	s = strings.ReplaceAll(s, "device atomic_int *", "int *")
	s = charRe.ReplaceAllString(s, "${1}signed char *")
	// address spaces: device and constant go (CUDA's own __device__ and __constant__ expand to those words, so the
	// prelude can't define them away); threadgroup is the prelude's
	s = addrRe.ReplaceAllString(s, "$1")
	s = sharedRe.ReplaceAllString(s, "${1}__shared__ __align__(16) ${2} ${3};") // Metal aligns threadgroup arrays for vector loads
	// kernel signatures
	for {
		loc := kernelRe.FindStringSubmatchIndex(s)
		if loc == nil {
			break
		}
		name := s[loc[2]:loc[3]]
		open := loc[1] // after '('
		end := strings.Index(s[open:], ") {")
		if end < 0 {
			return "", fmt.Errorf("%s: no body", name)
		}
		params := s[open : open+end]
		inMacro := strings.Contains(params, "\\")
		params = strings.ReplaceAll(params, "\\", "")
		var args, locals []string
		for _, p := range strings.Split(params, ",") {
			p = strings.Join(strings.Fields(p), " ")
			m := paramRe.FindStringSubmatch(p)
			if m == nil {
				return "", fmt.Errorf("%s: parameter %q has no attribute", name, p)
			}
			decl, attr := m[1], m[2]
			if attr == "buffer" {
				// Metal's constant references are values in CUDA (a reference parameter would be a pointer)
				decl = strings.Join(strings.Fields(strings.Replace(decl, "&", "", 1)), " ")
				args = append(args, decl)
				continue
			}
			f := strings.Fields(decl)
			if len(f) != 2 {
				return "", fmt.Errorf("%s: builtin %q", name, p)
			}
			expr, ok := builtins[attr][f[0]]
			if !ok {
				return "", fmt.Errorf("%s: unknown builtin %s %s", name, f[0], attr)
			}
			locals = append(locals, fmt.Sprintf("%s %s = %s;", f[0], f[1], expr))
		}
		sep, nl := ", ", "\n\t"
		if inMacro {
			nl = " \\\n\t"
		}
		head := "extern \"C\" __global__ void " + name + "(" + strings.Join(args, sep) + ") {" + nl + strings.Join(locals, nl)
		s = s[:loc[0]] + head + s[open+end+len(") {"):]
	}
	b.WriteString(s)
	return b.String(), nil
}
