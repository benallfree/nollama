#!/bin/sh
# Regenerates gpu_cuda.ptx from gpu.cu with nvcc, in NVIDIA's CUDA container (no GPU needed), so only the
# developer who changes gpu.cu needs Docker; builds and users never do. Run: go generate ./engine
set -e
cd "$(dirname "$0")"
IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
if command -v nvcc >/dev/null 2>&1; then
	nvcc -ptx -arch=compute_75 -O3 -o gpu_cuda.ptx gpu.cu
else
	docker run --rm -v "$PWD":/src -w /src "$IMAGE" nvcc -ptx -arch=compute_75 -O3 -o gpu_cuda.ptx gpu.cu >/dev/null
fi
echo "gpu_cuda.ptx: $(grep -c '^.visible .entry' gpu_cuda.ptx) kernels"
