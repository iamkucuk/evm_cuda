# Makefile for EVM CUDA — build, test, run, profile.
#
# Quick start:
#   make build          # compile the _evm_cuda.so extension (needs nvcc)
#   make test           # all 125 tests (needs GPU for CUDA suite)
#   make run-color      # magnify pulse on face.mp4
#   make profile        # CPU vs FP32 vs FP16 comparison
#   make help           # list all targets
#
# All targets are phony (no output file artifacts). The venv must already
# be active (`.venv` locally, or `source deploy/truba_env.sh` on TRUBA).

.PHONY: help build build-truba clean download \
        test test-baseline test-cuda \
        run-color run-motion \
        profile-color profile-motion profile slurm

# --- Paths + variables ------------------------------------------------------
ROOT     := $(PWD)
CUDA_DIR := $(ROOT)/cuda
SCRIPTS  := $(ROOT)/scripts
DATA     := $(ROOT)/data
OUTPUT   := $(ROOT)/output

export PYTHONPATH := $(CUDA_DIR)

# Defaults that can be overridden on the command line:
#   make profile N=10
N        ?= 5

# --- Help -------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Build ------------------------------------------------------------------
build: ## Build _evm_cuda.so via CMake + nvcc (any machine with nvcc)
	cmake -S $(CUDA_DIR) -B $(CUDA_DIR)/build \
		-DCMAKE_BUILD_TYPE=Release -G Ninja 2>/dev/null || \
	cmake -S $(CUDA_DIR) -B $(CUDA_DIR)/build -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CUDA_DIR)/build --config Release -j

build-truba: ## Build on TRUBA (loads gcc/cuda/cmake modules first)
	source deploy/truba_env.sh && bash deploy/build.sh

clean: ## Remove the CMake build directory
	rm -rf $(CUDA_DIR)/build

download: ## Download MIT sample videos + reference outputs
	python $(SCRIPTS)/download_samples.py face baby --with-references

# --- Tests ------------------------------------------------------------------
test: ## All tests: Python baseline + CUDA kernels (125 total)
	python -m pytest tests/ tests/cuda/ -q

test-baseline: ## Python baseline only (no GPU required, ~40s)
	python -m pytest tests/ -q

test-cuda: ## CUDA kernel tests vs Python baseline (needs GPU)
	python -m pytest tests/cuda/ -v

# --- Run pipelines ----------------------------------------------------------
run-color: ## Color magnification on face.mp4 (pulse)
	mkdir -p $(OUTPUT)
	python $(SCRIPTS)/run_evm.py $(DATA)/face.mp4 $(OUTPUT)/face_color.mp4 \
		--mode color --alpha 50 --level 4 --fl 0.8333 --fh 1.0 --chromatt 1

run-motion: ## Motion magnification on baby.mp4 (IIR)
	mkdir -p $(OUTPUT)
	python $(SCRIPTS)/run_evm.py $(DATA)/baby.mp4 $(OUTPUT)/baby_motion.mp4 \
		--mode iir --alpha 10 --lambda-c 16 --r1 0.4 --r2 0.05 --chromatt 0.1

# --- Profiling --------------------------------------------------------------
profile-color: ## Color pipeline FP32 stage breakdown
	python $(SCRIPTS)/profile_color.py $(N)

profile-motion: ## Motion pipeline FP32 stage breakdown
	python $(SCRIPTS)/profile_motion.py $(N)

profile: ## Full CPU vs FP32 vs FP16 comparison + render all videos
	python $(SCRIPTS)/profile_full_comparison.py

# --- TRUBA -------------------------------------------------------------------
slurm: ## Submit TRUBA SLURM job (build + test + profile in one job)
	sbatch deploy/submit_profile.slurm
