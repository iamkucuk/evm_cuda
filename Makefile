# Makefile for EVM CUDA — install, test, run, profile.
#
# Quick start:
#   make install-dev    # editable install + dev/build tooling (do this first)
#   make build          # rebuild after touching cuda/ (needs nvcc for the GPU part)
#   make test           # all 111 tests (63 of them need a GPU)
#   make run-color      # magnify pulse on face.mp4
#   make profile        # CPU vs FP32 vs FP16 comparison
#   make help           # list all targets
#
# All targets are phony (no output file artifacts).
#
# Every target runs $(PYTHON), which defaults to whatever `python` PATH resolves
# to. install-dev and build INSTALL into that interpreter, so both depend on
# check-env, which refuses any interpreter that is not an isolated environment.
# Activate the project environment first, or name the interpreter:
#
#   make install-dev PYTHON=.venv/bin/python
#
# pip is always invoked as `$(PYTHON) -m pip`, so pip and the target interpreter
# cannot disagree about where the install lands.
#
# There is no PYTHONPATH here any more: `evm` and the deprecated `evm_cuda`
# shim come from the editable install, which is the same code path a user gets
# from `pip install evm-magnify`.

.PHONY: help check-env install-dev build clean download \
        test test-baseline test-cuda \
        run-color run-motion \
        profile

# --- Paths + variables ------------------------------------------------------
# CURDIR, not PWD: PWD is the invoking shell's directory and is wrong under
# `make -C`, which would point `clean` at a different checkout.
ROOT     := $(CURDIR)
SCRIPTS  := $(ROOT)/scripts
DATA     := $(ROOT)/data
OUTPUT   := $(ROOT)/output

# Override on the command line:
#   make build PYTHON=.venv/bin/python
PYTHON   ?= python

# --- Help -------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Environment guard ------------------------------------------------------
# PYTHON defaults to plain `python`, which on any machine with conda on PATH is
# the conda BASE interpreter — install-dev would then install this project into
# conda base without a word. Refuse instead.
# Accepted: a venv (sys.prefix differs from sys.base_prefix), and a named conda
# env (has conda-meta/ but none of the root-only condabin|envs|pkgs directories
# — that is how the GPU box's evm-cuda env is laid out).
# Rejected: conda base, system and Homebrew interpreters.
check-env: ## Fail unless $(PYTHON) is an isolated env (venv or named conda env)
	@$(PYTHON) -c 'import os, sys; \
	p = sys.prefix; \
	conda = os.path.isdir(p + "/conda-meta"); \
	root = any(os.path.isdir(p + "/" + d) for d in ("condabin", "envs", "pkgs")); \
	raise SystemExit(0 if p != sys.base_prefix or (conda and not root) else \
	"\n  REFUSING TO INSTALL into " + sys.executable + "\n" \
	"  prefix: " + p + "\n" \
	"  That is not a virtual environment or a named conda env, so this would\n" \
	"  install evm-magnify into a system or conda-base interpreter.\n\n" \
	"  Activate the project environment, or name the interpreter:\n" \
	"      source .venv/bin/activate && make <target>\n" \
	"      make <target> PYTHON=.venv/bin/python\n")'

# --- Install / build --------------------------------------------------------
# One-time bootstrap. The isolated build lets pip fetch the backend itself, and
# [cuda-build] then leaves that toolchain in the venv so `make build` can skip
# isolation. Succeeds without nvcc — you get the CPU-only package.
install-dev: check-env ## Editable install with the dev + CUDA build tooling
	$(PYTHON) -m pip install -e ".[dev,cuda-build]" -v

# Rebuild after editing cuda/. --no-build-isolation reuses the toolchain that
# install-dev put in the venv instead of re-downloading it every time, so
# install-dev has to have run at least once. If it has not, pip stops with
# "BackendUnavailable: Cannot import 'scikit_build_core.build'" — run
# `make install-dev` and try again.
build: check-env ## Rebuild the package (compiles _evm_cuda when nvcc is present)
	$(PYTHON) -m pip install -e . --no-build-isolation -v

# Two things a build can leave inside the checkout, both gitignored:
#
#   src/evm/cuda/_evm_cuda*.so — cuda/CMakeLists.txt sets LIBRARY_OUTPUT_DIRECTORY
#     to ../src/evm/cuda, and pip builds in-tree, so BOTH `pip install .` and
#     `pip install -e .` write it there (measured on the GPU box 2026-08-09:
#     _evm_cuda.cpython-312-x86_64-linux-gnu.so, identical in both modes).
#   cuda/build/ — NOT produced by pip: scikit-build-core configures CMake in a
#     temporary directory. It is produced by the direct CMake entry point that
#     cuda/CMakeLists.txt:3 documents (`cmake -S cuda -B cuda/build`), which is
#     why .gitignore still lists it. Removing it here is a no-op after a pip
#     build and the whole point after a manual one.
#
# The wheel's own copy under site-packages/evm/cuda/ is outside the repo and
# stays out of scope: `pip uninstall evm-magnify` owns that one. No *.pyd glob —
# Windows is untested and has never produced one here.
clean: ## Delete build artifacts from the source tree (.so + cuda/build)
	rm -f $(ROOT)/src/evm/cuda/_evm_cuda*.so
	rm -rf $(ROOT)/cuda/build

download: ## Download MIT sample videos + reference outputs
	$(PYTHON) $(SCRIPTS)/download_samples.py face baby --with-references

# --- Tests ------------------------------------------------------------------
# tests/cuda/ is a subdirectory of tests/, so pytest de-duplicates the two
# arguments below: 111 cases total, of which tests/cuda/ contributes 63.
test: ## All tests: Python baseline + CUDA kernels (111 cases)
	$(PYTHON) -m pytest tests/ tests/cuda/ -q

# The 63 CUDA cases skip themselves without a GPU, which is what makes this
# runnable anywhere; on a GPU host it is the same run as `make test`.
test-baseline: ## Everything runnable without a GPU (48 pass, 63 CUDA cases skip)
	$(PYTHON) -m pytest tests/ -q

test-cuda: ## CUDA kernel tests vs Python baseline (needs GPU)
	$(PYTHON) -m pytest tests/cuda/ -v

# --- Run pipelines ----------------------------------------------------------
run-color: ## Color magnification on face.mp4 (pulse)
	mkdir -p $(OUTPUT)
	$(PYTHON) $(SCRIPTS)/run_evm.py $(DATA)/face.mp4 $(OUTPUT)/face_color.mp4 \
		--mode color --alpha 50 --level 4 --fl 0.8333 --fh 1.0 --chromatt 1

run-motion: ## Motion magnification on baby.mp4 (IIR)
	mkdir -p $(OUTPUT)
	$(PYTHON) $(SCRIPTS)/run_evm.py $(DATA)/baby.mp4 $(OUTPUT)/baby_motion.mp4 \
		--mode iir --alpha 10 --lambda-c 16 --r1 0.4 --r2 0.05 --chromatt 0.1

# --- Profiling --------------------------------------------------------------
profile: ## Full CPU vs FP32 vs FP16 comparison + render all videos
	$(PYTHON) $(SCRIPTS)/profile_full_comparison.py
