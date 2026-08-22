# The self-hosted GPU runner

How `.github/workflows/gpu.yml` gets a machine to run on, written for someone who has never
registered a GitHub Actions runner. Companion to `docs/dev/packaging-notes.md` §3, which
records how the GPU box was provisioned; nothing here contradicts it, and where a fact comes
from that file it is cited rather than restated differently.

> **Status, 2026-08-09: no runner is registered.** `~/actions-runner` does not exist on the
> machine and the repository has no runner in Settings → Actions → Runners. `gpu.yml` has
> therefore **never executed**. What has been proven is narrower and is stated exactly at the
> bottom: every step of that job was run by hand over SSH, in order, on the RTX 3090, and its
> commands produced the expected results. Registering the runner is the remaining step, and it
> needs a person at the keyboard — it takes a registration token that only the repository's
> Settings page can mint.

## Why a self-hosted runner at all

GitHub's hosted runners have no NVIDIA GPU. On them the entire `tests/cuda/` suite skips
itself (the extension is not built, so `import vidmag.cuda._vidmag_cuda` fails) and `pytest` still
exits 0. Both figures measured on 2026-08-09 on the `library-restructure` branch:

| Host | Result |
|---|---|
| macOS, no CUDA toolchain | 124 passed, **70 skipped** |
| `osiris`, RTX 3090 | **194 passed, 0 skipped** |

Those 70 cases are the project's real correctness claim — every CUDA kernel compared against
the CPU oracle. A pull-request CI that never runs them proves nothing about the port, which is
why plan decision D6 puts them on a self-hosted runner attached to the 3090, gated on manual
dispatch and on release rather than on every push.

## The machine, and what it cannot do

The GPU box is `osiris`: a Windows desktop (`DESKTOP-4LO089U`) running WSL2, distribution
"Pengwin", reached with `ssh osiris` over Tailscale. Full provisioning record in
`docs/dev/packaging-notes.md` §3. The five constraints that shape both the workflow and this
document:

**1. No administrator rights in the automated sessions.** `sudo -n true` answers
`sudo: a password is required` — a password prompt, not a refusal, so a person who knows the
password can still install system packages or a service; the non-interactive SSH sessions this
project runs from cannot. Anything the runner needs must work from userspace.

**2. No system compiler.** `/usr/bin/gcc`, `/usr/bin/g++` and `/usr/bin/c++` do not exist.
The whole toolchain lives in a conda environment created for this project at
`/home/furkan/miniconda3/envs/evm-cuda` — verified there on 2026-08-09:

```
python       /home/furkan/miniconda3/envs/evm-cuda/bin/python      3.12.13
nvcc         /home/furkan/miniconda3/envs/evm-cuda/bin/nvcc        12.9.86
cmake        /home/furkan/miniconda3/envs/evm-cuda/bin/cmake       4.4.2
ninja        /home/furkan/miniconda3/envs/evm-cuda/bin/ninja
g++          /home/furkan/miniconda3/envs/evm-cuda/bin/g++         (gxx_linux-64)
```

That environment is also somebody's interactive workspace, and it already holds a
non-editable install of `evm-cuda` pointing at `~/evm_cuda_dev`. The job therefore never
installs into it: it builds a throwaway venv under `$RUNNER_TEMP` and runs a plain
`pip install ".[dev]"` there, so CI cannot disturb hand testing and hand testing cannot
flatter CI.

**3. `nvidia-smi` is not on `PATH`.** Under WSL it exists only at
`/usr/lib/wsl/lib/nvidia-smi`. Both directories are prepended to `PATH` by the workflow's
first step, which then fails loudly, naming what is missing, if any of the six tools is not
found.

**4. WSL2 passes through CUDA only.** `/usr/lib/wsl/lib/libcuda.so` is there; OpenCL and
Vulkan are not. Nothing in this job needs them — it matters for Phase 4V, whose portable
backends have to be tested on the machine's *Windows* side instead.

**5. The runner is only up while WSL is up.** It stops whenever the Windows host sleeps,
hibernates or reboots, and does not come back until Windows is awake and WSL has been
started again. A job dispatched at a bad moment sits in "Queued", possibly for hours; it is
not lost, and it starts on its own when the runner reconnects. This is the whole reason
`gpu.yml` has no `on: push` trigger — a per-commit gate that stalls for hours is a gate
people route around. If it proves too flaky in practice, D6's named fallback is a cloud GPU
runner on release tags only.

One more caveat, from `packaging-notes.md` §3: this GPU is reached through the WSL
virtualization layer, so host↔device transfer timings differ from native Linux. Correctness
is unaffected; any benchmark number the job publishes must state the platform.

## Registering the runner

Do this once, at the machine or from an interactive SSH session (the token step needs a
browser, and `./run.sh` needs a place to keep running).

1. **Mint a registration token.** In the repository on GitHub: **Settings → Actions →
   Runners → New self-hosted runner**, then choose **Linux** / **x64**. The page prints a
   download-and-configure script containing a token. That token expires in about an hour and
   is a credential — never paste it into a file in this repository, an issue, or a chat.

2. **Download the runner** on `osiris`, inside WSL, using the exact `curl`/`tar` lines from
   that page. They carry the current runner version and its checksum, which is why they are
   not copied here — a hardcoded version in a document goes stale and a stale checksum reads
   as tampering.

   ```bash
   ssh osiris
   mkdir -p ~/actions-runner && cd ~/actions-runner
   # paste the curl + shasum + tar lines from the New self-hosted runner page
   ```

3. **Configure it, with the `cuda` label.** This is the one place where you must deviate from
   what GitHub prints:

   ```bash
   ./config.sh \
     --url https://github.com/iamkucuk/eulerian-video-magnification-cuda \
     --token <THE TOKEN FROM STEP 1> \
     --name osiris-3090 \
     --labels cuda \
     --work _work
   ```

   * `--labels cuda` — **required.** `gpu.yml` asks for
     `runs-on: [self-hosted, linux, x64, cuda]`. GitHub applies `self-hosted`, `linux` and
     `x64` automatically; `cuda` is the one that says "this machine actually has a GPU", and
     without it the job never matches a runner and waits forever. `.github/actionlint.yaml`
     declares the same label so the linter accepts the workflow.
   * `--name osiris-3090` — what appears in the Runners list.
   * `--work _work` — the checkout directory, under `~/actions-runner/_work`. It grows: the
     repository, plus a fresh venv per run. 532 GB were free on 2026-08-09.

4. **Start it.**

   ```bash
   cd ~/actions-runner && ./run.sh
   ```

   That is a foreground process; close the terminal and the runner goes offline. For
   something that survives a disconnect, run it under `tmux` (or `nohup ./run.sh &`).
   Installing it as a service (`sudo ./svc.sh install && sudo ./svc.sh start`) is possible
   here — `systemd` is PID 1 on this distribution, though `systemctl is-system-running`
   reports `degraded` — but it needs the sudo password, so it is the operator's call and has
   not been tried. Either way, constraint 5 stands: nothing runs while Windows is asleep.

5. **Check it.** Settings → Actions → Runners should show `osiris-3090` as **Idle** with the
   labels `self-hosted, linux, x64, cuda`. Then run the job: **Actions → GPU suite → Run
   workflow**.

## What the job does, in order

Read `.github/workflows/gpu.yml` for the exact commands; this is the map.

| Step | What it proves |
|---|---|
| Put the CUDA toolchain on `PATH` | `python`, `nvcc`, `cmake`, `ninja`, `g++`, `nvidia-smi` all resolve; names what is missing otherwise |
| Record which machine this ran on | `host.txt`: commit, GPU name/VRAM/driver/compute capability, nvcc and Python versions |
| Install into a throwaway venv | `pip install ".[dev]"` compiles `_vidmag_cuda`; `VIDMAG_CUDA_REQUIRE=1` makes a missing nvcc a build error instead of a quiet CPU-only package |
| The extension must have compiled | `have_cuda` is True and `require_cuda()` returns; prints the `.so` path |
| Fetch the MIT sample clips | `tests/test_against_mit_reference.py` needs `data/*.mp4`, and `actions/checkout` wipes untracked files, so this runs every time |
| Run the full suite | `pytest tests/ -q -p no:randomly --junitxml=…` |
| **No test may be skipped** | The gate. Reads the JUnit XML, writes `counts.json`, and fails listing every skipped case |
| Benchmark | `vidmag bench` on `face.mp4` (pulse) and `baby.mp4` (motion), FP32 and FP16 |
| Publish the evidence | Uploads `gpu-evidence/` as an artifact, `if: always()` |

The evidence leaves as an artifact — text plus the JUnit XML — and nothing is written back to
the repository. `PLAN.md` step 2.4 words the deliverable as "benchmark JSON to `benches/`",
with the success criterion "a dispatched GPU run commits `benches/*.json` with matching SHA";
**this job does not do that.** `vidmag bench` prints a table and has no JSON output, and
a job that pushes commits needs `contents: write` and a bot identity. Both are decisions for
whoever closes Phase 2; until they are taken, that success criterion is unmet and the numbers
live in the run's artifact.

Two job-level environment variables encode everything machine-specific, so moving to another
GPU host means editing two lines:

```yaml
VIDMAG_TOOLCHAIN_BIN: /home/furkan/miniconda3/envs/evm-cuda/bin
VIDMAG_WSL_LIB: /usr/lib/wsl/lib
```

### Why the skip gate exists

`pytest` exits 0 when tests skip. On a GPU host a skip is never benign: a skipped
`tests/cuda/…` case means the extension did not build or is not importable, and a skipped
`test_against_mit_reference` case means a sample clip is missing — `scripts/download_samples.py`
reports a failed fetch and still returns 0, so this gate, not the download step, is what
catches that. Expected here: **0 skipped, 194 passed**.

## When it goes wrong

| Symptom | Meaning |
|---|---|
| Job stuck in **Queued** | No runner with all four labels is online. Usually the Windows host is asleep; otherwise the runner was configured without `--labels cuda` |
| `::error::not on PATH: nvcc …` | The conda environment moved or was renamed. Fix `VIDMAG_TOOLCHAIN_BIN`, or re-create it per `packaging-notes.md` §3 |
| CMake error naming `VIDMAG_CUDA_REQUIRE` | `nvcc` is gone. That is the flag doing its job: the alternative is a silent CPU-only install |
| `FAIL: the extension did not build or does not import` | The wheel built but `_vidmag_cuda` will not load — a real regression; the install log is in the artifact |
| Gate lists `tests.cuda.*` skips | The extension is not importable inside the job's venv |
| Gate lists `test_against_mit_reference` skips | The MIT download failed (their server, or no network) |
| Cancelled at 60 minutes | The suite takes ~163 s on this GPU; a timeout means a hang, not slowness |

## What has actually been verified, and what has not

**Not verified, and cannot be from here:** that GitHub schedules this workflow onto the
runner, that `actions/checkout@v4` and `actions/upload-artifact@v4` behave on this host, and
that `$GITHUB_PATH` carries `PATH` between steps as documented. All of that needs a
registered runner and a real dispatch.

**Verified on 2026-08-09**, at commit `ed1611b` (the tree synced to `~/evm_cuda_dev` was
byte-identical to `HEAD` for every file):

* `actionlint 1.7.12` with `shellcheck 0.11.0` on `PATH`, so the shell sub-linter really ran:
  **0 errors** on `gpu.yml` (and on `deploy-pages.yml`). Its `pyflakes` rule checked nothing
  here, and cannot: actionlint only lints Python in steps declared `shell: python`, and this
  job's Python runs inside `python - <<'PY'` heredocs. Those two scripts are covered by
  having been executed instead — both halves of the gate, see below.
* Every step of the job except `checkout` and `upload-artifact` executed by hand over SSH on
  the RTX 3090, in the same order, with `PATH` carried forward between them the way
  `$GITHUB_PATH` would. Results: toolchain check passed; `pip install ".[dev]"` built
  `evm_cuda-0.1.0-cp312-cp312-linux_x86_64.whl` and `have_cuda` came back `True`; the four
  MIT clips downloaded; **194 passed in 162.94s**; the gate reported
  `{"tests": 194, "failures": 0, "errors": 0, "skipped": 0, "passed": 194}` and exited 0;
  both benchmarks produced per-stage tables (`color` FP32 9 ms compute + 66 ms transfer,
  `motion` FP32 75 ms + 108 ms, on `NVIDIA GeForce RTX 3090`).
* The gate's failing half was exercised too, against the JUnit XML from the CPU-only Mac run:
  it exited 1 and listed all 70 skipped cases with their reasons.
