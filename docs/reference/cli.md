# Command line

Installing the library puts `vidmag` on your path. It has four subcommands.

```
vidmag --help
```

## `magnify`

Run a pipeline over a video file.

```bash
vidmag magnify data/face.mp4 pulse.mp4 --preset pulse
```

The first two arguments are the input and output paths. Then choose a pipeline
one of two ways: `--preset NAME` for a named parameter set, or `--mode NAME`
with the individual values (`--mode` requires `--alpha`). One or the other is
required, not both.

| Option | Meaning |
|---|---|
| `--preset {pulse,motion,motion_phase,vibration}` | A named parameter set |
| `--mode {color,motion,butter,iir}` | A pipeline by name; needs `--alpha` and the values it uses |
| `--backend NAME` | Force a backend instead of choosing automatically |
| `--alpha`, `--level`, `--lambda-c`, `--fl`, `--fh`, `--r1`, `--r2`, `--chromatt` | Override individual values |
| `--sampling-rate` | Override the sampling rate in Hz; defaults to the input's frame rate |

The selected backend is printed before the work starts. A backend that has no
file-in/file-out pipeline of its own — the Metal, Vulkan, OpenCL and PyTorch
backends — has the reading and writing done for it, so `magnify` works on those
backends too, for the four pipelines they implement. The `motion_phase` preset
is the processor-only exception, and asking another backend for it raises.

## `stream`

Magnify a camera or file frame by frame, motion only. `0` is the first camera.

```bash
vidmag stream 0 --display
vidmag stream 0 --out session.mp4 --max-frames 300
```

`--display` shows a window (press `q` to stop); `--out` writes a file. See
[magnify a live feed](../recipes/streaming.md).

## `download`

Fetch the sample clips the original authors used. Needs a checkout of the
repository, since it wraps a script that is not part of the installed package.

```bash
vidmag download face baby
```

## `bench`

Time each stage on a graphics processor, for comparing configurations.

```bash
vidmag bench --help
```
