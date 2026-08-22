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
| `--fps` | Override the frame rate read from the file |

The selected backend is printed before the work starts.

!!! note "On Apple, AMD or Intel graphics"
    `magnify` runs only on the processor and on NVIDIA cards. On other graphics
    hardware it stops with an error naming the backend; add `--backend cpu`, or
    use the Python API `vidmag.magnify(...)`, which runs on every backend.

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
