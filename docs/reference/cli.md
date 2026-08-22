# Command line

Installing the library puts `vidmag` on your path.

```
vidmag --help
```

## `magnify`

Run a pipeline over a video file.

```bash
vidmag magnify data/face.mp4 pulse.mp4 --preset pulse
```

Useful options:

| Option | Meaning |
|---|---|
| `--preset NAME` | Which preset to run. Required. |
| `--backend NAME` | Force a backend instead of choosing automatically |
| `--alpha`, `--fl`, `--fh`, `--lambda-c` | Override individual preset values |
| `--fps` | Override the frame rate read from the file |

The backend that was selected is printed before the work starts.

## `download`

Fetch the sample clips used by the original authors. Needs a checkout of the
repository, since it wraps a script that is not part of the installed package.

```bash
vidmag download face baby
```

## `bench`

Time each stage on a graphics processor, for comparing configurations.

```bash
vidmag bench --help
```
