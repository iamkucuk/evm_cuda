# Security

## What this software does

It reads video files and processes them, on the processor or on a graphics
processor. It opens no network connections and starts no services. The
realistic risk is a malformed video file reaching the decoder.

## Where the risk actually is

Decoding is done by OpenCV and PyAV, which wrap FFmpeg. A malformed file that
causes a problem there is a vulnerability in those projects, not this one, and
should be reported to them. Keeping them up to date is the useful protection.

The compiled CUDA and OpenCL code assumes the shapes it is given are consistent
with the buffers behind them. The public operations check that; the compiled
module beneath does not, and is not a supported interface for exactly that
reason.

## Reporting something

Open a
[security advisory](https://github.com/iamkucuk/eulerian-video-magnification-cuda/security/advisories/new)
rather than a public issue, and please include a file or script that reproduces
it.

This is a research project maintained by one person. There is no guaranteed
response time, and no security support for older versions: fixes go onto the
current one.

## Supported versions

The most recent release only.
