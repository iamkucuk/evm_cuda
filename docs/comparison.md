# Compared with other implementations

Several implementations of this method exist. This page says plainly what is
different about this one, including where others are the better choice.

| | This project | Typical other implementations |
|---|---|---|
| Installable with `pip` | Yes | Sometimes |
| Checked against the authors' published output | Yes, automatically, on every change | Rarely |
| Runs on NVIDIA hardware | Yes, hand-written and tuned | Rarely |
| Runs on Apple, AMD, Intel hardware | Yes, through OpenCL | Rarely |
| Building blocks usable separately | Yes, on every backend | Sometimes |
| Continuous testing | Yes, on every commit | Rarely |
| Licence | Research and non-commercial | Often permissive |

## What is genuinely different here

**The comparison is the product.** The NumPy implementation is checked against
the original authors' own rendered output, and both graphics implementations
are then checked, operation by operation, against that NumPy version. A
magnification nobody has checked is a picture, not a measurement.

**It runs on hardware other implementations do not reach.** Most are processor
only. This one has a tuned NVIDIA path and a portable path covering Apple, AMD
and Intel.

**The parts are usable separately.** The pyramids, the filters and the colour
conversion are public, on every backend, so it can be a component in something
larger rather than only a program that makes a video.

## Where something else may suit you better

**If you need a permissive licence.** This one is restricted to non-commercial
use and cannot be otherwise, for the reasons on the
[licence page](licence.md). A cleanroom implementation under a permissive
licence would be the answer.

**If you want phase-based magnification.** The 2013 follow-up method produces
cleaner results at high amplification. It is not implemented here.

**If you want a graphical program.** This is a library and a command-line tool.

## Not claimed

No benchmark against other implementations is published here. Comparing speed
fairly means running everything on the same hardware with the same clip and
parameters, and that has not been done. The [performance](performance.md) page
reports only measurements of this project against its own reference.
