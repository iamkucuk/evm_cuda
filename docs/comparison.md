# Compared with other implementations

Several implementations of this method exist. This says plainly what is
different about this one, including where another is the better choice.

| | This project | Typical others |
|---|---|---|
| Installable with `pip` | Yes | Sometimes |
| Checked against the authors' published output | Yes, automatically, on every change | Rarely |
| Runs on NVIDIA hardware | Yes, hand-written and tuned | Rarely |
| Runs on Apple, AMD, Intel hardware | Yes, through Metal, Vulkan or OpenCL | Rarely |
| Phase-based method (2013) | Yes, on the processor | Sometimes |
| Building blocks usable separately | Yes, on every backend | Sometimes |
| Continuous testing | Yes, on every commit | Rarely |
| Licence | Research and non-commercial | Often permissive |

## What is genuinely different

**The comparison is the product.** The NumPy version is checked against the
authors' own rendered output, and every other backend is then checked, operation
by operation, against that NumPy version. A magnification nobody has checked is a
picture, not a measurement.

**It reaches hardware other implementations do not.** Most are processor only.
This has a tuned NVIDIA path and a portable path covering Apple, AMD and Intel.

**The parts are usable separately.** The pyramids, filters and colour conversion
are public, on every backend, so it can be a component in something larger
rather than only a program that makes a video.

## Where something else may suit you better

- **You need a permissive licence.** This one is restricted to non-commercial
  use and cannot be otherwise, for the reasons on the [licence page](licence.md).
  A cleanroom implementation under a permissive licence would be the answer.
- **You want a polished phase-based implementation.** The 2013 method is here,
  on the processor, and [checked against synthetic motion rather than the
  authors' output](concepts/phase-based.md) — weaker evidence, and it has no
  graphics backend yet. If that method is your main need, a dedicated
  implementation may be more complete.
- **You want a graphical program.** This is a library and a command-line tool.

## Not claimed

No benchmark against other implementations is published here. A fair speed
comparison means running everything on the same hardware with the same clip and
parameters, and that has not been done. The [performance](performance.md) page
reports only this project measured against its own reference.
