---
name: Result from hardware we cannot test
about: Report running the suite on an AMD, Intel or other device
title: "Hardware result: "
labels: hardware-report
---

This project has an NVIDIA card and an Apple machine to test on, and nothing
else. Support for other hardware is described as expected rather than verified
because nobody has run it. A report here changes that.

**The device**

- Name as reported (`python -c "from vidmag.opencl import runtime; print(runtime.device_name())"`):
- Operating system and version:
- Driver version:

**Test suite**

```
python -m pytest tests/ -q
```

Paste the output, including the skip count.

**Anything that failed**

Paste the failure in full.

**Speed, if you measured it**

The steps are in `benches/apple_m2_max_opencl_2026-08-10.md`.
