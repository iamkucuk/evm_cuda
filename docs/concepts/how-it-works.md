# How it works

The method rests on one observation: to make a small change visible, you do not
have to find what moved. You can amplify how each part of the picture changes
over time, and the movement follows.

That is what "Eulerian" means here. The alternative — tracking each feature
from frame to frame and exaggerating the path it took — is harder, slower, and
fails when the movement is smaller than a pixel. Watching a fixed point and
amplifying how it changes needs no tracking at all.

## The four stages

Every pipeline in this library does the same four things.

**1. Separate brightness from colour.** The frames are converted so that one
channel carries brightness and two carry colour. This matters because the two
kinds of change want different treatment: a pulse is mostly colour, a movement
is mostly brightness at edges, and being able to amplify them by different
amounts is what keeps the result usable.

**2. Split by scale.** The image is reduced, repeatedly. What is kept depends
on the job:

- For **colour** changes, only the reduced image is kept. Shrinking averages
  over a region, and a heartbeat changes a whole region of skin together while
  sensor noise does not — so the shrinking keeps the signal and discards much
  of the noise.
- For **motion**, the difference between each size and the next is kept. That
  gives a stack of images, each holding the detail at one scale. Movement shows
  up as change in these detail layers.

**3. Filter over time.** Each pixel is now a series of values through time.
Keeping only a band of frequencies discards everything changing too slowly or
too quickly to be what you are looking for. A heartbeat is around one cycle a
second; a guitar string is eighty.

**4. Amplify and add back.** What survives the filter is multiplied and added
to the original. The output is the original video with the selected change
made large.

## Why amplifying detail moves things

For motion this seems to prove too much: multiplying image detail should make
the picture brighter and darker, not make things move.

It works because of what a small shift does to an image. Shift a picture by a
distance far smaller than its detail, and the difference is very nearly the
image's spatial slope multiplied by the shift. Adding a multiple of that
difference back is therefore nearly the same as shifting further. The
approximation is good while the movement is small compared with the detail, and
it is why `lambda_c` exists: fine detail is where the approximation breaks
first, so fine detail is amplified less.

This also explains the failure everyone meets eventually. Push `alpha` high
enough and edges tear into ripples and haloes — that is the approximation
falling apart, not a bug.

## What the amplification schedule does

The motion pipelines do not amplify every scale by the same amount. They follow
a schedule from the original paper: full amplification at coarse scales, tapering
to none at the finest, with the crossover set by `lambda_c`. The coarsest and
finest layers are set to zero outright. This is the single largest reason the
published results look clean rather than shimmering.
