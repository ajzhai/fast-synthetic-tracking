# fast-synthetic-tracking

This repository implements Fast Synthetic Tracking (FST), an algorithm for detecting faint objects moving at constant rates by efficiently integrating image frames along all linear motion hypotheses.

## Installation

Create and activate a conda environment, then install the dependencies:

```bash
conda create -n fst python=3.11
conda activate fst
pip install -r requirements.txt
```

## Usage

The core algorithm is in `fst.py`, and `simulation.py` contains helpers for generating synthetic test data.
`FastLinRateSynTracking` takes a datacube (a 3D numpy array of shape `(Nt, Nx, Ny)`) and allocates about twice its memory for frame integration.
`lrstSearchAndDetect` searches all motion rates within one velocity direction and returns detections above a threshold as `((vx, vy), [(x, y, intensity), ...])`.

```python
import numpy as np
from fst import FastLinRateSynTracking, lrstSearchAndDetect

lrst = FastLinRateSynTracking(datacube)
Nt = datacube.shape[0]
kernel = np.ones((3, 3)) / 3
detections = []
for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    lrst.setVelDir(direction)
    lrstSearchAndDetect(lrst, (0, Nt, Nt, 0, 0, 0, 0), 100, kernel=kernel, detectedSig=detections)
```

Velocities are reported in units of pixels per `2**ceil(log2(Nt))` frames.
To compute the integrated image for one specific rate (in pixels/frame), use `performFST(lrst, (vx, vy))`.
Run `python example.py` for a complete demo that detects six simulated moving objects and plots their recovered positions and rates.
The demo searches a 256×256×16 datacube and takes about 10 seconds on a typical desktop; runtime grows with image area and roughly with the square of the number of frames.
Alternatively, `FST_v0.ipynb` is a self-contained Jupyter notebook with the same code and a larger 1000×1000×32 demo that takes several minutes (install Jupyter with `pip install notebook`).
