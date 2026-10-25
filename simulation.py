"""Helper functions for simulating datacubes with moving point sources."""
import numpy as np

# gaussian profile
def gaussianPsf(x, y, sigma):
    return np.exp(-(x**2 + y**2)*(0.5/sigma**2))

# generate a Guassian Kernel, normalized
def gaussianKernel(sigma, n=7):
    xx = np.tile(np.linspace(0, n-1, n), (n, 1))
    yy = xx. transpose()
    gk = gaussianPsf(xx - (n-1)/2, yy - (n-1)/2, sigma)
    return gk /np.sqrt(np.sum(gk**2))

# adding signals of a list of moving objects to the sky background
def addPointSignals(data, pos_list, vel_list, int_list, psf_size, nSub=7):
    xx = np.tile(np.linspace(0, (nSub-1),nSub), (nSub, 1))
    yy = xx.transpose()
    nObj = len(pos_list)
    centerOffset = round(0.5*(nSub-1))
    Nt = data.shape[0]

    for i in range(Nt):
      for j in range(nObj):
        (x0, y0) = pos_list[j]
        (vx, vy) = vel_list[j]
        this_x, this_y = x0 + vx*i, y0 + vy*i
        xI, yI = np.round(this_x).astype(int) - centerOffset, np.round(this_y).astype(int) - centerOffset
        this_psf = gaussianPsf(xx - (this_x - xI), yy - (this_y - yI), psf_size)
        data[i,xI:(xI+nSub)][:, yI:(yI+nSub)] += this_psf*int_list[j]
