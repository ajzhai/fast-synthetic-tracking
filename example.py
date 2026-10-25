"""Example: detect six simulated moving objects in a noisy datacube using FST."""
import numpy as np
import matplotlib.pyplot as plt

from fst import FastLinRateSynTracking, lrstSearchAndDetect
from simulation import gaussianKernel, addPointSignals

# Generate a test dataset
Nx, Ny, Nt = 256, 256, 16 # size of the data
np.random.seed(42)
testData = np.random.normal(0, 1, (Nt, Nx, Ny)) # uniform sky background with noise

# list of objects
pos_list = [(50, 75), (180, 20), (65, 180), (130, 155), (25, 205), (205, 180)]
vel_list = [(0.5, 0.8), (-0.25, 0.95), (0.2, -0.75), (-0.85, -0.16), [0.3, -0.4], [-0.7, -0.9]]
I_list = [1.4, 1.8, 1, 1.2, 1.6, 1.5]
psf_size = 1.0

addPointSignals(testData, pos_list, vel_list, I_list, psf_size)

# detection threshold
det_thr = 7

# search over different directions
vel_dir_list = [(1,1), (1,-1), (-1,1), (-1,-1)]

# instantiate the object that contains the frames to facilitate frame integration
lrst  =  FastLinRateSynTracking(testData)

detectedSig = []
d_th = np.sqrt(Nt)*det_thr # noise level increases after integrating Nt frames.

# adding tracking error
g_k = gaussianKernel(np.sqrt(np.log2(Nt)/48 + 1/24*(1-1/Nt) + psf_size**2))

for dir in vel_dir_list:
    lrst.setVelDir(dir)
    integrationInfo = (0, Nt, Nt, 0, 0, 0, 0)
    lrstSearchAndDetect(lrst, integrationInfo, d_th, kernel = g_k, detectedSig = detectedSig)

# Extract detected signals from the list
sig_list = []
for (xy, triple_list) in detectedSig:
    x, y = xy
    # triple_list is a list containing one tuple of three numbers
    a, b, c = triple_list[0]
    sig_list.append([x, y, a, b, c])

sig_array = np.array(sig_list)

# create plots to show the detections
plt.figure(num=1, figsize=(10, 5))
plt.clf()
plt.subplot(1,2,1)
plt.plot(sig_array[:,2] + sig_array[:,0]*0, sig_array [:, 3] + sig_array[:,1]*0, '.')
plt.plot([pos[0] for pos in pos_list], [pos[1] for pos in pos_list], 'r+')
plt.xlabel('X, coloum (pixel)')
plt.ylabel('Y, row (pixel)')
plt.title('Position Distribution of Deteciton')
plt.subplot(1,2,2)
Nv =2** np.ceil(np.log2(Nt))
plt.plot(sig_array[:,0]/Nv, sig_array [:, 1]/Nv, '.')
plt.plot([v[0] for v in vel_list], [v[1] for v in vel_list], 'r+')
plt.xlabel('Vx (pixel/frame)')
plt.ylabel('Vy (pixel/frame)')
plt.title('Motion Rate Distribution of Detection')
plt.show()
