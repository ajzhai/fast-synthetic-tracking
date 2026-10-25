"""Core implementation of Fast Synthetic Tracking (FST)."""
import numpy as np
import scipy.signal


class FastLinRateSynTracking:
    """ This class is defined to facilitate integration of frames using linear rate synthetic tracking"""

    # initialization, taking a datacube (frames to integrate) as input
    def __init__(self, dataCube):
        self.Nt = dataCube.shape[0]
        self.Nx = dataCube.shape[1]
        self.Ny = dataCube.shape[2]
        # Nt_ext computes the number of additional intermediate frames needed to ensure sufficient memory allocation when Nt is not a power of 2
        self.Nt_ext = max(2**(np.ceil(np.log2(self.Nt))).astype(int), np.ceil(np.log2(self.Nt)).astype(int) + self.Nt -1)

        self.I_int = np.full((self.Nt_ext + self.Nt, self.Nx, self.Ny), np.nan)
        self.I_int[0:self.Nt] = dataCube
        self.vxDir = 1      # specifying direction of x velocity
        self.vyDir = 1      # specifying direction of x velocity
        self.initFrame = 0  # index of the first frames to be integrated
        self.xOffset = 0    # pixels to shift along x for this level
        self.yOffset = 0    # pixels to shift along y for this level
        self.frameOffset = self.Nt   # an offset for storing frame pair integration results
        self.nFrame = self.Nt        # number of frames to integrate
        # These are used for the case Nt is not a power of 2 to bookkeep the frames left from integrating an odd number of frames
        self.lastFrame = 0  # lastFrame is a flag determining whether there is an unpaired frame left from adding frame pairs.
        self.lastFrameDur = 0  # this works together with the lastFrame, indicating which level this frame was left.

    # set the direction of motion
    def setVelDir(self, xyDir):
        if xyDir[0] < 0 :
          self.vxDir = -1
        else:
          self.vxDir = 1

        if xyDir[1] < 0 :
          self.vyDir = -1
        else:
          self.vyDir = 1

    # computing the index for frame shifts and index offsets to facilitate frame shifts
    def frameShiftIndex(self, dx, dy):
        dx, dy = round(dx), round(dy)
        idx_x_start1, idx_x_end1 = max(-dx, 0), min(self.Nx, self.Nx - dx)
        idx_y_start1, idx_y_end1 = max(-dy, 0), min(self.Ny, self.Ny - dy)
        idx_x_start2, idx_x_end2 = max(dx, 0), min(self.Nx, self.Nx + dx)
        idx_y_start2, idx_y_end2 = max(dy, 0), min(self.Ny, self.Ny + dy)
        return (idx_x_start1, idx_x_end1, idx_y_start1, idx_y_end1, idx_x_start2, idx_x_end2, idx_y_start2, idx_y_end2)

    # core routine to integrate consecutive frame pairs
    def frameIntegration(self, vx_bit, vy_bit):
        if self.allFramesStacked():
            return;
        dx = (self.xOffset + vx_bit)*self.vxDir
        dy = (self.yOffset + vy_bit)*self.vyDir
        nPairs = (self.nFrame - self.lastFrame)//2

        # new unpaired frame, record the step when the single frame occurred
        if self.lastFrame == 0 and self.nFrame % 2:
            self.lastFrameDur = 1

        self.I_int[self.frameOffset:(self.frameOffset + nPairs),:,:] = self.I_int[self.initFrame:(self.initFrame+nPairs*2):2,:,:]
        idx_x_start1, idx_x_end1, idx_y_start1, idx_y_end1, idx_x_start2, idx_x_end2, idx_y_start2, idx_y_end2 = self.frameShiftIndex(dx, dy)
        self.I_int[self.frameOffset:(self.frameOffset + nPairs),idx_x_start1:idx_x_end1,idx_y_start1:idx_y_end1] += self.I_int[(self.initFrame+1):(self.initFrame+2*nPairs):2,idx_x_start2:idx_x_end2,idx_y_start2:idx_y_end2]

        self.lastFrame = self.nFrame - 2*nPairs

        # Make sure to integrate the last frame when paired
        if self.lastFrame > 0:
            self.I_int[self.frameOffset + nPairs,:,:] = self.I_int[(self.initFrame+nPairs*2),:,:]
            if self.lastFrame > 1:
                dx = (self.xOffset + vx_bit)*self.vxDir*(1+self.lastFrameDur)/2
                dy = (self.yOffset + vy_bit)*self.vyDir*(1+self.lastFrameDur)/2

                idx_x_start1, idx_x_end1, idx_y_start1, idx_y_end1, idx_x_start2, idx_x_end2, idx_y_start2, idx_y_end2 = self.frameShiftIndex(dx, dy)
                self.I_int[self.frameOffset + nPairs,idx_x_start1:idx_x_end1,idx_y_start1:idx_y_end1] += self.I_int[(self.initFrame+self.nFrame-1),idx_x_start2:idx_x_end2,idx_y_start2:idx_y_end2]

                self.lastFrame -= 2
                nPairs += 1

        self.initFrame = self.frameOffset
        self.nFrame = nPairs + self.lastFrame
        self.frameOffset = self.frameOffset + self.nFrame
        self.xOffset = self.xOffset*2 + vx_bit
        self.yOffset = self.yOffset*2 + vy_bit
        self.lastFrameDur *= 0.5

    def sigDetection(self, d_th, kernel = np.ones((1,1)), reset=True):
        if self.nFrame > 1:
            return []
        else:
            detectionLocList = np.nonzero(scipy.signal.convolve2d(self.I_int[self.frameOffset-1,:,:], kernel, mode='same') > d_th)
            detectedSigList = [(rc[0], rc[1], self.I_int[self.frameOffset-1, rc[0], rc[1]]) for rc in zip(detectionLocList[0], detectionLocList[1])]
            if reset:
                self.resetTracking()
            if len(detectedSigList) > 0 :
                return ((self.xOffset * self.vxDir, self.yOffset * self.vyDir), detectedSigList)
            else:
                return []

    def resetTracking(self):
        self.initFrame = 0
        self.xOffset = 0
        self.yOffset = 0
        self.frameOffset = self.Nt
        self.nFrame = self.Nt
        self.lastFrame = 0
        self.lastFrameDur = 0

    def allFramesStacked(self):
        return self.nFrame == 1

    def setIntInfo(self, integrationInfo):
        self.initFrame = integrationInfo[0]
        self.nFrame = integrationInfo[1]
        self.frameOffset = integrationInfo[2]
        self.lastFrame = integrationInfo[3]
        self.xOffset = integrationInfo[4]
        self.yOffset = integrationInfo[5]
        self.lastFrameDur = integrationInfo[6]

    def getIntInfo(self):
        return (self.initFrame, self.nFrame, self.frameOffset, self.lastFrame, self.xOffset, self.yOffset, self.lastFrameDur)

    def getIntDataFrames(self, fIdx):
        return self.I_int[fIdx,:,:]

# This routine works with FastLinRateSynTracking class, which is the first argument,
# the second argument is the information at certain level of integration, tuple of (initFrame, nFrame, frameOffset, lastFrame, xOffset, yOffset, lastFrameDur)
# Detections are appended to detectedSig, which is also returned.
def lrstSearchAndDetect(lrst, integrationInfo, d_th, kernel = np.ones((1,1)), detectedSig = None):
  if detectedSig is None:
    detectedSig = []
  for x_bit in [0, 1]:
    for y_bit in [0, 1]:
      lrst.setIntInfo(integrationInfo)
      lrst.frameIntegration(x_bit, y_bit)

      if lrst.allFramesStacked():
        thisDetection = lrst.sigDetection(d_th, kernel = kernel, reset=False)
        if len(thisDetection) > 0:
          detectedSig.append(thisDetection)
      else:
        integrationInfoNew = lrst.getIntInfo()
        lrstSearchAndDetect(lrst, integrationInfoNew, d_th, kernel=kernel, detectedSig = detectedSig)
  return detectedSig

# This routine allows a user to compute an integrated image for a specific motion rate.
def performFST(lrst, rate):
    def int2bin(n, m=8):
        bits = np.zeros((m,))
        bin_str = bin(n)[2:][::-1]
        for ii in range(min(m,len(bin_str))):
            bits[ii] = int(bin_str[ii])
        return bits
    rate_abs = np.array(rate)
    lrst.resetTracking()
    integrationInfo = lrst.getIntInfo()
    dir = [1,1]
    for i in range(2):
        if rate_abs[i] < 0:
            rate_abs[i] = -rate_abs[i]
            dir[i] = -1
    lrst.setVelDir(dir)

    m = np.ceil(np.log2(integrationInfo[1])).astype(int)
    n2 = 2**m
    xy_bits = np.zeros((m,2)).astype(int)
    for j in range(2):
        xy_bits[:,j] = int2bin(np.round(rate_abs[j]*(n2-1)).astype(int), m)

    for i in range(m):
        lrst.frameIntegration(xy_bits[m-1-i,0], xy_bits[m-1-i,1])

    integrationInfo = lrst.getIntInfo()
    if integrationInfo[1] == 1:
        return lrst.getIntDataFrames(integrationInfo[2]-1)
    else:
        print('Warning: Inconsistency! Frames are not fully integrated!')
        return []
