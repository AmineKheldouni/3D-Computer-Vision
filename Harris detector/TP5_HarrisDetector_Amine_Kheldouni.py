import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
import sys
from operator import itemgetter
from scipy import signal
from scipy.misc import imread, imresize, imrotate
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter, maximum_filter
from scipy.spatial import distance_matrix


# Converts color image to gray
def rgb2gray(im):
    return 0.299 * im[..., 2] + 0.587 * im[..., 1] + 0.114 * im[..., 0]

def plotPoints(image, filtered_coords):
    plt.figure(figsize=(25, 20))
    plt.gray()
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords], '*', color='r', markersize=4)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def plotDetections(I, points):
    p = np.where(points > 0)
    p = [(p[0][i], p[1][i]) for i in range(len(p[0]))]
    plotPoints(I, p)

# Harris edge detection
def harris(im, threshold=0):
    """Personal implementation of Harris corner detector based on the course
    lecture.
    """
    w, h = im.shape
    dx = np.array([-0.5, 0, 0.5])
    kernel = 2
    gf = gaussian_filter1d(np.array([0] * kernel + [1] + [0] * kernel).astype(float), sigma=1)
    Gx = signal.convolve(gf, dx, mode="same").reshape(1, -1)
    Gy = Gx.T
    Ix = signal.convolve2d(im, Gx, mode="same")
    Iy = signal.convolve2d(im, Gy, mode="same")
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    gauss_integ = gaussian_filter1d(np.float_([0] * kernel + [1] + [0] * kernel), 2)

    gauss_integ = np.dot(gauss_integ.reshape(-1, 1), gauss_integ.reshape(1, -1))
    G2 = gaussian_filter(gauss_integ, sigma=2)
    Ix2 = signal.convolve2d(Ix2, G2, mode="same")
    Iy2 = signal.convolve2d(Iy2, G2, mode="same")
    Ixy = signal.convolve2d(Ixy, G2, mode="same")

    M = Ix2 * Iy2 - Ixy * Ixy - 0.04 * (Ix2 + Iy2) * (Ix2 + Iy2)

    M[M < threshold] = 0
    return M


def harrisCV(img):
    """ Open CV Harris corner detector """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    return dst


def nms(M, r):
    h, w = M.shape
    result = M.copy()
    for x in range(r, w - r):
        for y in range(r, h - r):
            if (result[y, x] != 0):
                maximality = np.max(result[y - r:y + r + 1, x - r:x + r + 1])
                if result[y, x] < 0.9 * maximality:
                    result[y, x] = 0
    return result


def anms(score, top=500):
    # Sorting with decreasing values of Harris score
    indices = np.argsort(score, axis=None)[::-1]
    h, w = score.shape
    ly = list(indices // w)
    lx = list(indices % w)
    i = indices[0] // w
    j = indices[0] % w
    s = score[i, j]
    processedPoints = [[i, j, s, np.inf]]
    # Looping over detected points
    for idx in indices[1:]:
        pos = np.array([idx // w, idx % w])
        s = score[pos[0], pos[1]]
        # If the score is too low, we can stop our ANMS since we have enough relevant points
        if score[pos[0], pos[1]] <= 0:
            break
        # Checking the score condition (being almost maximal with threshold ratio of 90%)
        tmp = np.where(np.array(processedPoints)[:, 2] * 0.9 > score[pos[0], pos[1]])
        # Updating radius which minimizes the distance to a processed point
        if len(tmp[0]) == 0:
            radius = np.inf
        else:
            radius = np.min(distance_matrix(np.array(processedPoints)[:, :2][tmp[0], :], pos.reshape(1, -1)))

        processedPoints.append([pos[0], pos[1], score[pos[0], pos[1]], radius])
    result = sorted(processedPoints, key=itemgetter(3), reverse=True)[:top]
    return result


# Chess Harris
path1 = 'images/Chessboard.png'
I1 = io.imread(path1)
7
threshold = 0
plt.figure(figsize=(18, 16))
dst = harrisCV(I1)
plotDetections(I1, dst)

points_nms = nms(dst, 8)
plotDetections(I1, points_nms)
points_anms = anms(dst, top=800)
points_anms = np.delete(points_anms, [2, 3], 1)
filtered_coords = points_anms
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I1, cmap='gray')
plt.plot([p[1] for p in filtered_coords],
         [p[0] for p in filtered_coords], '*', color='r', markersize=4)
plt.axis('off')
plt.show()

# Chess rotation:
I2 = imrotate(np.pad(io.imread(path1), ((0, 0), (200, 200), (0, 0)), 'constant'), 30, 'bilinear')
dst2 = harrisCV(I2)
plotDetections(I2, dst2)

nms2 = nms(dst2, 8)

plotDetections(I2, nms2)
anms2 = anms(dst2, top=800)
anms2 = np.delete(anms2, [2, 3], 1)
filtered_coords2 = anms2
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I2, cmap='gray')
plt.plot([p[1] for p in filtered_coords2],
         [p[0] for p in filtered_coords2], '*', color='r', markersize=4)
plt.axis('off')
plt.show()

# ENPC logo:
path2 = 'images/enpc.jpg'
I3 = io.imread(path2)

threshold = 600
plt.figure(figsize=(18, 16))
dst3 = harris(rgb2gray(I3), threshold=threshold)
plotDetections(I3, dst3)

points_nms3 = nms(dst3, 5)
plotDetections(I3, points_nms3 > threshold)

anms3 = anms(dst3, top=100)
anms3 = np.delete(anms3, [2, 3], 1)
filtered_coords3 = anms3
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I3, cmap='gray')
plt.plot([p[1] for p in filtered_coords3],
         [p[0] for p in filtered_coords3], '*', color='r', markersize=4)
plt.axis('off')
plt.show()

# ENPC logo:
path2 = 'images/enpc.jpg'
I3 = io.imread(path2)

threshold = 0
plt.figure(figsize=(18, 16))
dst3 = harris(rgb2gray(I3), threshold=threshold)
plotDetections(I3, dst3)

points_nms3 = nms(dst3, 5)
plotDetections(I3, points_nms3)

# Cow:
path3 = 'images/cow.png'
I4 = io.imread(path3)

threshold = 1000
plt.figure(figsize=(18, 16))
dst4 = harris(rgb2gray(I4), threshold=threshold)
plotDetections(I4, dst4)

points_nms4 = nms(dst4, 3)
plotDetections(I4, points_nms4)

anms4 = anms(dst4, top=800)
anms4 = np.delete(anms4, [2, 3], 1)
filtered_coords4 = anms4
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I4, cmap='gray')
plt.plot([p[1] for p in filtered_coords4],
         [p[0] for p in filtered_coords4], '*', color='r', markersize=4)
plt.axis('off')
plt.show()

# Eiffel Tower by day:
path4 = 'images/EiffelTowerDay.jpg'
I5 = io.imread(path4)

threshold = 20000
plt.figure(figsize=(18, 16))
dst5 = harris(rgb2gray(I5), threshold=threshold)
plotDetections(I5, dst5)

points_nms5 = nms(dst5, 3)
plotDetections(I5, points_nms5)

anms5 = anms(dst5, top=400)
anms5 = np.delete(anms5, [2, 3], 1)
filtered_coords5 = anms5
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I5, cmap='gray')
plt.plot([p[1] for p in filtered_coords5],
         [p[0] for p in filtered_coords5], '*', color='r', markersize=4)
plt.axis('off')
plt.show()

# Eiffel Tower by night:
path5 = 'images/EiffelTowerNight.jpg'
I6 = io.imread(path5)

threshold = 10000
plt.figure(figsize=(18, 16))
dst6 = harris(rgb2gray(I6), threshold=threshold)
plotDetections(I6, dst6)

points_nms6 = nms(dst6, 5)
plotDetections(I6, points_nms6)

anms6 = anms(dst6, top=400)
anms6 = np.delete(anms6, [2, 3], 1)
filtered_coords6 = anms6
plt.figure(figsize=(25, 20))
plt.gray()
plt.imshow(I6, cmap='gray')
plt.plot([p[1] for p in filtered_coords6],
         [p[0] for p in filtered_coords6], '*', color='r', markersize=4)
plt.axis('off')
plt.show()
