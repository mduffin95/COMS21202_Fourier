import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob, re, math
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import nn

Xsave = "data.npy"
ysave = "target.npy"

X = np.load(Xsave)
y = np.load(ysave)
X = X[:, [0,4]]
n = 3

# clf = KNeighborsClassifier(n_neighbors=n)
# clf.fit(X, y)
subdir = "test/"
paths = ["S1.GIF", "S2.GIF", "S3.GIF", "T1.GIF", "T2.GIF", "T3.GIF", "V1.GIF", "V2.GIF", "V3.GIF"]
for i, p in enumerate(paths):
    paths[i] = subdir + p

T = nn.get_all_features(paths)
T = T[:, [0,4]]
print(T)
nn.plot_boundaries(X, y, T, [0, 0, 0, 1, 1, 1, 2, 2, 2], n)
