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
X = X[:, [3,2]]
n = 3

# clf = KNeighborsClassifier(n_neighbors=n)
# clf.fit(X, y)
subdir = "test/"
paths = ["S.GIF", "S_2.GIF", "S_thick1.GIF", "T.GIF", "T_2.GIF", "V.GIF", "V_2.GIF", "V_3.GIF", "V_thick1.GIF"]
for i, p in enumerate(paths):
    paths[i] = subdir + p

T = nn.get_all_features(paths)
T = T[:, [3,2]]
print(T)
nn.plot_boundaries(X, y, T, [0, 0, 0, 1, 1, 2, 2, 2, 2], n)
