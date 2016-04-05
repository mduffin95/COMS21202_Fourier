import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob, re, math
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

s = re.compile("characters/S\d+.GIF")
t = re.compile("characters/T\d+.GIF")
v = re.compile("characters/V\d+.GIF")


Xsave = "data.npy"
ysave = "target.npy"
# Write code to go through every image and extract its features.
# Store its features, along with its class.

def get_scores(path):
    f = io.imread(path)
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum

    r = 10
    sectors = np.zeros([4], dtype=float)
    height = Magq.shape[0]
    width = Magq.shape[1]
    for (y, x), element in np.ndenumerate(Magq):
        v = height/2 - y
        u = (- width / 2) + x
        theta = np.arctan2( v, u )
        if math.sqrt( v ** 2 + u ** 2 ) <= r:
            if v==0 and u==0:
                continue
            i = math.floor(((theta + 9*math.pi/8) % (2*math.pi)) / (math.pi/4))
            i = i % 4
            sectors[i] += element ** 2
    return sectors

paths = glob.glob('characters/*.GIF')
try:
    X = np.load(Xsave)
    y = np.load(ysave)
except IOError:
    X = np.empty((0,4), dtype=float)
    y = np.empty((len(paths)), dtype=int)
    for i, p in enumerate(paths):
        if s.match(p):
            y[i] = 0
        elif t.match(p):
            y[i] = 1
        elif v.match(p):
            y[i] = 2
        else:
            y[i] = 3
        result = get_scores(p)
        X = np.vstack((X, result))
    np.save(Xsave, X)
    np.save(ysave, y)

print(X)
print(y)

n_neighbors = 3
X = X[:, [2,3]]
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X, y)

#--- Example code from scikit learn website ---
h = 10000000000  # step size in the mesh


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i')"
          % (n_neighbors, ))
plt.colorbar()

plt.show()

# --- End of example code ---
#
# print(scores)
# print(scores.shape)
