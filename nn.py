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

def get_features(path):
    f = io.imread(path)
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum

    r = 20
    sectors = np.zeros([4], dtype=float)
    height = Magq.shape[0]
    width = Magq.shape[1]
    for (y, x), element in np.ndenumerate(Magq):
        v = height/2 - y
        u = (- width / 2) + x
        theta = np.arctan2( v, u )
        if math.sqrt( v ** 2 + u ** 2 ) <= r and v>=0:
            if v==0 and u==0:
                continue
            segs = 12
            i = segs*theta / math.pi % segs
            if(i < 1 or i >=11):
                j = 0
            elif(i < 5 and i >= 1):
                j = 1
            elif(i < 7 and i >= 5):
                j = 2
            else:
                j = 3
            sectors[j] += element ** 2
    return sectors


def plot_boundaries(X, X_target, T, T_target, n_neighbors):
#--- Example code from scikit learn website, modified by me ---

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, X_target)
    h = 10000000000  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    g = h*100
    x_min, x_max = T[:, 0].min() - g, T[:, 0].max() + g
    y_min, y_max = T[:, 1].min() - g, T[:, 1].max() + g
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the test points
    plt.scatter(T[:, 0], T[:, 1], c=T_target, cmap=cmap_bold)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("3-Class classification (k = %i)"
              % (n_neighbors, ))
    # plt.colorbar()

    plt.show()

def get_all_features(paths):
    X = np.empty((0,4), dtype=float)
    for i, p in enumerate(paths):
        result = get_features(p)
        X = np.vstack((X, result))
    return X


if __name__=="__main__":
    paths = glob.glob('characters/*.GIF')
    try:
        X = np.load(Xsave)
    except IOError:
        X = get_all_features(paths)
        np.save(Xsave, X)

    try:
        y = np.load(ysave)
    except IOError:
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
        np.save(ysave, y)

    n=3
    X = X[:, [3,2]]
    plot_boundaries(X, y, X, y, n)
