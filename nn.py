import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob, re, math
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap
from features import *

s = re.compile("\w*/S\d+.GIF")
t = re.compile("\w*/T\d+.GIF")
v = re.compile("\w*/V\d+.GIF")


Xsave = "data.npy"
ysave = "target.npy"

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# Write code to go through every image and extract its features.
# Store its features, along with its class.

def get_features(path):
    print(path)
    fnum = 5
    f = io.imread(path)
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.log( np.absolute(q) + 1 )         # magnitude spectrum

    masks = np.empty((fnum, 400, 640), dtype=float)
    masks[0] = cross(50, 5, 0)
    masks[1] = line(5, 100, 0)
    a = line(5, 50, 23)
    b = line(5, 50, -23)
    masks[2] = np.logical_or(a, b).astype(np.uint8)
    masks[3] = triangle()
    masks[4] = ring(60, 10)
    # print(np.sum(mask3), np.sum(mask1))

    res = np.empty(fnum, dtype=float)
    for i in range(fnum):
        # print(masks[i].shape)
        masks[i] = np.multiply( Magq, masks[i] )
        masks[i] = np.power( masks[i], 2 )

        # plt.figure()
        # plt.imshow(masks[i], cmap="Greys") #masks[i, 100:300, 220:420]
        # plt.show()

        res[i] = np.sum( masks[i] )
        # print( res[i] )

    # print(res)
    return res


def plot_boundaries(X, X_target, T, T_target, n_neighbors):
#--- Example code from scikit learn website, modified by me ---

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, X_target)
    h = 0.001  # step size in the mesh
    #
    g = 0.1
    x_min, x_max = T[:, 0].min() - g, T[:, 0].max() + g
    y_min, y_max = T[:, 1].min() - g, T[:, 1].max() + g
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #
    # # Plot also the test points
    plt.scatter(T[:, 0], T[:, 1], c=T_target, cmap=cmap_bold)
    #
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("3-Class classification (k = %i)"
              % (n_neighbors, ))
    plt.colorbar()
    #
    plt.show()

def get_all_features(paths):
    X = np.empty((0,5), dtype=float)
    for i, p in enumerate(paths):
        result = get_features(p)
        X = np.vstack((X, result))
    return X

def plotmatrix(Matrix, target):
  r, c = Matrix.shape
  fig = plt.figure()
  # fig.suptitle("Comparison of Features", fontsize=20 )

  plotID = 1
  for i in range(c):
    for j in range(c):
      ax = fig.add_subplot( c, c, plotID )
      ax.scatter( Matrix[:,i], Matrix[:,j], c=target, cmap=cmap_bold )
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
#       ax.set_title("X=f" + str(i+1) + ", Y=f" + str(j+1))
      plotID += 1
  plt.show()
  # plt.savefig("feature_matrix.png")

def get_targets(paths):
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
    return y

if __name__=="__main__":
    paths = glob.glob('characters/*.GIF')
    try:
        X = np.load(Xsave)
    except IOError:
        X = get_all_features(paths)
        np.save(Xsave, X)

    try:
        X_targets = np.load(ysave)
    except IOError:
        X_targets = get_targets(paths)
        np.save(ysave, X_targets)

    paths = glob.glob('test/*.GIF')
    T=get_all_features(paths)
    T_normed = T / T.max(axis=0)
    T_normed = T_normed[:, [0,4]]
    T_targets = get_targets(paths)

    n=3
    X_normed = X / X.max(axis=0)
    # plotmatrix( X_normed, y )
    X_normed = X_normed[:, [0,4]]

    # print(X_normed)
    plot_boundaries(X_normed, X_targets, T_normed, T_targets, n)
