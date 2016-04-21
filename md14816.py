import numpy as np
from skimage import io
from skimage.draw import polygon, circle
from skimage.transform import rotate
from scipy import stats
import matplotlib.pyplot as plt
import glob, re, math
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap

s = re.compile("\w*/S\d+.GIF")
t = re.compile("\w*/T\d+.GIF")
v = re.compile("\w*/V\d+.GIF")


Xsave = "data.npy"
ysave = "target.npy"

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# Write code to go through every image and extract its features.
# Store its features, along with its class.

def cross(s, t, r): #s = length, t = thickness
    s += 0.5
    t += 0.5
    img = np.zeros((400, 640), dtype=float)
    mid_x = 320
    mid_y = 200
    X = [mid_x-t, mid_x+t, mid_x+t, mid_x+s, mid_x+s, mid_x+t, mid_x+t, mid_x-t, mid_x-t, mid_x-s, mid_x-s, mid_x-t]
    Y = [mid_y-s, mid_y-s, mid_y-t, mid_y-t, mid_y+t, mid_y+t, mid_y+s, mid_y+s, mid_y+t, mid_y+t, mid_y-t, mid_y-t]
    x = np.array(X)
    y = np.array(Y)
    rr, cc = polygon(y, x)
    img[rr, cc] = 1
    rr, cc = circle(mid_y, mid_x, 10)
    img[rr, cc] = 0
    img = rotate(img, r, preserve_range=True)
    return img


def line(h, w, r):
    h += 0.5
    w += 0.5
    img = np.zeros((400, 640), dtype=float)
    mid_x = 320
    mid_y = 200
    X = [mid_x-w, mid_x+w, mid_x+w, mid_x-w]
    Y = [mid_y-h, mid_y-h, mid_y+h, mid_y+h]
    x = np.array(X)
    y = np.array(Y)
    rr, cc = polygon(y, x)
    img[rr, cc] = 1
    rr, cc = circle(mid_y, mid_x, 10)
    img[rr, cc] = 0
    img = rotate(img, r, preserve_range=True)
    return img

def triangle():
    img = np.zeros((400, 640), dtype=float)
    x = np.array([369.5, 400.5, 369.5])
    y = np.array([49.5, 49.5, 120.5])
    rr, cc = polygon(y, x)
    img[rr, cc] = 1
    return img

def ring(r, t):
    img = np.zeros((400, 640), dtype=float)
    mid_x = 320
    mid_y = 200
    rr, cc = circle(mid_y, mid_x, r+t)
    img[rr, cc] = 1
    rr, cc = circle(mid_y, mid_x, r)
    img[rr, cc] = 0
    return img

def max_likelihood( data, targets, no_clusters ):
    means = []
    covs = []
    for i in range(no_clusters):
        z = np.array([x for x,y in zip(data, targets) if y==i])
        means.append(np.mean(z, axis=0))
        covs.append(np.cov(z.T))

    print(means)
    print(covs)

    delta = 0.001
    x = np.arange(0, 2, delta)
    X, Y = np.meshgrid(x,x)
    R = np.dstack((X, Y))

    mvn1 = stats.multivariate_normal.pdf(R, means[0], covs[0])
    mvn2 = stats.multivariate_normal.pdf(R, means[1], covs[1])
    mvn3 = stats.multivariate_normal.pdf(R, means[2], covs[2])

    diff1 = mvn1 - mvn2
    diff2 = mvn1 - mvn3
    diff3 = mvn2 - mvn3

    plt.contour(X, Y, diff1, [0], colors='teal')
    plt.contour(X, Y, diff2, [0], colors='brown')
    plt.contour(X, Y, diff3, [0], colors='purple')

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
        # masks[i] = np.power( masks[i], 2 )

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
    # plt.title("3-Class classification (k = %i)"
    #           % (n_neighbors, ))
    plt.xlabel("Feature 0 (Cross)", fontsize=18)
    plt.ylabel("Feature 4 (Ring)", fontsize=18)

    # plt.colorbar()
    #
    # plt.show()
    # plt.savefig("test_plot.png")

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


    n=4
    X_normed = X / X.max(axis=0)
    # plotmatrix( X_normed, X_targets )
    X_normed = X_normed[:, [0,4]]

    # paths = glob.glob('test/*.GIF')
    paths = ["characters/ab/A1.GIF", "characters/ab/B1.GIF"]
    T=get_all_features(paths)
    T_normed = T / X.max(axis=0)
    T_normed = T_normed[:, [0,4]]
    T_targets = get_targets(paths)



    # print(X_normed)
    plot_boundaries(X_normed, X_targets, T_normed, T_targets, n)
    max_likelihood(X_normed, X_targets, 3)
    plt.show()
