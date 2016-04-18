from skimage.draw import polygon, circle
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np

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

if __name__=="__main__":
    # a = generate_line(5, 50, 23)
    # b = generate_line(5, 50, -23)
    # c = np.logical_or(a, b)
    a = line(5, 50, 23)
    b = line(5, 50, -23)
    c = np.logical_or(a, b).astype(np.uint8)
    plt.figure()
    plt.imshow(c, cmap="Greys")
    plt.show()
