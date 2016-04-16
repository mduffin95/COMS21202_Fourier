from skimage.draw import polygon
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np

def generate_cross(s, t, r): #s = length, t = thickness
    s += 0.5
    t += 0.5
    img = np.zeros((400, 640), dtype=bool)
    mid_x = 320
    mid_y = 200
    X = [mid_x-t, mid_x+t, mid_x+t, mid_x+s, mid_x+s, mid_x+t, mid_x+t, mid_x-t, mid_x-t, mid_x-s, mid_x-s, mid_x-t]
    Y = [mid_y-s, mid_y-s, mid_y-t, mid_y-t, mid_y+t, mid_y+t, mid_y+s, mid_y+s, mid_y+t, mid_y+t, mid_y-t, mid_y-t]
    x = np.array(X)
    y = np.array(Y)
    rr, cc = polygon(y, x)
    img[rr, cc] = True
    img = rotate(img, r)
    return img


def generate_line(h, w, r):
    h += 0.5
    w += 0.5
    img = np.zeros((400, 640), dtype=bool)
    mid_x = 320
    mid_y = 200
    X = [mid_x-w, mid_x+w, mid_x+w, mid_x-w]
    Y = [mid_y-h, mid_y-h, mid_y+h, mid_y+h]
    x = np.array(X)
    y = np.array(Y)
    rr, cc = polygon(y, x)
    img[rr, cc] = True
    img = rotate(img, r)
    return img

if __name__=="__main__":
    a = generate_line(50, 10, 0)
    b = generate_line(30, 5, 45)
    c = np.logical_or(a, b)
    plt.figure()
    plt.imshow(c, cmap="Greys")
    plt.show()
