import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob, re, math

s = re.compile("characters/S\d+.GIF")
t = re.compile("characters/T\d+.GIF")
v = re.compile("characters/V\d+.GIF")

# Write code to go through every image and extract its features.
# Store its features, along with its class.

paths = glob.glob('characters/*.GIF')
labels = []
for p in paths:
    if s.match(p):
        labels.append(0)
    elif t.match(p):
        labels.append(1)
    elif v.match(p):
        labels.append(2)
    else:
        labels.append(3)

# print(paths[0])
# f = io.imread(paths[0])
f = io.imread("characters/T3.GIF")
f_f = np.array(f, dtype=float)
z = np.fft.fft2(f_f)           # do fourier transform
print(z[0,0])
q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
print(q[200,320])
Magq =  np.absolute(q)         # magnitude spectrum
Phaseq = np.angle(q)           # phase spectrum
test_set = Magq

r = 10
sectors = np.zeros([4], dtype=float)
height = test_set.shape[0]
width = test_set.shape[1]
for (y, x), element in np.ndenumerate(test_set):
    v = height/2 - y
    u = (- width / 2) + x
    theta = np.arctan2( v, u )
    if math.sqrt( v ** 2 + u ** 2 ) <= r:
        if v==0 and u==0:
            continue
        i = math.floor(((theta + 9*math.pi/8) % (2*math.pi)) / (math.pi/4))
        i = i % 4
        sectors[i] += element ** 2
        # print(x, y, element)
print(sectors)
