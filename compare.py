import glob, re
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
# %matplotlib inline

s = re.compile("characters/S\d+.GIF")
t = re.compile("characters/T\d+.GIF")
v = re.compile("characters/V\d+.GIF")

paths = glob.glob('characters/*.GIF')
S = np.empty((400,640,0), dtype=float)
T = np.empty((400,640,0), dtype=float)
V = np.empty((400,640,0), dtype=float)
for i, p in enumerate(paths):
    f = io.imread(p)
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum
    log = np.log( Magq + 1 )
    if s.match(p):
        S = np.dstack((S, log))
    elif t.match(p):
        T = np.dstack((T, log))
    elif v.match(p):
        V = np.dstack((V, log))

S = np.mean(S, axis=2)
T = np.mean(T, axis=2)
V = np.mean(V, axis=2)
print(S.shape)

plt.figure(1)
plt.axis('off')
plt.imshow( S, cmap='gray', aspect=1.6 )

plt.figure(2)
# plt.axis('off')
plt.imshow( T, cmap='gray' )

plt.figure(3)
# plt.axis('off')
plt.imshow( V, cmap='gray' )

plt.show()
