from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

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
