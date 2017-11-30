import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
from matplotlib import cm
import math


def surfacePlot(data, label):
    X = np.array([math.log10(t[0]) for t in data])
    Y = np.array([math.log10(t[1]) for t in data])
    Z = np.array([t[2] for t in data])

    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)

    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 150)

    xig, yig = np.meshgrid(xi, yi)


    surf = ax.plot_surface(xig, yig, zi, linewidth=1)

    diff = Z.max() - Z.min()
    ax.set_zlim(Z.min() - 0.5 * diff, Z.max() + 0.5 * diff)

    cset = ax.contour(xig, yig, zi, zdir='z', offset=Z.min() - 0.5 * diff, cmap='winter')

    plt.title(label)
    ax.set_xlabel('log10(L2 regularizer)')
    ax.set_ylabel('log10(kernel STDEV)')
    ax.set_zlabel(label)

    plt.show()


if __name__ == '__main__':
    # data format: (L2 regularization, Kernel initializer STDEV, IOU)
    IOU = [
        [1.00E-01, 1.00E-02, 0.952498116],
        [1.00E-06, 1.00E-02, 0.955517345],
        [1.00E-03, 1.00E-04, 0.946584508],
        [1.00E-03, 1.00E-03, 0.9529881662],
        [1.00E-03, 1.00E-02, 0.9527446009],
        [1.00E-03, 1.00E-01, 0.9474160764],
        [1.00E-08, 1.00E-02, 0.9549213814],
        [1.00E-08, 1.00E-03, 0.9534359807],
        [1.00E-08, 1.00E-04, 0.9496271436]]
    # data format: (L2 regularization, Kernel initializer STDEV, cross entropy loss)
    LOSS = [
        [1.00E-01, 1.00E-02, 0.05743462592],
        [1.00E-06, 1.00E-02, 0.05625329167],
        [1.00E-03, 1.00E-04, 0.05878185481],
        [1.00E-03, 1.00E-03, 0.06963738799],
        [1.00E-03, 1.00E-02, 0.06068232283],
        [1.00E-03, 1.00E-01, 0.0707276687],
        [1.00E-08, 1.00E-02, 0.05834158882],
        [1.00E-08, 1.00E-03, 0.06546074152],
        [1.00E-08, 1.00E-04, 0.06471049041]]

    surfacePlot(IOU, 'IOU CV')
    surfacePlot(LOSS, 'train error')
