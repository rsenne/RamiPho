import numpy as np
import matplotlib.pyplot as plt


__all__ = ["bSpline"]


class bSpline:
    def __init__(self, n, d, c):

        kv = np.array(([0] * d + list(np.arange(0, c - d + 1)) + [c - d] * d))
        u = np.linspace(0, c - d, n)

        self.b = np.zeros((n, c))  # basis
        bb = np.zeros((n, c))  # basis buffer
        left = np.clip(np.floor(u), 0, c - d - 1).astype(int)  # left knot vector indices
        right = left + d + 1  # right knot vector indices

        # Go!
        nrange = np.arange(n)
        self.b[nrange, left] = 1.0
        for j in range(1, d + 1):
            crange = np.arange(j)[:, None]
            bb[nrange, left + crange] = self.b[nrange, left + crange]
            self.b[nrange, left] = 0.0
            for i in range(j):
                f = bb[nrange, left + i] / (kv[right + i] - kv[right + i - j])
                self.b[nrange, left + i] = self.b[nrange, left + i] + f * (kv[right + i] - u)
                self.b[nrange, left + i + 1] = f * (u - kv[right + i - j])

    def create_spline_map(self, left_inds, length):
        spl_map = np.zeros(shape=(np.shape(self.b)[1], length))
        for i in range(np.shape(spl_map)[0]):
            for ind in left_inds:
                spl_map[i, ind:ind+np.shape(self.b)[0]] = self.b[:, i]
        return spl_map


"""
def some_func(x):
    return -2*(x-1.3)**4+4*np.sin(x)+2*x**2+7

    x=np.linspace(0,3.225,31)
y=some_func(x)

glm_model = sm.GLM(y, b, family=sm.families.Gaussian())
glm_fit = glm_model.fit()
predict = glm_fit.predict(b)
coef = glm_fit.params

scaled_splines = b * coef

sb.set()
plt.plot(x, predict, '--', x, y)
plt.show()

inbetween = np.zeros((31,9))

cdd = np.vstack((b,inbetween,b,inbetween))
"""
b_test = bSpline(30, 3, 9)
y0 = b_test.create_spline_map([4, 90, 100], 150)
for y in range(np.shape(y0)[0]):
    plt.plot(y0[y,:])