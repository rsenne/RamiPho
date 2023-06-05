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
            try:
                for ind in left_inds:
                    spl_map[i, ind:ind + np.shape(self.b)[0]] = self.b[:, i]
            except ValueError:
                for ind in left_inds[:-3]:
                    spl_map[i, ind:ind + np.shape(self.b)[0]] = self.b[:, i]


        spl_dict = {f"spline_var{i}": spl_map[i, :] for i in range(np.shape(spl_map)[0])}
        return spl_map, spl_dict