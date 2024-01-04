import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def colorize_2d_matrix(mat, min_val=10.0, max_val=100.0):
    invalid = np.where(mat.flatten() < 1e-9)[0]

    h, w = mat.shape[:2]

    cmap = plt.get_cmap("jet")
    cnorm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    mat_color = cmap(cnorm(mat.flatten()))
    mat_color = (mat_color[:, :3] * 255.0).astype(np.uint8)

    if len(invalid) > 0:
        mat_color[invalid, :] = 0

    mat_color = mat_color.reshape([h, w, 3])
    mat_color = mat_color[:, :, ::-1]
    return mat_color