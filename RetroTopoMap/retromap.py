from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def create_retro_map(image, num_columns, num_rows, spacing=1, norm=True,
                     stat=np.sum, plot_3d=False):
    if norm:
        image = normalize(image)

    low_res_image, bins_x, bins_y = reduce_res(image, num_columns, num_rows,
                                               stat)
    if plot_3d:
        pass
    else:
        retro_map(low_res_image, bins_x, bins_y, spacing)


def normalize(image):
    norm = abs(image.sum())
    norm_image = image / norm
    return norm_image


def reduce_res(image, num_columns, num_rows, stat):
    low_res_image = np.zeros((num_columns, num_rows))
    len_x = image.shape[0]
    len_y = image.shape[1]

    assert num_columns < len_x, "Oversampling"
    assert num_rows < len_y, "Oversampling"

    bins_x = np.linspace(0, len_x - 1, num_columns)
    bins_y = np.linspace(0, len_y - 1, num_rows)

    xspace = np.linspace(0, len_x - 1, len_x)
    yspace = np.linspace(0, len_y - 1, len_y)

    digitize_x = np.digitize(xspace, bins_x, right=True)
    digitize_y = np.digitize(yspace, bins_y, right=True)

    for i, j in product(range(1, num_columns+1), range(1, num_rows+1)):
        ix_x = digitize_x == i
        ix_y = digitize_y == j
        im = image[ix_x, :][:, ix_y]
        low_res_image[i - 1, j - 1] = stat(im)

    return low_res_image, bins_x, bins_y


def retro_map(image, bins_x, bins_y, spacing):
    f, ax = plt.subplots(1, 1, figsize=(6, 6))

    image_range = abs(image.max() - image.min())

    assert spacing > 0, "Spacing must be greater than zero"
    offset_arr = np.linspace(0,
                             len(bins_y) - 1,
                             len(bins_y)) * image_range * spacing
    for row, offset in zip(image, offset_arr):
        ax.plot(bins_x, row + offset, color='C0')

    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_yticks([])

    f.tight_layout()
    f.show()
