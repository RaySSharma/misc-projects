from itertools import product
import numpy as np
import matplotlib.pyplot as plt

def create_retro_map(image, num_columns, num_rows, spacing):
    norm_image = normalize(image)
    low_res_image, bins_x, bins_y = reduce_res(norm_image, num_columns, num_rows)
    retro_map(low_res_image, bins_x, bins_y, spacing)

def normalize(image):
    row_sums = image.sum(axis=1)
    row_sums[row_sums == 0] = 1
    norm_image = image / row_sums[:, np.newaxis]
    return norm_image

def reduce_res(image, num_columns, num_rows):
    low_res_image = np.zeros((num_columns, num_rows))
    len_x = image.shape[0]
    len_y = image.shape[1]

    assert num_columns < len_x, "Oversampling"
    assert num_rows < len_y, "Oversampling"

    bins_x = np.linspace(0, len_x, num_columns)
    bins_y = np.linspace(0, len_y, num_rows)

    xspace = np.linspace(0, len_x-1, len_x)
    yspace = np.linspace(0, len_y-1, len_y)

    digitize_x = np.digitize(xspace, bins_x)
    digitize_y = np.digitize(yspace, bins_y)

    for i, j in product(range(1, num_columns), range(1, num_rows)):
        ix_x = digitize_x == i
        ix_y = digitize_y == j
        im = image[ix_x,:][:,ix_y]
        low_res_image[i-1,j-1] = im.sum()

    return low_res_image, bins_x, bins_y

def retro_map(image, bins_x, bins_y, spacing):
    f, ax = plt.subplots(1,1, figsize=(6,6))

    image_range = abs(image.max() - image.min())
    bins = np.linspace(0, spacing, len(bins_y))
    offset_arr = bins * image_range
    print(bins)
    for row, offset in zip(image, offset_arr):
        ax.plot(bins_x, row+offset, color='C0')

    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_yticks([])

    f.tight_layout()
    f.show()

