import h5py
import numpy as np


def reconstruct_indices(grid_filename, center, radius):
    grid_data, grid_header = load_grid_file(grid_filename)
    grid, grid_edges = grid_header['grid'][:], grid_header['grid_edges'][:]
    redshift = grid_header['redshift'].value
    boxsize = grid_header['boxsize'].value
    cells_per_side = len(grid_header['grid_edges'][:]) - 1

    grid_of_interest = map_region_to_grid(center, radius, grid_edges, redshift, boxsize)
    cells_of_interest = enumerate_cells(grid_of_interest, cells_per_side)
    indices = map_cells_to_indices(grid_data, cells_of_interest)
    return np.int64(indices)


def periodic_wrap(relpos, scale, boxsize):
    bphys = boxsize * scale
    bad = np.where(np.abs(relpos) > bphys/2.)
    if type(bphys) == np.ndarray:
        relpos[bad] = -1.0 * (relpos[bad] / np.abs(relpos[bad])) * np.abs(bphys[bad] - np.abs(relpos[bad]))
    else:
        relpos[bad] = -1.0 * (relpos[bad] / np.abs(relpos[bad])) * np.abs(bphys - np.abs(relpos[bad]))


def load_grid_file(grid_filename):
    grid_file = h5py.File(grid_filename, 'r')
    grid_data = grid_file['data']
    grid_header = grid_file['header']

    return grid_data, grid_header


def map_region_to_grid(center, radius, grid_edges, redshift, boxsize):
    edges_low = center - radius
    edges_hi = center + radius
    scale = 1 / (1 + redshift)

    xspan = np.linspace(edges_low[0], edges_hi[0], 1001)
    yspan = np.linspace(edges_low[1], edges_hi[1], 1001)
    zspan = np.linspace(edges_low[2], edges_hi[2], 1001)
    periodic_wrap(xspan, scale, boxsize)
    periodic_wrap(yspan, scale, boxsize)
    periodic_wrap(zspan, scale, boxsize)

    xcells = np.digitize(xspan, bins=grid_edges) - 1
    ycells = np.digitize(yspan, bins=grid_edges) - 1
    zcells = np.digitize(zspan, bins=grid_edges) - 1

    all_cells = np.array([xcells, ycells, zcells]).T
    grid = np.unique(all_cells, axis=0)
    return grid


def enumerate_cells(cells, cells_per_side):
    i, j, k = cells.T
    cell_num = i + cells_per_side * j + cells_per_side**2 * k
    return cell_num


def map_cells_to_indices(grid_data, cell_numbers):
    cell_numbers_str = cell_numbers.astype(str)
    indices = [grid_data[col][:] for col in cell_numbers_str]
    return np.hstack(indices)
