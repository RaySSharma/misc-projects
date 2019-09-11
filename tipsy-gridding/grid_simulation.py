import argparse
import warnings
import itertools
from glob import glob

import pynbody
import h5py
import numpy as np

SNAPSHOT_ROOT = '/data/REPOSITORY/cosmo/romulus25/'
OUT_DIR = '/home/ramonsharma/'
CELLS_PER_SIDE = 10


class Grid:
    def __init__(self):
        self.parser = self.arguments()
        self.timestep = self.parser.Timestep
        self.particles = self.parser.Particles

        self.snapshot = glob(SNAPSHOT_ROOT + '*' + self.timestep)[0]

        print('Gridding snapshot:', self.snapshot)
        print('Tracking only', self.particles, 'particles')

        self.sim = self._load(self.snapshot, self.particles)
        self.redshift = self._redshift()
        self.boxsize = self._boxsize()

        self.coords = self._coords()
        self.index = self._indices()

        self.cell_num_grid, self.grid = self.enumerate_grid()
        self.coords_grid, self.grid_edges = self.coords_to_grid()
        self.coords_cells = self.calc_cell_numbers(self.coords_grid)

        self.hdf_file = self.create_hdf()
        self.data, self.header = self.hdf_file['data'], self.hdf_file['header']

        self.save_header()
        self.save_data()

    @staticmethod
    def arguments():
        parser = argparse.ArgumentParser(
            description=
            'Place particle indices of Tipsy simulation on a grid for partial loading.'
        )
        parser.add_argument('Timestep', type=str, help='Simulation timestep')
        parser.add_argument('Particles', type=str,
                            help='Stars, dm, or gas particles')
        return parser.parse_args()

    @staticmethod
    def _load(snapshot, particle_type='stars', take=None):
        sim = pynbody.load(snapshot, take=take)
        print('Simulation snapshot loaded.')

        sim.physical_units()
        if particle_type == 'stars':
            print('Star particle type selected...')
            sim_part = sim.s
        elif particle_type == 'gas':
            print('Gas particle type selected...')
            sim_part = sim.g
        else:
            print('Dark matter particle type selected.')
            sim_part = sim.dm

        return sim_part

    def _boxsize(self):
        boxsize = self.sim.properties['boxsize']
        return float(boxsize)

    def _redshift(self):
        scale = self.sim.properties['a']
        return (1 / scale) - 1

    def _coords(self):
        with self.sim.ancestor.immediate_mode:
            pos = self.sim['pos']
        scale = 1 / (1 + self.redshift)
        self.periodic_wrap(pos, scale, self.boxsize)
        print('Loaded particle coordinates')
        return pos

    def _indices(self):
        with self.sim.ancestor.immediate_mode:
            index = self.sim.get_index_list(self.sim.ancestor)
        print('Loaded', len(index), 'particle indices')
        return index

    def periodic_wrap(self, relpos, scale, boxsize):
        bphys = boxsize * scale
        bad = np.where(np.abs(relpos) > bphys / 2.)
        if type(bphys) == np.ndarray:
            relpos[bad] = -1.0 * (relpos[bad] / np.abs(relpos[bad])) * np.abs(bphys[bad] - np.abs(relpos[bad]))
        else:
            relpos[bad] = -1.0 * (relpos[bad] / np.abs(relpos[bad])) * np.abs(bphys - np.abs(relpos[bad]))

    def enumerate_grid(self):
        cell_num_side = range(CELLS_PER_SIDE)
        grid_num_iter = itertools.product(cell_num_side, cell_num_side, cell_num_side)
        grid_num = np.array([num for num in grid_num_iter])
        print('Enumerated grid')
        return self.calc_cell_numbers(grid_num), grid_num

    def calc_cell_numbers(self, cells):
        i, j, k = cells.T
        cell_num = i + CELLS_PER_SIDE*j + CELLS_PER_SIDE**2 * k
        return cell_num

    def coords_to_grid(self):
        bins = np.linspace(-self.boxsize / 2, self.boxsize / 2,
                           CELLS_PER_SIDE + 1)
        x, y, z = self.coords.T
        grid_x = np.digitize(x, bins) - 1
        grid_y = np.digitize(y, bins) - 1
        grid_z = np.digitize(z, bins) - 1
        print('Split particles into grids')
        return np.array([grid_x, grid_y, grid_z]).T, bins

    def generate_outfile_name(self):
        snapshot_name = self.snapshot.split('/')[-1]
        outfile = OUT_DIR + snapshot_name + '.' + str(
            CELLS_PER_SIDE) + 'cells.' + str(self.particles) + '.hdf5'
        return outfile

    def create_hdf(self):
        outfile = self.generate_outfile_name()
        f = h5py.File(outfile, 'w')
        f.create_group('data')
        f.create_group('header')
        print('Outputting grid indices to', outfile)
        return f

    def save_header(self):
        self.header.create_dataset(name='grid_edges', data=self.grid_edges)
        self.header.create_dataset(name='grid', data=self.grid)
        self.header.create_dataset(name='redshift', data=self.redshift)
        self.header.create_dataset(name='boxsize', data=self.boxsize)
        print('Saved header')

    def add_dataset(self, cell_number):
        ix = np.where(self.coords_cells == cell_number)[0]
        indices_cell = self.index[ix]
        name_cell = str(cell_number)
        self.data.create_dataset(name=name_cell, data=indices_cell,
                                 dtype='uint32')

    def save_data(self):
        for cell_number, cell in enumerate(self.cell_num_grid):
            print(cell_number, '/', CELLS_PER_SIDE**3)
            self.add_dataset(cell)

        print('Gridding done.')
        self.hdf_file.close()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Grid()
