import argparse
import sys
import warnings
from glob import glob

import pynbody
import imageio
import matplotlib.pyplot as plt
import numpy as np

SNAPSHOT_ROOT = '/data/REPOSITORY/cosmo/romulus25/'

RENDER_STARS = True
GRIDDED = True
GRID_ROOT = '/home/ramonsharma/'
CELLS_PER_SIDE = 10

OUT_DIR = '/home/ramonsharma/anim/'
FPS = 24


class Image:
    def __init__(self):
        self.parser = self.arguments()
        self.halo_num = self.parser.Halo
        self.timestep = self.parser.Timestep
        self.radius = self.parser.Width / 2
        self.particles = self.parser.Particles
        self.duration = self.parser.duration
        self.direction = self.parser.Direction

        self.snapshot = glob(SNAPSHOT_ROOT + '*' + self.timestep)[0]

        print('Analyzing snapshot:', self.snapshot)
        print('Halo:', self.halo_num)
        print('Within radius:', self.radius, 'kpc')
        print('Only', self.particles, 'particles')

        self.sim = self._load(self.snapshot)
        print('Simulation loaded')
        self.halo_center, self.halo_angmom = self.halo_center_angmom(
            self.sim, self.halo_num, particle_type=self.particles)

        if GRIDDED:
            import reconstruct_grid
            grid_filename = glob(GRID_ROOT + '*' + self.timestep + '*' +
                                 str(CELLS_PER_SIDE) + 'cells.' +
                                 self.particles + '.hdf5')[0]
            self.take = reconstruct_grid.reconstruct_indices(
                grid_filename, self.halo_center, self.radius)
            print('Grid loaded')
        else:
            self.take = None

        self.sim = self._load(self.snapshot, particle_type=self.particles,
                              take=self.take)

        self.image_sim = self.restrict_to_sphere(self.halo_center, self.radius)
        self.center_sim()

        self.align_sim()
        self.num_images = self.rotate_and_image()
        self.generate_movie()

    @classmethod
    def arguments(cls):
        parser = argparse.ArgumentParser(
            description=
            'Generate Pynbody images of a halo within a simulation snapshot')
        parser.add_argument('Halo', type=int, help='Halo number')
        parser.add_argument('Timestep', type=str, help='Simulation timestep')
        parser.add_argument('Particles', type=str,
                            help='Stars, dm, or gas particles')
        parser.add_argument('Width', type=float,
                            help='Width of imaging region (kpc)')
        parser.add_argument('Direction', type=str, help='x, y, or z')
        parser.add_argument('-d', '--duration', type=float, help='Duration of animation',
                            default=5)
        parser.add_argument('-s', '--sideon', action='store_true',
                            help='Set image orientation to side-on',
                            default=False)
        return parser.parse_args()

    def _load(self, snapshot, particle_type=None, take=None):
        sim = pynbody.load(snapshot, take=take)
        sim.physical_units()

        if particle_type == 'stars':
            sim_part = sim.s
        elif particle_type == 'gas':
            sim_part = sim.g
        elif particle_type == 'dm':
            sim_part = sim.dm
        else:
            sim_part = sim

        return sim_part

    def halo_center_angmom(self, sim, halo_num, particle_type='stars'):
        halo = sim.ancestor.halos(dosort=True).load_copy(halo_num)
        halo.physical_units()

        if particle_type == 'stars':
            halo_part = halo.s
        elif particle_type == 'gas':
            halo_part = halo.g
        else:
            halo_part = halo.dm

        print('Halo loaded')
        center = pynbody.analysis.halo.hybrid_center(halo)
        angmom_vec = pynbody.analysis.angmom.ang_mom_vec(halo_part)
        if not sum(angmom_vec) or not sum(angmom_vec) == sum(angmom_vec):
            print(
                'Not enough particles to calculate angular momentum, exiting...'
            )
            sys.exit()

        print('Calculated center and angular momentum')
        return center, angmom_vec

    def restrict_to_sphere(self, center, radius):
        sphere_filter = pynbody.filt.Sphere(radius, center)
        return self.sim[sphere_filter]

    def periodic_wrap(self, relpos, scale, boxsize):
        bphys = boxsize * scale
        bad = np.where(np.abs(relpos) > bphys / 2.)
        if type(bphys) == np.ndarray:
            relpos[bad] = -1.0 * (relpos[bad] / np.abs(
                relpos[bad])) * np.abs(bphys[bad] - np.abs(relpos[bad]))
        else:
            relpos[bad] = -1.0 * (relpos[bad] / np.abs(
                relpos[bad])) * np.abs(bphys - np.abs(relpos[bad]))

    def center_sim(self):
        self.image_sim['pos'] -= self.halo_center
        boxsize = float(self.image_sim.properties['boxsize'])
        scale = self.image_sim.properties['a']
        self.periodic_wrap(self.image_sim['pos'], scale, boxsize)
        print('Halo centered')

    def align_sim(self):
        if self.parser.sideon:
            rotation_matrix = pynbody.analysis.angmom.calc_sideon_matrix(
                self.halo_angmom)
            print('Sim aligned sideon')
        else:
            rotation_matrix = pynbody.analysis.angmom.calc_faceon_matrix(
                self.halo_angmom)
            print('Sim aligned faceon')

        self.image_sim.transform(rotation_matrix)

    def rotate_and_image(self):
        total_rotation = 0
        calibrate = True
        image_num = 0
        degrees_per_image = 360 / (FPS * self.duration)

        while total_rotation <= 360:
            if self.direction == 'x':
                self.image_sim.rotate_x(degrees_per_image)
            elif self.direction == 'y':
                self.image_sim.rotate_y(degrees_per_image)
            elif self.direction == 'z':
                self.image_sim.rotate_z(degrees_per_image)
            total_rotation += degrees_per_image

            self.generate_image(calibrate=calibrate, image_num=image_num)
            print('Generated image', image_num, '/', int(360 / degrees_per_image))
            calibrate = False
            image_num += 1

        print('Done imaging')
        return image_num


    def generate_image(self, units='Msol Mpc**-2', cmap='viridis', vmin=1e10,
                       vmax=1e14, calibrate=False, image_num=0, **kwargs):

        while True:
            if RENDER_STARS:
                pynbody.plot.stars.render(self.image_sim,
                                          width=2 * self.radius)
            else:
                pynbody.plot.image(self.image_sim, width=2 * self.radius,
                                   units=units, cmap=cmap, vmin=vmin,
                                   vmax=vmax, **kwargs)

            if calibrate and not RENDER_STARS:
                plt.ion()
                plt.show()
                recal = input('Re-calibrate (y/n): ')
                if recal != 'y':
                    plt.ioff()
                    calibrate = False
                    continue
                vmin = 10**float(input('New log(vmin): '))
                vmax = 10**float(input('New log(vmax): '))
            else:
                plt.tight_layout()
                plt.savefig(OUT_DIR + str(image_num) + '.png')
                break

    def generate_movie(self):
        filenames = [(OUT_DIR + str(num) + '.png') for num in range(self.num_images)]
        images = [imageio.imread(filename) for filename in filenames]
        outfile = OUT_DIR + 'sim.gif'
        imageio.mimsave(outfile, images, fps=FPS)
        print('Movie output to', outfile)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Image()
