import numpy as np

from numba import jit
from numba import float64
from numba.types import Array
from numba.experimental import jitclass

import time


def crunching_numbers(workload):
    local = 0
    for i in range(workload):
        local = local + 1
    return local


@jit(nopython=True)
def crunching_numbers_jit(workload):
    local = 0
    for i in range(workload):
        local = local + 1
    return local


@jit(nopython=True)
def heat_diffusion(size, dx, dt, alpha, steps):
    u = np.zeros(size)
    u[0] = 1
    for step in range(steps):
        u_new = u.copy()
        for i in range(1, size - 1):
            u_new[i] = u[i] + alpha * dt / dx ** 2 * (u[i - 1] - 2 * u[i] + u[i + 1])
        u = u_new
    return u


spec = [
    ('u_init', Array(float64, 1, 'C')),
    ('dx', float64),
    ('alpha', float64),
]


@jitclass(spec)
class HeatDiffusionClass:
    def __init__(self, size=int(1.e4), dx=0.1, alpha=1.):
        self.u_init = np.zeros(size)
        self.u_init[0] = 1
        self.alpha = alpha
        self.dx = dx
        self.u_init = np.zeros(size)

    def simulate(self, dt, steps):
        u_local = self.u_init.copy()
        for step in range(steps):
            u_new = u_local.copy()
            for i in range(1, len(u_local) - 1):
                u_new[i] = u_local[i] + self.alpha * dt / self.dx ** 2 * (u_local[i - 1] - 2 * u_local[i] + u_local[i + 1])
            u_local = u_new
        return u_local


def simple_script():
    print('non jit... ', end='')
    start = time.time()
    crunching_numbers(int(1.e8))
    end = time.time()
    print(f'{end - start:.4f}')

    print('jit 1st... ', end='')
    start = time.time()
    crunching_numbers_jit(int(1.e8))
    end = time.time()
    print(f'{end - start:.4f}')

    print('jit 2nd... ', end='')
    start = time.time()
    crunching_numbers_jit(int(1.e8))
    end = time.time()
    print(f'{end - start:.4f}')


def heat_diffusion_script():
    print('running... ', end='')
    start = time.time()
    heat_diffusion(size=int(1.e4), dx=0.1, dt=0.1, alpha=1, steps=int(1.e4))
    end = time.time()
    print(f'{end - start:.4f}')


def heat_diffusion_class_script():
    # hd = HeatDiffusionClass(int(1.e4), float64(.1), float64(1.))
    hd = HeatDiffusionClass()
    print('simulating... ', end='')
    start = time.time()
    # hd.simulate(dt=.1, steps=int(2.e3))
    hd.simulate(.1, int(2.e3))
    end = time.time()
    print(f'{end - start:.4f}')


def main():
    heat_diffusion_class_script()


if __name__ == '__main__':
    main()
