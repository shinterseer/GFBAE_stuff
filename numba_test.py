import numpy as np

from numba import jit
from numba import float64, int32
from numba.types import Array
from numba.experimental import jitclass

import time


@jit(nopython=True)
def crunching_numbers(workload):
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

    def simulate(self, dt, steps):
        u_local = self.u_init.copy()
        local_alpha = self.alpha  # for some reason this makes it like 20-30% faster
        local_dx = self.dx  # for some reason this makes it like 20-30% faster
        for step in range(steps):
            u_new = u_local.copy()
            for i in range(1, len(u_local) - 1):
                u_new[i] = u_local[i] + local_alpha * dt / local_dx ** 2 * (u_local[i - 1] - 2 * u_local[i] + u_local[i + 1])
            u_local = u_new
        return u_local


def simple_script(manual_precompilation=True):
    if manual_precompilation:
        start = time.time()
        print('Manually precompiling... ', end='')
        crunching_numbers.compile((int32,))
        end = time.time()
        print(f'{end - start:.4f}')

    workload = int(1.e8)
    print('pure python... ', end='')
    start = time.time()
    crunching_numbers.py_func(workload)
    end = time.time()
    print(f'{end - start:.4f}')

    workload = int32(1.e8)
    print('numba compiled version...  ', end='')
    start = time.time()
    crunching_numbers(workload)
    end = time.time()
    print(f'{end - start:.4f}')


def heat_diffusion_script(manual_precompilation=True):
    # function header:
    # heat_diffusion(size, dx, dt, alpha, steps)
    if manual_precompilation:
        start = time.time()
        print('Manually precompiling... ', end='')
        heat_diffusion.compile((int32, float64, float64, float64, int32,))
        end = time.time()
        print(f'{end - start:.4f}')

    args = (int(1.e4), 0.1, 0.001, 1., int(1.e4))
    print('pure python... ', end='')
    start = time.time()
    heat_diffusion.py_func(*args)
    end = time.time()
    print(f'{end - start:.4f}')

    args_numba = (int32(1.e4), float64(0.1), float64(0.001), float64(1.), int32(1.e4))
    print('numba compiled version...  ', end='')
    start = time.time()
    heat_diffusion(*args_numba)
    end = time.time()
    print(f'{end - start:.4f}')


def heat_diffusion_class_script():
    # hd = HeatDiffusionClass(int(1.e4), float64(.1), float64(1.))
    hd = HeatDiffusionClass()
    print('simulating... ', end='')
    start = time.time()
    # hd.simulate(dt=.1, steps=int(2.e3))
    hd.simulate(float64(.001), int32(1.e4))
    end = time.time()
    print(f'{end - start:.4f}')


def main():
    print('heat diffusion class script')
    heat_diffusion_class_script()
    print('heat diffusion class script')
    heat_diffusion_class_script()
    print('')
    print('heat diffusion script')
    heat_diffusion_script()


if __name__ == '__main__':
    main()
