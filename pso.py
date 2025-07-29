import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool


def pso(cost_fn, dim, n_particles=30, n_iters=100,
        bounds=(0, 1), w=0.7, c1=1., c2=1., print_every=False, stepsize=1, randomness=0,
        visualize=False, checking_pos=False, seed=0, num_processes=1):
    np.random.seed(seed)
    # Bounds
    lb, ub = np.array(bounds[0]), np.array(bounds[1])
    lb = np.full(dim, lb) if np.isscalar(lb) else lb
    ub = np.full(dim, ub) if np.isscalar(ub) else ub

    # Initialize particles
    pos = np.random.uniform(lb, ub, size=(n_particles, dim))
    vel = np.zeros((n_particles, dim))
    pbest = pos.copy()
    pbest_val = np.array([cost_fn(p)['cost_total'] for p in pbest])

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    with Pool(processes=num_processes) as pool:

        for it in range(n_iters):
            start_time = time.time()
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)

            # Update velocity and position
            random_direction = np.random.randn(*vel.shape)
            rand_norms = np.linalg.norm(random_direction, axis=1, keepdims=True)  # shape (n_particles, 1)
            rand_normalized = random_direction / (rand_norms + 1.e-12)  # add epsilon to avoid division by zero

            vel = (w * vel
                   + c1 * r1 * (pbest - pos)
                   + c2 * r2 * (gbest - pos))

            vel_norms = np.linalg.norm(vel, axis=1, keepdims=True)  # shape (n_particles, 1)
            vel_normalized = vel / (vel_norms + 1.e-12)  # add epsilon to avoid division by zero

            vel_final = stepsize * ((1 - randomness) * vel_normalized + randomness * rand_normalized)

            pos += vel_final
            pos = np.clip(pos, lb, ub)

            # Evaluate
            # vals = np.array([cost_fn(p)['cost_total'] for p in pos])

            list_of_pbest_val = pool.map(cost_fn, pos)
            vals = np.array([p['cost_total'] for p in list_of_pbest_val])

            # Update personal best
            better = vals < pbest_val
            pbest[better] = pos[better]
            pbest_val[better] = vals[better]

            # Update global best
            gbest_idx = np.argmin(pbest_val)
            if pbest_val[gbest_idx] < gbest_val:
                gbest_val = pbest_val[gbest_idx]
                gbest = pbest[gbest_idx].copy()

            if print_every is not None and it % print_every == 0:
                print(f"Iter {it + 1}/{n_iters}: Best cost = {gbest_val:.2e}, "
                      f"time per iteration = {time.time() - start_time:.2e} s")
                if checking_pos:
                    check_cost(gbest, cost_fn)
                if visualize:
                    visualization(vel_final, pos, pbest, pbest_val, gbest, gbest_val)

    return gbest, gbest_val


def check_cost(pos, cost_fn):
    print(f'actuation strat: {pos}')
    print(f'cost total: {cost_fn(pos)["cost_total"]:.2e}, cost comfort:  {cost_fn(pos)["cost_comfort"].sum():.2e}')


def visualization(vel, pos, pbest, pbest_val, gbest, gbest_val):
    x = 0

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].set_title("new position")
    axs[0, 1].set_title("velocity")
    for particle in range(pos.shape[0]):
        # axs[0, 0].plot(pos[particle, :], color=(0, 0, 0, .1))
        axs[0, 0].step(range(24), pos[particle, :], where='post', color=(0, 0, 0, .1))
        axs[0, 1].step(range(24), vel[particle, :], where='post', color=(0, 0, 0, .1))

    axs[1, 0].set_title("best")
    axs[1, 0].step(range(24), gbest, where='post', color=(0, 0, 0, 1))

    plt.show(block=True)
