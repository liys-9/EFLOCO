import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool
import numpy as np
import tqdm
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde
import torch
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")


# Data generation helpers
def load_problems(batch_size):
    problem_gen_params = env_params['problem_gen_params']
    problems = get_random_problems(batch_size, env_params['node_cnt'], problem_gen_params)
    return problems


def get_random_problems(batch_size, node_cnt, problem_gen_params):
    # Generate an (a)symmetric distance matrix
    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Symmetrization: for ATSP we should NOT symmetrize (kept here from older code)
    while True:
        old_problems = problems.clone()
        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)).min(dim=3)
        if (problems == old_problems).all():
            break

    # Scale distance matrix
    scaled_problems = problems.float() / scaler
    return scaled_problems


# Recover coordinates via MDS
def recover_coordinates_from_distance_matrix(distance_matrix):
    # Use MDS to recover 2D coordinates from a distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1234)
    coordinates = mds.fit_transform(distance_matrix)
    return coordinates


def save_tours_to_file(atsp_matrices, tours, filename="atsp_solutions.txt"):
    with open(filename, "w") as f:
        for idx, (atsp_matrix, tour) in enumerate(zip(atsp_matrices, tours)):
            # Write ATSP distance matrix
            for row in atsp_matrix:
                f.write(" ".join(f"{distance:.2f}" for distance in row))
                f.write("\n")
            f.write("output ")
            # Write the tour
            f.write(" ".join(str(node_idx + 1) for node_idx in tour))
            f.write(f" {tour[0] + 1}\n\n")  # Return to the start

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=2560)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="concorde")
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"atsp{opts.max_nodes}_test_concorde.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Data generation parameters
    env_params = {
        'node_cnt': 20,
        'problem_gen_params': {
            'int_min': 0,
            'int_max': 1000 * 1000,
            'scaler': 1000 * 1000
        },
        'pomo_size': 20  # same as node_cnt
    }

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            batch_nodes_coord = load_problems(opts.batch_size)


            def solve_atsp(distance_matrix):
                # Recover coordinates from the distance matrix (MDS)
                coordinates = recover_coordinates_from_distance_matrix(distance_matrix)

                # Solve using Concorde (via an EUC_2D embedding)
                scale = 1e6
                solver = TSPSolver.from_data(coordinates[:, 0] * scale, coordinates[:, 1] * scale, norm="EUC_2D")
                solution = solver.solve(verbose=False)
                tour = solution.tour
                return tour


            # Multiprocessing
            with Pool(opts.batch_size) as p:
                tours = p.map(solve_atsp, [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

            for idx, tour in enumerate(tours):
                num_nodes = batch_nodes_coord.shape[1]
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + " " + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(" output ")
                    f.write(" ".join(str(node_idx + 1) for node_idx in tour))
                    f.write(f" {tour[0] + 1} \n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of ATSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")
    save_tours_to_file(atsp_matrices, tours, filename="atsp_solutions.txt")
    print("All tours and distance matrices were saved to 'atsp_solutions.txt'")