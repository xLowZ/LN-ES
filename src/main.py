import numpy as np
from classes import Parameters
from es import evolutionary_strategy, evolutionary_strategy_z
from utils import timer, evaluate_algorithm, algrthm_exe, print_top_solutions


def full_opt() -> None:

    solutions, elapsed_time = algrthm_exe(evolutionary_strategy, Parameters.execution_times, fixed_seed=False)

    evaluate_algorithm(solutions, Parameters.execution_times)

    print_top_solutions(solutions)

    timer(elapsed_time)

def only_z_opt() -> None:

    random_pos: np.ndarray = None

    solutions, elapsed_time = algrthm_exe(lambda: evolutionary_strategy_z(random_pos), Parameters.execution_times, fixed_seed=False)

    evaluate_algorithm(solutions, Parameters.execution_times)

    print_top_solutions(solutions)

    timer(elapsed_time)


def main() -> None:
    # only_z_opt()

    full_opt()


if __name__ == "__main__":
    main()

    