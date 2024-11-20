import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
from classes import Parameters, Individual, IndividualZ
from utils import timer, evaluate_algorithm
from es.mu_plus_es import mutate_and_evaluate_offspring, selection

def initializationZ(population_size: int, cartesian_pos: np.ndarray=None, all_fixed: bool=True) -> list[IndividualZ]:
    """Inicializa a população de indivíduos.
    
    Args:
        population_size (int): Número de indivíduos na população.
        cartesian_pos (np.ndarray, opcional): Coordenadas fixas para x e y.
        all_fixed (bool, opcional): Se True, fixa x e y para todos os indivíduos.
    
    Returns:
        list[IndividualZ]: População inicial de soluções.
    """

    initial_population: list[IndividualZ] = []

    if all_fixed or cartesian_pos is not None:
        
        if cartesian_pos is None:
            fixed_x, fixed_y = np.random.uniform(*Parameters.range), np.random.uniform(*Parameters.range)
        else:
            fixed_x, fixed_y = cartesian_pos[0], cartesian_pos[1]

        for _ in range(population_size):
            # Valores fixos de x e y, z é aleatório
            genes = np.array([fixed_x, fixed_y, np.random.uniform(*Parameters.range)])
            initial_population.append(IndividualZ(genes))

    else:
        # Inicializa valores de x, y e z de forma aleatória para cada indivíduo
        genes_population = np.random.uniform(*Parameters.range, (population_size, Parameters.dimensions)) 
        strategy_population = np.random.uniform(*IndividualZ.sigma, population_size)     

        for i in range(population_size):
            initial_population.append(IndividualZ(genes=genes_population[i], strategy=strategy_population[i]))


    for individual in initial_population:
        individual.evaluate()

    if population_size > Parameters.mu:
        initial_population.sort(key=lambda ind: ind.fitness_value)
        initial_population = initial_population[:Parameters.mu]

    return initial_population


def crossoverZ(first_parent: IndividualZ, second_parent: IndividualZ) -> tuple[IndividualZ, IndividualZ]:

    alpha = np.random.uniform(0, 1)

    first_child_z = alpha * first_parent.z + (1 - alpha) * second_parent.z
    second_child_z = (1 - alpha) * first_parent.z + alpha * second_parent.z

    first_child_strategy = alpha * first_parent.strategy + (1 - alpha) * second_parent.strategy
    second_child_strategy = (1 - alpha) * first_parent.strategy + alpha * second_parent.strategy
    
    first_child_genes  = np.array([first_parent.x, first_parent.y, first_child_z])
    second_child_genes = np.array([second_parent.x, second_parent.y, second_child_z])
    
    return IndividualZ(first_child_genes, first_child_strategy), IndividualZ(second_child_genes, second_child_strategy)

def create_new_generationZ(previous_generation: list[Individual | IndividualZ]) -> list[IndividualZ]:
    """
    Criação da nova geração utilizando `(μ + λ)-ES`.

    Args:
        previous_generation (list[IndividualZ]): Pupulação atual

    Returns:
        list[IndividualZ]: Nova geração com μ indivíduos
    """

    new_generation: list[IndividualZ] = []
    
    while len(new_generation) < Parameters.lambda_:
    
        first_parent  = selection(previous_generation)
        second_parent = selection(previous_generation)
        
        if np.random.rand() < Parameters.crossover_rate:
            first_child, second_child = crossoverZ(first_parent, second_parent)
        else:
            first_child  = IndividualZ(np.copy(first_parent.genes), np.copy(first_parent.strategy))
            second_child = IndividualZ(np.copy(second_parent.genes), np.copy(second_parent.strategy))
        
        new_generation.extend([first_child, second_child])
    
    mutate_and_evaluate_offspring(new_generation)
    
    new_generation += previous_generation

    new_generation.sort(key=lambda ind: ind.fitness_value)
    
    return new_generation[:Parameters.mu]


def create_new_generation_oneZ(previous_generation: list[IndividualZ]) -> list[IndividualZ]:
    """
    Criação da nova geração `(1 + λ)-ES`.
    
    Args:
        previous_generation (list[IndividualZ]): Pupulação atual

    Returns:
        list[IndividualZ]: Nova geração com 1 indivíduo.
    """

    new_generation: list[IndividualZ] = []
    
    parent = previous_generation[0]

    while len(new_generation) < Parameters.lambda_:

        child = IndividualZ(np.copy(parent.genes), np.copy(parent.strategy)) 

        child.mutate()
        child.evaluate()

        new_generation.append(child)
    
    new_generation += previous_generation

    new_generation.sort(key=lambda ind: ind.fitness_value)
    
    return new_generation[:Parameters.mu]


def evolutionary_strategy_z(cartesian_pos: np.ndarray=None) -> IndividualZ:
    """
    Função principal do ES.

    Args:
        cartesian_pos (np.ndarray, optional): posição preestabelecida. Defaults to None.

    Returns:
        IndividualZ: Melhor solução.
    """

    if Parameters.mu == 1:
        create_new_gen = create_new_generation_oneZ
        pop_init = Parameters.mu + Parameters.lambda_
    else:
        create_new_gen = create_new_generationZ 
        pop_init = Parameters.mu

    population = initializationZ(pop_init, cartesian_pos=cartesian_pos)
    
    for generation in range(Parameters.iterations):
        population = create_new_gen(population)
        
        if (generation + 1) % (Parameters.iterations / 10) == 0 or generation == Parameters.iterations - 1:
            print(f"Geração {generation + 1}: Genes: {population[0].genes} Melhor fitness = {population[0].fitness_value}")
    
    return population[0]


def main():
    
    solutions: list[IndividualZ] = []

    start = time.time()
    for tests in range(Parameters.execution_times):
        np.random.seed(tests)

        best_solution = evolutionary_strategy_z()
        print(f"{tests+1}ª Exec: Melhor Solução: {best_solution.z}, Fitness: {best_solution.fitness_value}")

        solutions.append(best_solution)

    end = time.time()

    evaluate_algorithm(solutions, Parameters.execution_times)

    solutions = np.sort(solutions)

    for solution in solutions:
        print(solution)

    timer(end - start)

if __name__ == "__main__":
    # np.random.seed(2)
    main()    
