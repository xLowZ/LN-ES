import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils import timer, evaluate_algorithm, algrthm_exe, print_top_solutions
from classes import Parameters, Individual, IndividualZ


def initialization(population_size: int) -> list[Individual]:
    """
    Inicializa a população de indivíduos.

    Args:
        population_size (int): Tamanho inicial da população.

    Returns:
        list[Individual]: População inicial de soluções.
    """
    
    genes_population:    np.ndarray = np.random.uniform(*Parameters.range, (population_size, Parameters.dimensions))
    strategy_population: np.ndarray = np.random.uniform(*Parameters.sigma, (population_size, Parameters.dimensions))

    initial_population = [Individual(genes=genes_population[i], strategy=strategy_population[i]) for i in range(population_size)]

    for individual in initial_population:
        individual.evaluate()

    if population_size > Parameters.mu:
        initial_population.sort(key=lambda ind: ind.fitness_value)
        initial_population = initial_population[:Parameters.mu]

    return initial_population



def selection(population: list[Individual | IndividualZ]) -> Individual | IndividualZ:
    """
    Seleção dos pais para gerar um novo indivíduo.

    Args:
        population (list[Individual]): População atual

    Returns:
        Individual: Pai selecionado
    """
    
    winner_idx: int = 0
    pop_size:   int = len(population)

    
    if Parameters.selection_method == "random":

        winner_idx = np.random.choice(range(pop_size)) 


    elif Parameters.selection_method == "tournament":

        selected_indx = np.random.choice(range(pop_size), Parameters.tournament_candidates, replace=False)
        winner_idx = selected_indx[0]
        
        for idx in selected_indx[1:]:
            if population[idx].fitness_value < population[winner_idx].fitness_value:
                winner_idx = idx

    return population[winner_idx]


def crossover(first_parent: Individual, second_parent: Individual) -> tuple[Individual, Individual]:
    """
    Recombinação entre duas soluções para gerar dois novos indivíduos

    Args:
        first_parent (Individual): Primeira solução selecionada
        second_parent (Individual): Segunda solução selecionada

    Returns:
        tuple[Individual, Individual]: Duas novas soluções
    """

    ind_size: int = len(first_parent)

    first_parent_genes:  np.ndarray = first_parent.genes
    second_parent_genes: np.ndarray = second_parent.genes

    first_parent_strategy:  np.ndarray = first_parent.strategy
    second_parent_strategy: np.ndarray = second_parent.strategy

    first_child_genes:  np.ndarray = np.ones(ind_size)
    second_child_genes: np.ndarray = np.ones(ind_size)

    first_child_strategy:  np.ndarray = np.ones(ind_size)
    second_child_strategy: np.ndarray = np.ones(ind_size)


    if Parameters.crossover_method == "one_point":

        point: int = np.random.randint(1, ind_size)

        first_child_genes  = np.concatenate([first_parent_genes[:point], second_parent_genes[point:]])
        second_child_genes = np.concatenate([second_parent_genes[:point], first_parent_genes[point:]])
    
        first_child_strategy  = np.concatenate([first_parent_strategy[:point], second_parent_strategy[point:]])
        second_child_strategy = np.concatenate([second_parent_strategy[:point], first_parent_strategy[point:]])


    elif Parameters.crossover_method == "two_point":
        
        first_point, second_point = sorted(np.random.choice(range(1, ind_size), 2, replace=False))

        first_child_genes = np.concatenate([first_parent_genes[:first_point],
                                            second_parent_genes[first_point:second_point],
                                            first_parent_genes[second_point:]])
        
        first_child_strategy = np.concatenate([first_parent_strategy[:first_point],
                                            second_parent_strategy[first_point:second_point],
                                            first_parent_strategy[second_point:]])
                                           
        second_child_genes = np.concatenate([second_parent_genes[:first_point],
                                             first_parent_genes[first_point:second_point],
                                             second_parent_genes[second_point:]])
        
        second_child_strategy = np.concatenate([second_parent_strategy[:first_point],
                                             first_parent_strategy[first_point:second_point],
                                             second_parent_strategy[second_point:]])


    elif Parameters.crossover_method == "uniform":

        mask = np.random.randint(0, 2, size=ind_size).astype(bool)

        first_child_genes  = np.where(mask, second_parent_genes, first_parent_genes)
        second_child_genes = np.where(mask, first_parent_genes, second_parent_genes)

        first_child_strategy  = np.where(mask, second_parent_strategy, first_parent_strategy)               
        second_child_strategy = np.where(mask, first_parent_strategy, second_parent_strategy)
    

    elif Parameters.crossover_method == "arithmetic":

        alpha = np.random.uniform(0, 1)

        first_child_genes    = alpha * first_parent.genes + (1 - alpha) * second_parent.genes
        first_child_strategy = alpha * first_parent.strategy + (1 - alpha) * second_parent.strategy

        second_child_genes    = (1 - alpha) * first_parent.genes + alpha * second_parent.genes
        second_child_strategy = (1 - alpha) * first_parent.strategy + alpha * second_parent.strategy


    return Individual(first_child_genes, first_child_strategy), Individual(second_child_genes, second_child_strategy)


def mutate_and_evaluate_offspring(offspring: list[Individual | IndividualZ]) -> None:
    """
    Mutação e avaliação dos descendentes.

    Args:
        offspring (list[Individual | IndividualZ]): Lista de descendentes
    """

    for child in offspring:
        child.mutate()
        child.evaluate()


def create_new_generation(previous_generation: list[Individual]) -> list[Individual]:
    """
    Criação da nova geração utilizando `(μ + λ)-ES`.

    Args:
        previous_generation (list[Individual]): Pupulação atual

    Returns:
        list[Individual]: Nova geração com μ indivíduos
    """

    offspring: list[Individual] = []
    
    while len(offspring) < Parameters.lambda_:

        first_parent:  Individual = selection(previous_generation)
        second_parent: Individual = selection(previous_generation)
        
        if np.random.rand() < Parameters.crossover_rate:
            first_child, second_child = crossover(first_parent, second_parent)
        else:
            first_child  = Individual(np.copy(first_parent.genes), np.copy(first_parent.strategy))
            second_child = Individual(np.copy(second_parent.genes), np.copy(second_parent.strategy))

        offspring.extend([first_child, second_child])
    
    mutate_and_evaluate_offspring(offspring)
    
    new_generation = offspring + previous_generation

    new_generation.sort(key=lambda ind: ind.fitness_value)      

    return new_generation[:Parameters.mu]


def create_new_generation_one_mu(previous_generation: list[Individual]) -> list[Individual]:
    """
    Criação da nova geração `(1 + λ)-ES`.
    
    Args:
        previous_generation (list[Individual]): Pupulação atual

    Returns:
        list[Individual]: Nova geração com 1 indivíduo.
    """

    offspring: list[Individual] = []
    
    parent = previous_generation[0]

    while len(offspring) < Parameters.lambda_:

        child = Individual(np.copy(parent.genes), np.copy(parent.strategy)) 

        child.mutate()
        child.evaluate()

        offspring.append(child)
    
    new_generation = offspring + previous_generation

    new_generation.sort(key=lambda ind: ind.fitness_value)

    return new_generation[:Parameters.mu]


def evolutionary_strategy() -> Individual:
    """
    Função principal do algoritmo de `ES`.

    Returns:
        Individual: Melhor solução encontrada
    """

    if Parameters.mu == 1:
        create_new_gen = create_new_generation_one_mu
        pop_init = Parameters.mu + Parameters.lambda_
    else:
        create_new_gen = create_new_generation 
        pop_init = Parameters.mu

    population = initialization(pop_init)
    
    for generation in range(Parameters.iterations):

        population = create_new_gen(population)

        if (generation + 1) % (Parameters.iterations / 10) == 0 or generation == Parameters.iterations - 1:
            print(f"Geração {generation + 1}: Genes: {population[0].genes} Melhor fitness = {population[0].fitness_value}")
    
    return population[0]


def main() -> None:

    solutions, elapsed_time = algrthm_exe(evolutionary_strategy, Parameters.execution_times, fixed_seed=False)

    evaluate_algorithm(solutions, Parameters.execution_times)

    print_top_solutions(solutions)

    timer(elapsed_time)

if __name__ == "__main__":

    # np.random.seed(11)

    main()

