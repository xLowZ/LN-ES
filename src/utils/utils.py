import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import time
from classes import Individual

def timer(elapsed_time: float) -> None:
    """
    Mostra o tempo decorrido formatado.
    """

    if elapsed_time < 60:
        print(f"Time: {elapsed_time:.4f} seconds")
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Time: {minutes} minute{''if minutes == 1 else 's'} and {seconds:.1f} second{''if seconds == 1 else 's'}")
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"Time: {hours} hour{''if hours == 1 else 's'}, {minutes} minute{''if minutes == 1 else 's'} and {seconds:.1f} second{''if seconds == 1 else 's'}")


def evaluate_algorithm(results: list[Individual], exec_times: int) -> None:
    """
    Avalia resultados obtidos e exibe boxplot.

    Args:
        results (list[Individual]): soluções encontradas em n execuções
        exec_times (int): quantidade de vezes que o AE foi executado
    """
    
    fnc = "Função Objetivo"

    solution_values: list[float] = [sol.fitness_value for sol in results]
    solution_values = np.sort(solution_values)

    for sol in results:
        print(sol.genes, sol.fitness_value)

    sorted_results = np.sort(results)

    mean_value     = np.mean(solution_values)
    median_value   = np.median(solution_values)
    std_value      = np.std(solution_values)
    best_value     = np.min(solution_values)
    worst_value    = np.max(solution_values)
    first_quartile = np.percentile(solution_values, 25)
    third_quartile = np.percentile(solution_values, 75)
    l2             = np.linalg.norm(sorted_results[0].genes)
    iiq            = third_quartile - first_quartile

    print(f"Algoritmo executado {exec_times} vezes.")
    print(f"Resulados para a {fnc}:")
    print(f"Média das soluções: {mean_value}")
    print(f"Mediana das soluções: {median_value}")
    print(f"Desvio padrão das soluções: {std_value}")
    print(f"Melhores Genes: {sorted_results[0].genes}")
    print(f"Melhor Fitness: {best_value}")
    print(f"Pior solução encontrada: {worst_value}")
    print(f"Primeiro Quartil: {first_quartile}")
    print(f"Terceiro Quartil: {third_quartile}")
    print(f"Intervalo Interquartil: {iiq}")
    print(f"Norma L2: {l2}")


    plt.figure(figsize=(10, 6))
    

    box = plt.boxplot(solution_values, vert=True, patch_artist=False,
                    #   boxprops=dict(facecolor='lightblue', color='blue'),
                      boxprops=dict(color='blue'),
                      whiskerprops=dict(color='blue'),
                      capprops=dict(color='black', linewidth=2),
                      medianprops=dict(color='red', linewidth=3))

    plt.title('Boxplot dos Valores de Fitness', fontsize=16, fontweight='bold')
    plt.xlabel('', fontsize=14)
    plt.ylabel('Valor de Fitness', fontsize=14)
    plt.grid(True)

    plt.rcParams['font.family'] = 'Arial' 
    plt.rcParams['font.size'] = 12

    plt.tight_layout()
    plt.show()


def algrthm_exe(ea_fnc: callable, num_exec: int, fixed_seed: bool=False) -> tuple[list[Individual], float]:
    """
    Função que executará o algoritmo.

    Args:
        ea_fnc (callable): Função principal do AE
        num_exec (int): número de execuções (nº de soluções)

    Returns:
        tuple[list[Individual], float]: Melhores soluções de cada execução e tempo total decorrido
    """

    solutions: list[Individual] = []

    start = time.time()

    for tests in range(num_exec):
        
        if not fixed_seed:
            np.random.seed(tests)

        best_solution = ea_fnc()
        print(f"{tests+1}ª Exec: Melhor Solução: {best_solution.genes}, Fitness: {best_solution.fitness_value}")

        solutions.append(best_solution)

    end = time.time()

    elapsed_time = end - start

    solutions.sort(key=lambda ind: ind.fitness_value)

    return solutions, elapsed_time


def print_top_solutions(solutions: list[Individual]) -> None:
    """
    Imprime as 5 melhores e 5 piores soluções encontradas

    Args:
        solutions (list[Individual]): melhores indivíduos de cada execução do AE
    """

    print("5 Melhores soluções:")
    for ind in solutions[:5]:
        print(f"\t{ind.genes}, {ind.fitness_value:.20f}")

    print("5 Piores soluções:")
    for ind in solutions[-5::][::-1]:
        print(f"\t{ind.genes}, {ind.fitness_value:.20f}")

if __name__ == '__main__':
    print("") 