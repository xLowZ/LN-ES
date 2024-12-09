class Parameters:
    mu: int = 15
    lambda_: int  = mu * 7
    dimensions: int = 30
    range: tuple = (-5.12, 5.12)
    sigma: tuple = (-10, 10) 
    tournament_candidates: int = 2
    crossover_rate: float = 0.7
    iterations: int = int(1e5) 
    mutation_rate: float = 1/dimensions
    selection_method: str = "tournament"
    crossover_method: str = "uniform"
    execution_times: int = 10
