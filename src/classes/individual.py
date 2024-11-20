import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from classes import Parameters


class Individual:
    # Limites
    sigma: tuple = Parameters.sigma
    size:  int   = Parameters.dimensions

    # F. objetivo
    A:  int = 10
    Ai: int = A * size

    # Parâmetros de mutação
    tau:       float = 1 / np.sqrt(2. * np.sqrt(size))
    tau_prime: float = 1 / np.sqrt(2. * size)

    def __init__(self, genes: np.ndarray=None, strategy: np.ndarray=None):
        if genes is None:
            self.__genes = np.random.uniform(*Parameters.range, Parameters.dimensions)
        else:
            self.__genes = genes

        if strategy is None:
            self.__strategy = np.random.uniform(*self.sigma, Parameters.dimensions)
        else:
           self.__strategy = strategy 

        self.__fitness_value = None
    

    def evaluate(self):
        self.__fitness_value = self.Ai + np.sum(self.__genes**2 - self.A * np.cos(2 * np.pi * self.__genes))

    def mutate_simple(self):
        """Mutação log-normal nas estratégias e genes."""

        t0_n = self.tau_prime * np.random.normal(0, 1)

        for idx in range(self.size): 
            if np.random.rand() < Parameters.mutation_rate:
                self.__strategy[idx] *= np.exp(t0_n + self.tau * np.random.normal(0, 1))
                self.__genes[idx] += self.__strategy[idx] * np.random.normal(0, 1)
                self.__genes[idx]  = np.clip(self.__genes[idx], *Parameters.range)
                
    def mutate_vect(self):
        """Mutação log-normal nas estratégias e genes com vetorização."""
        
        mutation_mask = np.random.rand(self.size) < Parameters.mutation_rate

        t0_n = self.tau_prime * np.random.normal(0, 1)        

        # Atualização dos step sizes para todos os elementos onde mutation_mask é True
        noise_strategy = t0_n + self.tau * np.random.normal(0, 1, self.size)
        self.__strategy[mutation_mask] *= np.exp(noise_strategy[mutation_mask])
        
        noise_genes = self.__strategy * np.random.normal(0, 1, self.size)
        self.__genes[mutation_mask] += noise_genes[mutation_mask]
        
        self.__genes = np.clip(self.__genes, *Parameters.range)

    #Mutação
    mutate = mutate_simple if Parameters.dimensions < 30 else mutate_vect

    @property
    def strategy(self):
        return self.__strategy
    
    @property
    def genes(self):
        return self.__genes

    @property
    def x(self):
        return self.__genes[0]

    @property
    def y(self):
        return self.__genes[1]
    
    @property
    def z(self):
        return self.__genes[2]

    @property
    def fitness_value(self):
        return self.__fitness_value

    @genes.setter
    def genes(self, new_genes):

        if len(new_genes) != self.size:
            raise ValueError(f"O cromossomo deve ter {self.size} genes.")
        
        new_genes = np.clip(new_genes, *Parameters.range)
        
        self.__genes = new_genes

    @strategy.setter
    def strategy(self, new_strategy):

        if len(new_strategy) != self.size:
            raise ValueError(f"A estratégia deve ter {self.size} dimensões.")
        
        new_strategy = np.clip(new_strategy, *self.sigma)
        
        self.__strategy = new_strategy

    def __repr__(self):
        return f"Individual(genes={self.__genes}, strategy={self.__strategy}, fitness_value={self.__fitness_value:.10f})"


    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}], fitness_value = {self.fitness_value:.10f}"


    def __eq__(self, other: "Individual") -> bool:
        if isinstance(other, Individual):
            return np.all(self.__genes == other.genes)
        return False

    def __len__(self):
        return len(self.__genes)

    def __lt__(self, other):
        if self.__fitness_value == other.fitness_value:
            return np.linalg.norm(self.__genes) < np.linalg.norm(other.genes)
        return self.__fitness_value < other.fitness_value
