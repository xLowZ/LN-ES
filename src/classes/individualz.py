import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from classes import Parameters

class IndividualZ:
    # Limites
    sigma: tuple = (-10, 10)
    size:  int   = Parameters.dimensions - 2

    # F. objetivo
    A:  int = 10
    Ai: int = A * Parameters.dimensions

    # Parâmetros de mutação
    tau:       float = 1 / np.sqrt(2. * np.sqrt(size))
    tau_prime: float = 1 / np.sqrt(2. * size)

    def __init__(self, genes: np.ndarray=None, strategy: float=None):
        self.__genes = genes

        if strategy is None:
            self.__strategy = np.random.uniform(*self.sigma)
        else:
           self.__strategy = strategy 

        self.__fitness_value = None
    
    def evaluate(self):
        self.__fitness_value = self.Ai + np.sum(self.__genes ** 2 - self.A * np.cos(2 * np.pi * self.__genes))

    def mutate(self):
        """Mutação log-normal na estratégia e gene."""

        t0_n = self.tau_prime * np.random.normal(0, 1)

        if np.random.rand() < Parameters.mutation_rate:
            self.__strategy *= np.exp(t0_n + self.tau * np.random.normal(0, 1))
            self.z += self.__strategy * np.random.normal(0, 1)
            self.z  = np.clip(self.z, *Parameters.range)
                
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

        if len(new_genes) != Parameters.dimensions:
            raise ValueError(f"O cromossomo deve ter {Parameters.dimensions} genes.")
        
        new_genes = np.clip(new_genes, **Parameters.range)
        
        self.__genes = new_genes

    @z.setter
    def z(self, new_gene: float):
        new_gene = np.clip(new_gene, *Parameters.range)
        self.__genes[2] = new_gene

    def __repr__(self):
        return f"Individual(genes={self.__genes}, strategy={self.__strategy}, fitness_value={self.__fitness_value:.10f})"


    def __str__(self):
        return f"Individual: x = {self.x}, y = {self.y}, z = {self.z}, fitness_value = {self.fitness_value:.30f}"


    def __eq__(self, other: "IndividualZ") -> bool:
        if isinstance(other, IndividualZ):
            return np.all(self.__genes == other.genes)
        return False


    def __len__(self):
        return len(self.__genes)


    def __lt__(self, other):
        if self.__fitness_value == other.fitness_value:
            return np.linalg.norm(self.__genes) < np.linalg.norm(other.genes)
        return self.__fitness_value < other.fitness_value
