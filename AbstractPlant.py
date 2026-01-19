from abc import ABC, abstractmethod

class AbstractPlant(ABC):
    
    @abstractmethod
    def step(self, U) -> float:
        pass