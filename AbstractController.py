from abc import ABC, abstractmethod

class AbstractController(ABC):
    
    @abstractmethod
    def step(self, E) -> float:
        pass