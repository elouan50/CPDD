from typing_extensions import Annotated
from pydantic import BaseModel
import numpy as np

class InfoNCELoss(BaseModel):
    embeddings: Annotated[list[list[float]], "Embedding from both augmentations."]
    temperature: Annotated[float, "Temperature parameter for distillation loss."] = 1.0

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Embeddings: {self.embeddings}")
        print(f"  Temperature: {self.temperature}")
        
    def compute_loss(self):
        M = len(self.embeddings) // 2
        t = self.temperature
        
        print(f"Number of prototypes (M): {M}")
        print(f"Embeddings: {self.embeddings}")
                
        loss = 0.0
        for i in range(M):
            
            num = np.exp(sim(np.array(self.embeddings[i]), np.array(self.embeddings[i + M])) / t)

            # Positive pair: (i, i + M)
            denom1 = 0.0
            for k in range(2 * M):
                if k != i:
                    denom1 += np.exp(sim(np.array(self.embeddings[i]), np.array(self.embeddings[k])) / t)
                        
            # Positive pair: (i + M, i)
            denom2 = 0.0
            for k in range(2 * M):
                if k != i + M:
                    denom2 += np.exp(sim(np.array(self.embeddings[i + M]), np.array(self.embeddings[k])) / t)
                   
            # Accumulate loss
            loss += - np.log(num / denom1) - np.log(num / denom2)
        
        return loss / (2 * M)


def sim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
