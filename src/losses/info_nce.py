from typing_extensions import Annotated
from pydantic import BaseModel
import numpy as np

class InfoNCELoss(BaseModel):
    embeddings: Annotated[list[np.ndarray[float]], "Embedding from the first augmentation."]
    temperature: Annotated[float, "Temperature parameter for distillation loss."] = 1.0

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Embeddings2: {self.embeddings}")
        print(f"  Temperature: {self.temperature}")
        
    def compute_loss(self):
        # Placeholder for actual InfoNCE loss computation
        M = len(self.embeddings1)
        loss = 0.0
        for i in range(M):
            pass
        
        return loss
