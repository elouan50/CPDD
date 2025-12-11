from typing_extensions import Annotated
from pydantic import BaseModel
import numpy as np

from .distillation import DistillationLoss
from .info_nce import InfoNCELoss

class CPDDLoss(BaseModel):
    temperature: Annotated[float, "Temperature parameter for InfoNCE loss."] = 2.0
    distillation_loss: Annotated[str, "Distillation loss to use (KL or MMD or MSE)"] = "MSE"
    lambda_cpdd: Annotated[float, "Trade-off factor for CPDD loss."] = 0.5
    embeddings_teacher: Annotated[list[list[float]], "Embeddings from the teacher model."]
    embeddings_student: Annotated[list[list[float]],
                                    "Embeddings from the student model, for both augmentations. "
                                    "Expected format: [z1', z2', ..., zM', z1'', z2'', ..., zM''] "
                                    "where zi' and zi'' are two embedded augmentations of the same prototype."
                                    ]

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Lambda CPDD: {self.lambda_cpdd}")
        print(f"  Distillation loss function: {self.distillation_loss}")
        print(f"  Size of the whole dataset: {len(self.embeddings_teacher)}")
        print(f"  Size of distilled dataset: {len(self.embeddings_student)//2}")
    
    def compute_loss(self):
        
        info_nce_loss = InfoNCELoss(
            embeddings=self.embeddings_student,
            temperature=self.temperature
        ).compute_loss()
        
        M = len(self.embeddings_student) // 2
        
        similarity_teacher = embeddings_to_similarity(self.embeddings_teacher, self.temperature)
        similarity_student = embeddings_to_similarity(self.embeddings_student[:M], self.temperature)
        
        distillation_loss = DistillationLoss(
            similarity_teacher=similarity_teacher,
            similarity_student=similarity_student,
            distillation_loss=self.distillation_loss
        ).compute_loss()
        
        print(f"InfoNCE Loss: {info_nce_loss}")
        print(f"Distillation Loss: {distillation_loss}")
        
        total_loss = info_nce_loss + self.lambda_cpdd * distillation_loss
        return total_loss.item()


def sim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def embeddings_to_similarity(embeddings: list[list[float]], temperature: float) -> list[list[float]]:
    """Convert embeddings to similarity graph using cosine similarity."""
    embeddings_array = np.array(embeddings)
    N = embeddings_array.shape[0]
    
    similarity_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            similarity_matrix[i, j] = sim(embeddings_array[i], embeddings_array[j]) / temperature
    
    return similarity_matrix
