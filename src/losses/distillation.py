from typing_extensions import Annotated
from pydantic import BaseModel
import torch
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy

class DistillationLoss(BaseModel):
    similarity_teacher: Annotated[list[list[float]], "Similarity graph for the whole dataset"]
    similarity_student: Annotated[list[list[float]], "Similarity graph for distillated dataset"]
    distillation_loss: Annotated[str, "Distillation loss to use (KL or MMD or MSE)"] = "MSE"
    temperature: Annotated[float, "Temperature parameter for KL loss."] = 1.0

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Distillation loss function: {self.distillation_loss}")
        print(f"  Whole dataset similarity: {self.similarity_teacher}")
        print(f"  Distilled set similarity: {self.similarity_student}")
        
    def compute_loss(self):
        
        print(f"Computing distillation loss using {self.distillation_loss}...")
        print(f"Similarity Teacher: {self.similarity_teacher}")
        print(f"Similarity Student: {self.similarity_student}")
        
        if self.distillation_loss == "MSE":
            return mse_loss(similarity_teacher=self.similarity_teacher,
                            similarity_student=self.similarity_student)
            
        if self.distillation_loss == "MMD":
            return mmd_loss(similarity_teacher=self.similarity_teacher,
                            similarity_student=self.similarity_student)
            
        elif self.distillation_loss == "KL":
            
            def delta(i,j, epsilon=1e-3):
                return 1.0 if abs(i - j) < epsilon else 0.0
            
            def distribution_teacher(s):
                output = 0.0
                N = len(self.similarity_teacher)
                for i in range(N):
                    for j in range(N):
                        output += delta(s, self.similarity_teacher[i][j])
                return output / (N ** 2)
            
            def distribution_student(s):
                output = 0.0
                M = len(self.similarity_student)
                for i in range(M):
                    for j in range(M):
                        output += delta(s, self.similarity_student[i][j])
                return output / (M ** 2)
            
            return kl_loss(distribution_teacher, distribution_student, self.temperature)
            
        else:
            raise ValueError(f"Unsupported distillation loss: {self.distillation_loss}")


def mse_loss(similarity_teacher, similarity_student) -> float:
    return torch.nn.functional.mse_loss(
        torch.tensor(similarity_teacher),
        torch.tensor(similarity_student)
    )


def mmd_loss(similarity_teacher, similarity_student) -> float:
    
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    metric = MaximumMeanDiscrepancy()
    metric.attach(default_evaluator, "mmd")

    state = default_evaluator.run([[torch.tensor(similarity_teacher),
                                    torch.tensor(similarity_student)]])

    return state.metrics["mmd"]


def kl_loss(teacher, student, temperature) -> float:
    
    n=1000
    support = torch.linspace(-temperature, temperature, n)
    p = torch.tensor([teacher(s.item()) for s in support])
    q = torch.tensor([student(s.item()) for s in support])
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    
    print(f"Teacher distribution: {torch.sum(p).item()}")
    print(f"Student distribution: {torch.sum(q).item()}")
    
    return torch.nn.functional.kl_div(p, q, reduction='batchmean')
