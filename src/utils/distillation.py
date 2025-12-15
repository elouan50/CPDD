from typing import Annotated
from pydantic import BaseModel

class DistillationHyperparameters(BaseModel):
    loss_type: Annotated[str, "Type of distillation loss to use (e.g., 'KL', 'MSE', 'MMD')."] = "MSE"
    temperature: Annotated[float, "Temperature parameter for distillation loss."] = 1.0
    lambda_distillation: Annotated[float, "Trade-off factor for distillation loss."] = 0.5

    
    def display_config(self):
        print(f"Distillation Hyperparameters:")
        print(f"  Loss Type: {self.loss_type}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Lambda Distillation: {self.lambda_distillation}")
