from typing_extensions import Annotated
from pydantic import BaseModel

class DistillationLoss(BaseModel):
    temperature: Annotated[float, "Temperature parameter for distillation loss."] = 2.0
    alpha: Annotated[float, "Weighting factor between distillation loss and student loss."] = 0.5

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha: {self.alpha}")
