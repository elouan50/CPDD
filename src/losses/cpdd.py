from typing_extensions import Annotated
from pydantic import BaseModel

from .distillation import DistillationLoss
from .info_nce import InfoNCELoss

class CPDDLoss(BaseModel):
    temperature: Annotated[float, "Temperature parameter for distillation loss."] = 2.0
    lambda_cpdd: Annotated[float, "Trade-off factor for CPDD loss."] = 0.5

    def display_config(self):
        print(f"Distillation Loss Configuration:")
        print(f"  Temperature: {self.temperature}")
