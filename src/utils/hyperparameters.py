from typing import Annotated
from pydantic import BaseModel

class TrainingHyperparameters(BaseModel):
    epochs: Annotated[int, "Number of training epochs."] = 10
    batch_size: Annotated[int, "Size of each training batch."] = 32
    learning_rate: Annotated[float, "Learning rate for the optimizer."] = 0.001
    weight_decay: Annotated[float, "Weight decay (L2 regularization) factor."] = 1e-4
    momentum: Annotated[float, "Momentum factor for the optimizer."] = 0.9
    dropout_rate: Annotated[float, "Dropout rate for regularization."] = 0.5
