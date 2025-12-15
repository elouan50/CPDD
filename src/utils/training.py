from typing import Annotated, Optional
from pydantic import BaseModel

class TrainingHyperparameters(BaseModel):
    epochs: Annotated[Optional[int], "Number of training epochs."] = 10
    batch_size: Annotated[Optional[int], "Size of each training batch."] = 32
    learning_rate: Annotated[Optional[float], "Learning rate for the optimizer."] = 0.001
    weight_decay: Annotated[Optional[float], "Weight decay (L2 regularization) factor."] = 1e-4
    momentum: Annotated[Optional[float], "Momentum factor for the optimizer."] = 0.9
    dropout_rate: Annotated[Optional[float], "Dropout rate for regularization."] = 0.5
    
    def display_config(self):
        print(f"Training Hyperparameters:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Weight Decay: {self.weight_decay}")
        print(f"  Momentum: {self.momentum}")
        print(f"  Dropout Rate: {self.dropout_rate}")
