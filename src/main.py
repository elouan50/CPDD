from typing import Annotated
from pydantic import BaseModel
from utils import TrainingHyperparameters

class CPDD(BaseModel):
    dataset: Annotated[str, "The dataset that will be ditillated."] = "cifar-10"
    teacher: Annotated[str, "Teacher model that gives ground truth."] = "dinov3/convnext-small-ltdetr-coco"
    student: Annotated[str, "Student model that will learn from the teacher."] = "resnet18"
    training_hyperparameters: Annotated[TrainingHyperparameters, "Training hyperparameters."] = TrainingHyperparameters()
    
    def display_config(self):
        print(f"Dataset: {self.dataset}")
        print(f"Teacher Model: {self.teacher}")
        print(f"Student Model: {self.student}")
        print("Training Hyperparameters:")
        for field in self.training_hyperparameters.__annotations__:
            value = getattr(self.training_hyperparameters, field)
            description = self.training_hyperparameters.__annotations__[field].__metadata__[0]
            print(f"  {field} ({description}): {value}")
            
            
    

if __name__ == "__main__":
    config = CPDD()
    config.display_config()
    
