from typing import Annotated
from pydantic import BaseModel

import matplotlib.pyplot as plt
import torch
import lightly_train

from utils import TrainingHyperparameters, DistillationHyperparameters
from losses import CPDDLoss
from models import get_resnet18_model
from datasets import load_cifar10

class CPDD(BaseModel):
    dataset:        Annotated[str, "The dataset that will be distillated."]     = "cifar10"
    teacher:        Annotated[str, "Teacher model that gives ground truth."]    = "dinov3"
    student:        Annotated[str, "Student model that learns from the teacher."] = "resnet18"
    training_hyperparameters:       Annotated[TrainingHyperparameters, "Training hyperparameters."] = TrainingHyperparameters()
    distillation_hyperparameters:   Annotated[DistillationHyperparameters, "Distillation hyperparameters."] = DistillationHyperparameters()
    student_model:  Annotated[object, "The student model instance."]            = None
    teacher_model:  Annotated[object, "The teacher model instance."]            = None

    def display_config(self):
        print(f"Dataset:        {self.dataset}")
        print(f"Teacher Model:  {self.teacher}")
        print(f"Student Model:  {self.student}")
        self.training_hyperparameters.display_config()
        self.distillation_hyperparameters.display_config()
    
    def __init__(self):
        super().__init__()
        
        # Initialize the teacher model
        if self.teacher == "dinov3":
            self.teacher_model = lightly_train.load_model("dinov3/convnext-small-ltdetr-coco")
        else:
            raise ValueError(f"Unsupported teacher model: {self.teacher}")
        
        # Initialize the student model
        if self.student == "resnet18":
            self.student_model = get_resnet18_model(num_classes=10)
        else:
            raise ValueError(f"Unsupported student model: {self.student}")
        

    def compute_teacher_embeddings(self, dataset):
        embeddings = []
        labels = []
        
        if dataset == "cifar10":
            train_loader, _ = load_cifar10(batch_size=self.training_hyperparameters.batch_size)
            self.teacher_model.eval()
            with torch.no_grad():
                for images, _ in train_loader:
                    outputs = self.teacher_model.predict(images.to("cuda" if torch.cuda.is_available() else "cpu"))
                    embeddings.extend(outputs.cpu().numpy())
        
        if dataset == "coco128":
            # Placeholder for COCO128 dataset loading and embedding computation
            lightly_train.pretrain(
                out="out/my_experiment",  # Output directory
                data="coco128_unlabeled",  # Directory with images
                model="dinov3/vitt16",  # Model to train
                method="distillation",  # Pretraining method
                method_args={
                    "teacher": "dinov3/vits16"  # Teacher model for distillation
                },
                epochs=5,  # Small number of epochs for demonstration
                batch_size=32,  # Small batch size for demonstration
            )
        
        return embeddings, labels
    
    def distill(self):
        # Compute embeddings for teacher and student
        teacher_embeddings, _ = self.compute_embeddings(self.teacher_model, self.train_loader)
        student_embeddings, _ = self.compute_embeddings(self.student_model, self.train_loader)
        
        # Initialize CPDD Loss
        cpdd_loss = CPDDLoss(
            temperature=self.distillation_hyperparameters.temperature,
            distillation_loss=self.distillation_hyperparameters.loss_type,
            lambda_cpdd=self.distillation_hyperparameters.lambda_distillation,
            embeddings_teacher=teacher_embeddings,
            embeddings_student=student_embeddings
        )
        
        # Compute loss
        loss = cpdd_loss.compute_loss()
        print(f"Total CPDD Loss: {loss}")
        

if __name__ == "__main__":
    config = CPDD()
    config.display_config()
    
    images, labels = next(iter(config.train_loader))
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()
