from typing import Annotated, Literal
from pydantic import BaseModel
from tqdm import trange
from time import sleep

import matplotlib.pyplot as plt
import torch
import lightly_train

from utils import SimCLRTrainingHyperparameters, DistillationHyperparameters
from losses import DistillationLoss, InfoNCELoss
from models import ResNetSimCLR
from datasets import load_cifar10
from simclr.simclr import SimCLR

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CPDD(BaseModel):
    dataset_name:                   Annotated[str, "The dataset that will be distillated."]                 = "cifar10"
    model_architecture:             Annotated[Literal["resnet18", "resnet50"], "DNN architecture that is used for CPDD algorithm."]     = "resnet18"
    simclr_training_hyperparameters:Annotated[SimCLRTrainingHyperparameters, "Training hyperparameters."]   = SimCLRTrainingHyperparameters()
    distillation_hyperparameters:   Annotated[DistillationHyperparameters, "Distillation hyperparameters."] = DistillationHyperparameters()
    cpdd_epochs:                    Annotated[int, "Number of epochs for CPDD."]                            = 10
    dataset:                        Annotated[object, "The dataset instance."]                              = None
    distilled_dataset:              Annotated[object, "The distilled dataset instance."]                    = None
    model:                          Annotated[object, "The model instance."]                                = None
    
    def __init__(self):
        super().__init__()
        
        # Initialize the dataset
        if self.dataset_name == "cifar10":
            self.dataset = load_cifar10()[0]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Initialize the model
        if self.model_architecture == "resnet18":
            self.model = ResNetSimCLR(base_model=self.model_architecture, out_dim=10)
        else:
            raise ValueError(f"Unsupported student model: {self.model_architecture}")

    def display_config(self):
        print("-----------------------------------")
        print(f"Dataset:        {self.dataset_name}")
        print(f"Model:          {self.model_architecture}")
        self.simclr_training_hyperparameters.display_config()
        self.distillation_hyperparameters.display_config()
        print("-----------------------------------")

    def distillation(self):
        updated_distilled_dataset = self.dataset
        
        
        return updated_distilled_dataset
        
    def training(self):
        ## Use lightly_train to pretrain the model with SimCLR
        
        # lightly_train.pretrain(
        #     out="crash_test",
        #     data="src/coco128_unlabeled",
        #     model="torchvision/resnet18",
        #     method="simclr",
        #     epochs=1,
        #     batch_size=256,
        #     overwrite=True,
        # )
        
        ## Or use the imported implementation of SimCLR from src/simclr/simclr.py
        simclr = SimCLR(
            data='./datasets',
            dataset_name=self.dataset_name,
            arch=self.model_architecture,
            workers=12,
            epochs=self.simclr_training_hyperparameters.epochs,
            batch_size=self.simclr_training_hyperparameters.batch_size,
            lr=self.simclr_training_hyperparameters.learning_rate,
            weight_decay=self.simclr_training_hyperparameters.weight_decay,
            disable_cuda=False,
            fp16_precision=False,
            temperature=self.simclr_training_hyperparameters.temperature,
            )
        updated_model = simclr.train()
        
        return updated_model
    
    def start(self):
        
        print("Starting CPDD...")
        sleep(1)
        
        for epoch in trange(self.cpdd_epochs):
            # Distillation phase
            self.distilled_dataset = self.distillation()
            
            # Training phase
            self.model = self.training()
            sleep(0.2)
        
        print("Done.")
        
        return self.dataset


if __name__ == "__main__":
    try:
        config = CPDD()
        config.display_config()
        distilled_dataset = config.start()
    except Exception as e:
        print(f"Error: {e}")

    # images, labels = next(iter(config.dataset))
    # print(labels[0].item(),':', cifar10_labels[labels[0].item()])
    # plt.imshow(images[0].permute(1, 2, 0)*0.5+0.5)
    # plt.show()
