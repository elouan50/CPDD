import logging
import os
import sys
from pydantic import BaseModel
from typing import Annotated, Literal
from tqdm import tqdm

from simclr.data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr.models.resnet_simclr import ResNetSimCLR
from simclr.utils import save_config_file, accuracy, save_checkpoint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

torch.manual_seed(0)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class SimCLR(BaseModel):
    data: Annotated[str, "path to dataset"] = './data'
    dataset_name: Annotated[Literal["stl10", "cifar10"], "dataset name"] = 'cifar10'
    arch: Annotated[str, model_names] = 'resnet18'
    workers: Annotated[int, "number of workers"] = 12
    epochs: Annotated[int, "number of total epochs to run"] = 200
    batch_size: Annotated[int, "batch size"] = 256
    lr: Annotated[float, "initial learning rate"] = 0.0003
    weight_decay: Annotated[float, "weight decay (default: 1e-4)"] = 1e-4
    seed: Annotated[int, "seed for initializing training."] = None
    disable_cuda: Annotated[bool, "Disable CUDA"] = False
    fp16_precision: Annotated[bool, "Whether or not to use 16-bit precision GPU training."] = False
    out_dim: Annotated[int, "feature dimension (default: 128)"] = 128
    log_every_n_steps: Annotated[int, "Log every n steps"] = 100
    temperature: Annotated[float, "temperature"] = 0.07
    n_views: Annotated[int, "number of views"] = 2
    gpu_index: Annotated[int, "Gpu index."] = 0
    
    device: Annotated[str, "Device to use for training."] = 'cpu'
    train_loader: Annotated[object, "Training data loader."] = None
    model: Annotated[object, "The model instance."] = None
    optimizer: Annotated[object, "The optimizer instance."] = None
    scheduler: Annotated[object, "The learning rate scheduler instance."] = None
    writer: Annotated[object, "Tensorboard SummaryWriter instance."] = None
    criterion: Annotated[object, "Loss function."] = None
    
    def model_post_init(self, __context):
        
        # check if gpu training is available
        if not self.disable_cuda and torch.cuda.is_available():
            self.device = 'cuda'
            cudnn.deterministic = True
            cudnn.benchmark = True
        else:
            self.device = 'cpu'
            self.gpu_index = -1
        
        dataset = ContrastiveLearningDataset(f"{self.data}/{self.dataset_name}")
        
        train_dataset = dataset.get_dataset(self.dataset_name, self.n_views)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, drop_last=True)
        
        self.model = ResNetSimCLR(base_model=self.arch, out_dim=self.out_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                    eta_min=0, last_epoch=-1)
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self):

        scaler = GradScaler(device=self.device, enabled=self.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.epochs} epochs.")
        logging.info(f"Training with gpu: {not(self.disable_cuda)}.")

        for epoch_counter in range(self.epochs):
            for images, _ in tqdm(self.train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast(self.device, enabled=self.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.epochs)
        save_checkpoint({
            'epoch': self.epochs,
            'arch': self.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        return self.model
        
if __name__ == "__main__":
    simclr = SimCLR()
    simclr.train()
