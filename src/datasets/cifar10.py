from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_cifar10(batch_size=32, num_workers=0):
    """
    Load CIFAR-10 dataset and return train and test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader
