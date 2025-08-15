"""
Data utilities for MNIST digit recognition project
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from config import Config

def get_data_transforms(train=True):
    """
    Get data transforms for training and testing
    
    Args:
        train (bool): Whether to get transforms for training (with augmentation)
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if train:
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((Config.NORMALIZE_MEAN,), (Config.NORMALIZE_STD,))
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((Config.NORMALIZE_MEAN,), (Config.NORMALIZE_STD,))
        ])
    
    return transform

def get_data_loaders(batch_size=None, val_split=0.1):
    """
    Get data loaders for training, validation, and testing
    
    Args:
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of training data to use for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.NN_BATCH_SIZE
    
    # Get transforms
    train_transform = get_data_transforms(train=True)
    test_transform = get_data_transforms(train=False)
    
    # Download and load datasets
    train_dataset = datasets.MNIST(
        root=Config.DATA_DIR,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = datasets.MNIST(
        root=Config.DATA_DIR,
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def visualize_samples(data_loader, num_samples=8, title="Sample Images"):
    """
    Visualize sample images from the data loader
    
    Args:
        data_loader: DataLoader containing images
        num_samples (int): Number of samples to visualize
        title (str): Title for the plot
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Select subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        row = i // 4
        col = i % 4
        
        # Denormalize image for visualization
        image = image * Config.NORMALIZE_STD + Config.NORMALIZE_MEAN
        image = torch.clamp(image, 0, 1)
        
        axes[row, col].imshow(image.squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {label.item()}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.RESULTS_DIR}/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, data_loader, num_samples=8, device=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained model
        data_loader: DataLoader containing test images
        num_samples (int): Number of samples to visualize
        device: Device to run inference on
    """
    if device is None:
        device = Config.DEVICE
    
    model.eval()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Select subset and move to device
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    predicted = predicted.cpu()
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Model Predictions')
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        # Denormalize image for visualization
        image = images[i] * Config.NORMALIZE_STD + Config.NORMALIZE_MEAN
        image = torch.clamp(image, 0, 1)
        
        axes[row, col].imshow(image.squeeze(), cmap='gray')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[row, col].set_title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}', 
                                color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.RESULTS_DIR}/predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

class MNISTDatasetCustom(torch.utils.data.Dataset):
    """
    Custom MNIST Dataset class for loading from binary files (backward compatibility)
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx]
        
        if self.transform:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.unsqueeze(0)  # Add channel dimension
        
        return image, label