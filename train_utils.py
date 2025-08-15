"""
Training utilities for MNIST digit recognition project
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import Config

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class TrainingHistory:
    """Track training history and metrics"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None, lr=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        if self.train_accuracies and self.val_accuracies:
            axes[0, 1].plot(self.epochs, self.train_accuracies, label='Train Acc', marker='o')
            axes[0, 1].plot(self.epochs, self.val_accuracies, label='Val Acc', marker='s')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate plot
        if self.learning_rates:
            axes[1, 0].plot(self.epochs, self.learning_rates, marker='o')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1, 1].plot(self.epochs, loss_diff, marker='o', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def train_classifier(model, train_loader, val_loader, epochs=None, device=None):
    """
    Train a classification model with comprehensive tracking
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs (defaults to Config.NN_EPOCHS)
        device: Device to train on (defaults to Config.DEVICE)
    
    Returns:
        tuple: (trained_model, training_history)
    """
    if epochs is None:
        epochs = Config.NN_EPOCHS
    if device is None:
        device = Config.DEVICE
    
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.NN_LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Setup early stopping and history tracking
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)
    history = TrainingHistory()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate epoch metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        scheduler.step(val_loss_avg)
        
        # Update history
        history.update(epoch + 1, train_loss_avg, val_loss_avg, train_acc, val_acc, current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Early stopping check
        if early_stopping(val_loss_avg, model):
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.1f}s')
    
    return model, history

def train_vae(model, train_loader, val_loader, epochs=None, device=None):
    """
    Train a VAE model with comprehensive tracking
    
    Args:
        model: VAE model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs (defaults to Config.VAE_EPOCHS)
        device: Device to train on (defaults to Config.DEVICE)
    
    Returns:
        tuple: (trained_model, training_history)
    """
    if epochs is None:
        epochs = Config.VAE_EPOCHS
    if device is None:
        device = Config.DEVICE
    
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.VAE_LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Setup early stopping and history tracking
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)
    history = TrainingHistory()
    
    print(f"Training VAE on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, _) in enumerate(train_pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            loss_dict = model.loss_function(x_recon, data, mu, logvar)
            
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_dict['recon_loss'].item()
            train_kl_loss += loss_dict['kl_loss'].item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Total': f'{loss.item():.2f}',
                'Recon': f'{loss_dict["recon_loss"].item():.2f}',
                'KL': f'{loss_dict["kl_loss"].item():.2f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for data, _ in val_pbar:
                data = data.to(device)
                x_recon, mu, logvar = model(data)
                loss_dict = model.loss_function(x_recon, data, mu, logvar)
                
                val_loss += loss_dict['total_loss'].item()
                val_recon_loss += loss_dict['recon_loss'].item()
                val_kl_loss += loss_dict['kl_loss'].item()
                
                val_pbar.set_postfix({
                    'Total': f'{loss_dict["total_loss"].item():.2f}',
                    'Recon': f'{loss_dict["recon_loss"].item():.2f}',
                    'KL': f'{loss_dict["kl_loss"].item():.2f}'
                })
        
        # Calculate epoch metrics
        train_loss_avg = train_loss / len(train_loader.dataset)
        val_loss_avg = val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        scheduler.step(val_loss_avg)
        
        # Update history
        history.update(epoch + 1, train_loss_avg, val_loss_avg, lr=current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): '
              f'Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, '
              f'Train Recon: {train_recon_loss/len(train_loader.dataset):.4f}, '
              f'Train KL: {train_kl_loss/len(train_loader.dataset):.4f}, LR: {current_lr:.6f}')
        
        # Generate samples periodically
        if (epoch + 1) % 10 == 0:
            visualize_vae_samples(model, epoch + 1, device)
        
        # Early stopping check
        if early_stopping(val_loss_avg, model):
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.1f}s')
    
    return model, history

def evaluate_classifier(model, test_loader, device=None):
    """
    Comprehensive evaluation of a classification model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics
    """
    if device is None:
        device = Config.DEVICE
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader)
    
    # Generate classification report
    class_report = classification_report(all_labels, all_predictions, 
                                       target_names=[str(i) for i in range(10)],
                                       output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{Config.RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels
    }

def visualize_vae_samples(model, epoch, device=None, num_samples=16):
    """Visualize VAE generated samples"""
    if device is None:
        device = Config.DEVICE
    
    model.eval()
    
    with torch.no_grad():
        samples = model.generate(num_samples, device)
        
        # Create grid plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Generated Samples - Epoch {epoch}')
        
        for i in range(num_samples):
            row = i // 4
            col = i % 4
            
            # Convert to numpy and denormalize
            img = samples[i].cpu().squeeze().numpy()
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{Config.RESULTS_DIR}/vae_samples_epoch_{epoch}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

def save_model(model, model_name, epoch=None, optimizer=None, history=None):
    """Save model with metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'config': {
            'device': str(Config.DEVICE),
            'timestamp': time.time()
        }
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if history is not None:
        save_dict['history'] = history.__dict__
    
    save_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pth')
    torch.save(save_dict, save_path)
    print(f'Model saved to {save_path}')
    
    return save_path