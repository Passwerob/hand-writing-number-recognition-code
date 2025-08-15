# MNIST Digit Recognition - Optimized Implementation

A comprehensive and optimized implementation of handwritten digit recognition using PyTorch, featuring both traditional neural networks and Variational Autoencoders (VAE) for MNIST dataset.

## 🚀 Features

- **Multiple Model Architectures**: 
  - Improved Multi-layer Perceptron (MLP) with dropout and batch normalization
  - Convolutional Neural Network (CNN) for better feature extraction
  - Variational Autoencoder (VAE) for generative modeling
  - Convolutional VAE for improved image modeling

- **Advanced Training Features**:
  - GPU support with automatic device detection
  - Early stopping to prevent overfitting
  - Learning rate scheduling
  - Comprehensive progress tracking with tqdm
  - Data augmentation for better generalization
  - Validation split for model selection

- **Comprehensive Evaluation**:
  - Detailed classification reports
  - Confusion matrices with visualization
  - Training history plots
  - VAE sample generation and interpolation
  - Reconstruction quality evaluation

- **Production-Ready Code**:
  - Modular architecture with separate concerns
  - Configuration management
  - Command-line interface
  - Model saving and loading
  - Error handling and logging

## 📁 Project Structure

```
.
├── config.py              # Configuration settings
├── data_utils.py           # Data loading and preprocessing
├── models.py               # Model architectures
├── train_utils.py          # Training utilities and evaluation
├── train_classifier.py     # Classification training script
├── train_vae.py           # VAE training script
├── requirements.txt        # Dependencies
├── README.md              # This file
├── data/                  # Dataset directory (auto-created)
├── models/                # Saved models directory (auto-created)
└── results/               # Output plots and results (auto-created)
```

## 🛠 Installation

1. **Clone the repository** (or use the existing files):
```bash
git clone <repository-url>
cd mnist-digit-recognition
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **The required directories will be created automatically** when you run the scripts.

## 🎯 Quick Start

### Training a Classification Model

**Train a Convolutional Neural Network** (recommended):
```bash
python train_classifier.py --model conv --epochs 20 --visualize
```

**Train a Multi-layer Perceptron**:
```bash
python train_classifier.py --model mlp --epochs 20 --visualize
```

### Training a VAE Model

**Train a Variational Autoencoder**:
```bash
python train_vae.py --model mlp --epochs 50 --visualize
```

**Train a Convolutional VAE**:
```bash
python train_vae.py --model conv --epochs 50 --visualize
```

## ⚙️ Advanced Usage

### Command Line Options

#### Classification Training (`train_classifier.py`)
```bash
python train_classifier.py [OPTIONS]

Options:
  --model {mlp,conv}     Model architecture (default: conv)
  --epochs INT           Number of epochs (default: 20)
  --batch-size INT       Batch size (default: 128)
  --lr FLOAT             Learning rate (default: 0.001)
  --no-save              Don't save the trained model
  --no-eval              Skip final evaluation
  --visualize            Show sample visualizations
```

#### VAE Training (`train_vae.py`)
```bash
python train_vae.py [OPTIONS]

Options:
  --model {mlp,conv}     VAE architecture (default: mlp)
  --epochs INT           Number of epochs (default: 50)
  --batch-size INT       Batch size (default: 128)
  --lr FLOAT             Learning rate (default: 0.001)
  --latent-dim INT       Latent space dimension (default: 32)
  --beta FLOAT           Beta for β-VAE (default: 1.0)
  --no-save              Don't save the trained model
  --visualize            Show sample visualizations
```

### Configuration

Modify `config.py` to adjust default parameters:

```python
class Config:
    # Training hyperparameters
    NN_EPOCHS = 20
    NN_LEARNING_RATE = 0.001
    NN_BATCH_SIZE = 128
    
    # VAE hyperparameters
    VAE_EPOCHS = 50
    VAE_LATENT_DIM = 32
    VAE_BETA = 1.0
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-4
```

## 📊 Model Architectures

### 1. Improved Neural Network (MLP)
- Multiple hidden layers with configurable dimensions
- Batch normalization for stable training
- Dropout for regularization
- Xavier weight initialization

### 2. Convolutional Neural Network (CNN)
- 3 convolutional layers with batch normalization
- Max pooling for spatial reduction
- Fully connected layers with dropout
- Kaiming weight initialization

### 3. Variational Autoencoder (VAE)
- Encoder-decoder architecture
- Reparameterization trick for gradient flow
- β-VAE support for disentangled representations
- Sample generation capabilities

### 4. Convolutional VAE
- Convolutional encoder and transpose convolutional decoder
- Better suited for image data
- Spatial structure preservation

## 🔧 Key Improvements from Original

### Performance Optimizations
- **GPU Support**: Automatic CUDA detection and utilization
- **Data Loading**: Optimized data loaders with multiple workers
- **Model Architecture**: Improved networks with batch normalization and proper initialization
- **Training**: Advanced optimizers (AdamW), learning rate scheduling, early stopping

### Code Quality
- **Modular Design**: Separated concerns into different modules
- **Configuration Management**: Centralized hyperparameter management
- **Error Handling**: Comprehensive try-catch blocks
- **Documentation**: Detailed docstrings and comments
- **Type Safety**: Clear parameter types and return values

### Features Added
- **Data Augmentation**: Random rotation and translation for better generalization
- **Validation Split**: Proper train/validation/test split
- **Comprehensive Evaluation**: Classification reports, confusion matrices, training plots
- **Model Persistence**: Save/load functionality with metadata
- **Progress Tracking**: Real-time training progress with tqdm
- **Visualization**: Sample images, predictions, VAE generations, interpolations

### Removed Issues
- **Hardcoded Paths**: Now uses torchvision datasets with automatic download
- **Poor Structure**: Organized into logical modules
- **Limited Evaluation**: Added comprehensive metrics and visualizations
- **No GPU Support**: Full CUDA support
- **Manual Data Loading**: Replaced with robust torchvision datasets

## 📈 Expected Results

### Classification Performance
- **CNN Model**: ~99%+ accuracy on MNIST test set
- **MLP Model**: ~98%+ accuracy on MNIST test set
- **Training Time**: ~2-5 minutes on GPU, ~10-20 minutes on CPU

### VAE Performance
- **Reconstruction**: High-quality digit reconstructions
- **Generation**: Diverse and realistic digit samples
- **Latent Space**: Smooth interpolations between digits

## 🔍 Output Files

After training, you'll find the following in the `results/` directory:
- `training_history_*.png`: Training/validation loss and accuracy plots
- `confusion_matrix.png`: Test set confusion matrix
- `predictions.png`: Sample predictions with correct/incorrect labels
- `vae_samples_*.png`: Generated VAE samples
- `vae_interpolation.png`: Latent space interpolations
- `vae_reconstruction.png`: Original vs reconstructed images

Trained models are saved in the `models/` directory with metadata.

## 🚀 Advanced Features

### Custom Model Usage
```python
from models import ConvNet, ImprovedVAE
from train_utils import train_classifier
from data_utils import get_data_loaders

# Load data
train_loader, val_loader, test_loader = get_data_loaders()

# Create and train model
model = ConvNet(num_classes=10)
trained_model, history = train_classifier(model, train_loader, val_loader)
```

### Hyperparameter Tuning
Easily experiment with different hyperparameters by modifying the configuration or using command-line arguments.

### Model Comparison
Train multiple models and compare their performance using the comprehensive evaluation metrics.

## 🤝 Contributing

This project demonstrates best practices for deep learning projects:
- Clean, modular code structure
- Comprehensive documentation
- Proper error handling
- Configurable hyperparameters
- Extensive evaluation metrics

## 📝 License

This project is provided for educational purposes. Feel free to use and modify for learning and research.

## 🙏 Acknowledgments

- Original MNIST dataset by Yann LeCun
- PyTorch framework for deep learning implementation
- torchvision for dataset handling and transforms



