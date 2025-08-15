"""
Configuration file for MNIST digit recognition project
"""
import torch
import os

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths (will be updated to use torchvision datasets)
    DATA_DIR = './data'
    RESULTS_DIR = './results'
    MODELS_DIR = './models'
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Training hyperparameters for basic neural network
    NN_BATCH_SIZE = 128
    NN_LEARNING_RATE = 0.001
    NN_EPOCHS = 20
    NN_MOMENTUM = 0.9
    NN_HIDDEN_DIMS = [512, 256, 128]
    
    # Training hyperparameters for VAE
    VAE_BATCH_SIZE = 128
    VAE_LEARNING_RATE = 1e-3
    VAE_EPOCHS = 50
    VAE_INPUT_DIM = 784
    VAE_HIDDEN_DIM = 512
    VAE_LATENT_DIM = 32
    VAE_BETA = 1.0  # Beta for beta-VAE
    
    # Data preprocessing
    NORMALIZE_MEAN = 0.1307  # MNIST mean
    NORMALIZE_STD = 0.3081   # MNIST std
    
    # Evaluation
    TEST_BATCH_SIZE = 1000
    
    # Visualization
    SAMPLE_SIZE = 16
    SAVE_SAMPLES = True
    SAVE_MODEL = True
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-4