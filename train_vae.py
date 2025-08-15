#!/usr/bin/env python3
"""
Optimized MNIST VAE Training Script
"""
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_utils import get_data_loaders, visualize_samples
from models import ImprovedVAE, ConvVAE
from train_utils import train_vae, visualize_vae_samples, save_model

def main():
    parser = argparse.ArgumentParser(description='Train MNIST VAE')
    parser.add_argument('--model', type=str, choices=['mlp', 'conv'], default='mlp',
                       help='VAE architecture to use (default: mlp)')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {Config.VAE_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {Config.VAE_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {Config.VAE_LEARNING_RATE})')
    parser.add_argument('--latent-dim', type=int, default=None,
                       help=f'Latent dimension (default: {Config.VAE_LATENT_DIM})')
    parser.add_argument('--beta', type=float, default=None,
                       help=f'Beta for beta-VAE (default: {Config.VAE_BETA})')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--visualize', action='store_true',
                       help='Show sample visualizations')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.epochs is not None:
        Config.VAE_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.VAE_BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.VAE_LEARNING_RATE = args.lr
    if args.latent_dim is not None:
        Config.VAE_LATENT_DIM = args.latent_dim
    if args.beta is not None:
        Config.VAE_BETA = args.beta
    
    print("=" * 50)
    print("MNIST VAE Training")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.VAE_EPOCHS}")
    print(f"Batch Size: {Config.VAE_BATCH_SIZE}")
    print(f"Learning Rate: {Config.VAE_LEARNING_RATE}")
    print(f"Latent Dimension: {Config.VAE_LATENT_DIM}")
    print(f"Beta: {Config.VAE_BETA}")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.VAE_BATCH_SIZE, 
        val_split=0.1
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize sample data
    if args.visualize:
        print("\nVisualizing sample data...")
        visualize_samples(train_loader, title="Training Data Samples")
    
    # Create model
    print(f"\nCreating {args.model} VAE model...")
    if args.model == 'mlp':
        model = ImprovedVAE(
            input_dim=Config.VAE_INPUT_DIM,
            hidden_dim=Config.VAE_HIDDEN_DIM,
            latent_dim=Config.VAE_LATENT_DIM,
            beta=Config.VAE_BETA
        )
    elif args.model == 'conv':
        model = ConvVAE(
            latent_dim=Config.VAE_LATENT_DIM,
            beta=Config.VAE_BETA
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Train model
    print(f"\nStarting training...")
    try:
        trained_model, history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=Config.VAE_EPOCHS,
            device=Config.DEVICE
        )
        print("Training completed successfully!")
        
        # Plot training history
        print("\nPlotting training history...")
        history.plot_history(save_path=f'{Config.RESULTS_DIR}/vae_training_history_{args.model}.png')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return
    
    # Generate and visualize samples
    print("\nGenerating samples...")
    try:
        visualize_vae_samples(
            model=trained_model,
            epoch='final',
            device=Config.DEVICE,
            num_samples=Config.SAMPLE_SIZE
        )
        
        # Generate interpolation samples
        print("Generating interpolation samples...")
        generate_interpolation_samples(trained_model, Config.DEVICE)
        
    except Exception as e:
        print(f"Sample generation failed with error: {e}")
    
    # Save model
    if not args.no_save and Config.SAVE_MODEL:
        print("\nSaving model...")
        model_name = f'mnist_vae_{args.model}'
        save_path = save_model(
            model=trained_model,
            model_name=model_name,
            history=history
        )
        print(f"Model saved to: {save_path}")
    
    # Reconstruction evaluation
    print("\nEvaluating reconstruction quality...")
    try:
        evaluate_reconstruction(trained_model, test_loader, Config.DEVICE)
    except Exception as e:
        print(f"Reconstruction evaluation failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("VAE Training completed!")
    print("=" * 50)

def generate_interpolation_samples(model, device, num_interpolations=8):
    """Generate interpolation samples between two random points in latent space"""
    model.eval()
    
    with torch.no_grad():
        # Sample two random points in latent space
        z1 = torch.randn(1, Config.VAE_LATENT_DIM).to(device)
        z2 = torch.randn(1, Config.VAE_LATENT_DIM).to(device)
        
        # Create interpolation
        alphas = torch.linspace(0, 1, num_interpolations).to(device)
        interpolated_samples = []
        
        for alpha in alphas:
            z_interp = alpha * z2 + (1 - alpha) * z1
            if hasattr(model, 'decode'):
                sample = model.decode(z_interp)
            else:
                sample = model.decoder(z_interp)
            interpolated_samples.append(sample.cpu())
        
        # Visualize interpolation
        fig, axes = plt.subplots(1, num_interpolations, figsize=(16, 2))
        fig.suptitle('Latent Space Interpolation')
        
        for i, sample in enumerate(interpolated_samples):
            img = sample.view(28, 28).numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'α={alphas[i]:.2f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{Config.RESULTS_DIR}/vae_interpolation.png', dpi=150, bbox_inches='tight')
        plt.show()

def evaluate_reconstruction(model, test_loader, device, num_samples=8):
    """Evaluate reconstruction quality"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    original_images, _ = next(data_iter)
    original_images = original_images[:num_samples].to(device)
    
    with torch.no_grad():
        # Get reconstructions
        reconstructed, _, _ = model(original_images)
        
        # Move to CPU for visualization
        original_images = original_images.cpu()
        reconstructed = reconstructed.cpu()
        
        # Visualize original vs reconstructed
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
        fig.suptitle('Original vs Reconstructed Images')
        
        for i in range(num_samples):
            # Original image
            orig_img = original_images[i].squeeze()
            # Denormalize for visualization
            orig_img = orig_img * Config.NORMALIZE_STD + Config.NORMALIZE_MEAN
            orig_img = torch.clamp(orig_img, 0, 1)
            
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            recon_img = reconstructed[i].view(28, 28)
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{Config.RESULTS_DIR}/vae_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate reconstruction loss
        recon_loss = torch.nn.functional.binary_cross_entropy(
            reconstructed, original_images.view(-1, 784), reduction='mean'
        )
        print(f"Average reconstruction loss: {recon_loss.item():.4f}")

if __name__ == '__main__':
    # Set matplotlib backend for headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    # Run training
    main()