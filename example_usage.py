#!/usr/bin/env python3
"""
Example usage of the optimized MNIST recognition modules
"""
import torch
import matplotlib.pyplot as plt

# Import optimized modules
from config import Config
from data_utils import get_data_loaders, visualize_samples, visualize_predictions
from models import ConvNet, ImprovedVAE
from train_utils import train_classifier, train_vae, evaluate_classifier, save_model

def main():
    print("MNIST Recognition - Example Usage")
    print("=" * 50)
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load data with proper splits
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.NN_BATCH_SIZE,
        val_split=0.1
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some samples
    print("\n2. Visualizing sample data...")
    visualize_samples(train_loader, num_samples=8, title="Training Data Samples")
    
    # Example 1: Train a classification model
    print("\n3. Training a CNN classifier...")
    classifier = ConvNet(num_classes=10)
    
    # Train for fewer epochs in this example
    trained_classifier, classifier_history = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Reduced for example
        device=device
    )
    
    # Evaluate the classifier
    print("\n4. Evaluating classifier...")
    eval_results = evaluate_classifier(
        model=trained_classifier,
        test_loader=test_loader,
        device=device
    )
    
    print(f"Classification Accuracy: {eval_results['accuracy']:.2f}%")
    
    # Visualize predictions
    print("\n5. Visualizing predictions...")
    visualize_predictions(
        model=trained_classifier,
        data_loader=test_loader,
        num_samples=8,
        device=device
    )
    
    # Save the classifier
    print("\n6. Saving classifier...")
    save_model(
        model=trained_classifier,
        model_name='example_classifier',
        history=classifier_history
    )
    
    # Example 2: Train a VAE model
    print("\n7. Training a VAE...")
    vae = ImprovedVAE(
        input_dim=Config.VAE_INPUT_DIM,
        hidden_dim=Config.VAE_HIDDEN_DIM,
        latent_dim=Config.VAE_LATENT_DIM,
        beta=Config.VAE_BETA
    )
    
    # Train for fewer epochs in this example
    trained_vae, vae_history = train_vae(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Reduced for example
        device=device
    )
    
    # Generate samples with VAE
    print("\n8. Generating VAE samples...")
    generated_samples = trained_vae.generate(num_samples=16, device=device)
    
    # Visualize generated samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('VAE Generated Samples')
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        img = generated_samples[i].cpu().squeeze().numpy()
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.RESULTS_DIR}/example_vae_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save the VAE
    print("\n9. Saving VAE...")
    save_model(
        model=trained_vae,
        model_name='example_vae',
        history=vae_history
    )
    
    # Demonstrate model loading
    print("\n10. Demonstrating model loading...")
    saved_classifier_path = f'{Config.MODELS_DIR}/example_classifier.pth'
    
    if torch.cuda.is_available():
        checkpoint = torch.load(saved_classifier_path)
    else:
        checkpoint = torch.load(saved_classifier_path, map_location='cpu')
    
    # Create new model and load weights
    loaded_classifier = ConvNet(num_classes=10)
    loaded_classifier.load_state_dict(checkpoint['model_state_dict'])
    loaded_classifier.to(device)
    
    print(f"Successfully loaded model: {checkpoint['model_class']}")
    print(f"Model was saved with config: {checkpoint['config']}")
    
    # Test the loaded model
    loaded_classifier.eval()
    with torch.no_grad():
        # Get a test batch
        test_data, test_labels = next(iter(test_loader))
        test_data = test_data[:8].to(device)
        test_labels = test_labels[:8]
        
        # Make predictions
        outputs = loaded_classifier(test_data)
        _, predicted = torch.max(outputs, 1)
        
        # Check accuracy
        correct = (predicted.cpu() == test_labels).sum().item()
        print(f"Loaded model accuracy on test batch: {correct}/{len(test_labels)} = {100*correct/len(test_labels):.1f}%")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("Check the results/ directory for visualizations")
    print("Check the models/ directory for saved models")
    print("=" * 50)

if __name__ == '__main__':
    # Set matplotlib backend for better compatibility
    import matplotlib
    matplotlib.use('Agg')
    
    main()