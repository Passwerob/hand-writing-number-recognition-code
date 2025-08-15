#!/usr/bin/env python3
"""
Optimized MNIST Classification Training Script
"""
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_utils import get_data_loaders, visualize_samples, visualize_predictions
from models import ImprovedNet, ConvNet
from train_utils import train_classifier, evaluate_classifier, save_model

def main():
    parser = argparse.ArgumentParser(description='Train MNIST Classifier')
    parser.add_argument('--model', type=str, choices=['mlp', 'conv'], default='conv',
                       help='Model architecture to use (default: conv)')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {Config.NN_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {Config.NN_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {Config.NN_LEARNING_RATE})')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip final evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Show sample visualizations')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.epochs is not None:
        Config.NN_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.NN_BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.NN_LEARNING_RATE = args.lr
    
    print("=" * 50)
    print("MNIST Classification Training")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.NN_EPOCHS}")
    print(f"Batch Size: {Config.NN_BATCH_SIZE}")
    print(f"Learning Rate: {Config.NN_LEARNING_RATE}")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.NN_BATCH_SIZE, 
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
    print(f"\nCreating {args.model} model...")
    if args.model == 'mlp':
        model = ImprovedNet(
            input_size=784,
            hidden_dims=Config.NN_HIDDEN_DIMS,
            num_classes=10,
            dropout_rate=0.3
        )
    elif args.model == 'conv':
        model = ConvNet(num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Train model
    print(f"\nStarting training...")
    try:
        trained_model, history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=Config.NN_EPOCHS,
            device=Config.DEVICE
        )
        print("Training completed successfully!")
        
        # Plot training history
        print("\nPlotting training history...")
        history.plot_history(save_path=f'{Config.RESULTS_DIR}/training_history_{args.model}.png')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return
    
    # Save model
    if not args.no_save and Config.SAVE_MODEL:
        print("\nSaving model...")
        model_name = f'mnist_classifier_{args.model}'
        save_path = save_model(
            model=trained_model,
            model_name=model_name,
            history=history
        )
        print(f"Model saved to: {save_path}")
    
    # Evaluate model
    if not args.no_eval:
        print("\nEvaluating model on test set...")
        try:
            eval_results = evaluate_classifier(
                model=trained_model,
                test_loader=test_loader,
                device=Config.DEVICE
            )
            
            print(f"\nFinal Test Results:")
            print(f"Accuracy: {eval_results['accuracy']:.2f}%")
            print(f"Loss: {eval_results['loss']:.4f}")
            
            # Show sample predictions
            if args.visualize:
                print("\nVisualizing predictions...")
                visualize_predictions(
                    model=trained_model,
                    data_loader=test_loader,
                    device=Config.DEVICE
                )
            
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == '__main__':
    # Set matplotlib backend for headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    # Run training
    main()