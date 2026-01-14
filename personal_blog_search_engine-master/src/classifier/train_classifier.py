import os
import argparse
from classifier import BlogClassifier
import logging
import torch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the blog classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing crawled data')
    parser.add_argument('--model_path', type=str, default='models/blog_classifier', help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for tokenization')
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Initialize and train the classifier
    classifier = BlogClassifier(model_path=args.model_path)
    
    # Log device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Training classifier on data from {args.data_dir}...")
    print(f"Using batch size: {args.batch_size}, epochs: {args.epochs}, max_length: {args.max_length}")
    
    try:
        accuracy = classifier.train(
            data_dir=args.data_dir,
            test_size=args.test_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_length=args.max_length
        )
        
        print(f"Training complete. Best model accuracy: {accuracy:.4f}")
        print(f"Model saved to {os.path.abspath(args.model_path)}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
        classifier.model.save_pretrained(args.model_path)
        classifier.tokenizer.save_pretrained(args.model_path)
        print("Model state saved.")
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 