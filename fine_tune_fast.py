"""
Fast fine-tuning script with optimized parameters for speed.
"""
import os
import argparse
import json
import random
import sys
from dataset_processor import LibriSpeechProcessor
from fine_tuning_client import FineTuningClient
from verify_dataset import verify_dataset
from transformers import WhisperProcessor

def main():
    parser = argparse.ArgumentParser(description="Fast fine-tune a speech recognition model using LibriSpeech")
    parser.add_argument('--dataset_path', type=str, default='/Users/abhangsudhirpawar/Documents/Akai/LibriSpeech/train-clean-100',
                        help='Path to the LibriSpeech train-clean-100 dataset')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to use for fine-tuning')
    parser.add_argument('--model_path', type=str, default='./whisper-small',
                        help='Path to the local pre-trained model directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size (increased for speed)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for fine-tuning (slightly higher)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--min_duration', type=float, default=2.0,
                        help='Minimum audio duration in seconds (increased for efficiency)')
    parser.add_argument('--max_duration', type=float, default=10.0,
                        help='Maximum audio duration in seconds (reduced for speed)')
    parser.add_argument('--output_dir', type=str, default='./whisper-finetuned',
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--dry_run', action='store_true',
                        help='Process the dataset but do not start training')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify the dataset path first
    print(f"Verifying LibriSpeech dataset at: {args.dataset_path}")
    if not verify_dataset(args.dataset_path):
        print("Dataset verification failed. Please check the path and try again.")
        print("Use the verify_dataset.py script for more detailed diagnostics:")
        print(f"python verify_dataset.py --dataset_path \"{args.dataset_path}\"")
        sys.exit(1)
    
    # Process LibriSpeech dataset
    print(f"Processing LibriSpeech dataset from: {args.dataset_path}")
    processor = LibriSpeechProcessor(args.dataset_path)
    samples = processor.prepare_dataset(
        max_samples=args.max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    # Split data into training and validation sets
    random.shuffle(samples)
    validation_size = int(len(samples) * args.validation_split)
    training_data = samples[validation_size:]
    validation_data = samples[:validation_size]
    
    print(f"Training samples: {len(training_data)}")
    print(f"Validation samples: {len(validation_data)}")
    
    # Load WhisperProcessor for preprocessing
    whisper_processor = WhisperProcessor.from_pretrained(args.model_path)
    
    # Preprocess data for training
    print("Preprocessing training data...")
    preprocessed_training = processor.preprocess_for_training(whisper_processor)
    
    # Preprocess validation data if available
    if validation_data:
        print("Preprocessing validation data...")
        # Create temporary processor with validation data
        val_processor = LibriSpeechProcessor(args.dataset_path)
        val_processor.samples = validation_data
        preprocessed_validation = val_processor.preprocess_for_training(whisper_processor)
    else:
        preprocessed_validation = None
    
    # Initialize the fine-tuning client
    client = FineTuningClient(model_path=args.model_path)
    
    if not args.dry_run:
        print("Starting optimized fine-tuning...")
        print("Optimizations enabled:")
        print("- MPS (Apple Silicon GPU) acceleration")
        print("- Gradient checkpointing for memory efficiency") 
        print("- Increased batch size for speed")
        print("- Parallel data loading")
        print("- Optimized learning rate schedule")
        
        finetuned_model_path = client.fine_tune(
            preprocessed_samples=preprocessed_training,
            validation_samples=preprocessed_validation,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,  # Reduced for speed
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            warmup_steps=50  # Reduced warmup for speed
        )
        
        print(f"Fast fine-tuning completed successfully!")
        print(f"Fine-tuned model saved to: {finetuned_model_path}")
        
        # Quick test of the model
        print("\nTesting the fine-tuned model...")
        try:
            from transformers import WhisperForConditionalGeneration
            test_model = WhisperForConditionalGeneration.from_pretrained(finetuned_model_path)
            print(f"✓ Model loaded successfully from {finetuned_model_path}")
            print(f"✓ Model parameters: {test_model.num_parameters():,}")
        except Exception as e:
            print(f"Warning: Could not test model: {e}")
            
    else:
        print("Dry run completed. No fine-tuning was started.")
        print(f"Dataset processed: {len(training_data)} training samples, {len(validation_data)} validation samples.")
        print("To start the actual fine-tuning, run without the --dry_run flag.")
        print(f"Estimated training time with optimizations: ~{len(training_data) // args.batch_size * args.epochs * 30} seconds")

if __name__ == "__main__":
    main()
