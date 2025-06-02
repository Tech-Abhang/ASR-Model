#!/usr/bin/env python3
"""
Simple evaluation script that compares WER between original and fine-tuned models.

This script evaluates both whisper-small and whisper-finetuned models on the LibriSpeech test-clean dataset
and compares their Word Error Rate (WER) to measure the improvement from fine-tuning.

Usage Examples:
    # Basic comparison with default settings (50 samples)
    python simple_evaluate.py
    
    # Test with more samples for robust evaluation
    python simple_evaluate.py --max_samples 100
    
    # Use custom model paths
    python simple_evaluate.py --base_model ./my-whisper-small --finetuned_model ./my-whisper-finetuned
    
    # Use different test dataset
    python simple_evaluate.py --test_dataset /path/to/other/test/dataset --max_samples 30

Results:
    - JSON file with detailed results saved to output directory
    - PNG visualization comparing WER between models
    - Console output showing WER comparison and improvement percentage
"""
import os
import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from dataset_processor import LibriSpeechProcessor

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis strings
    
    Args:
        reference (str): Reference text
        hypothesis (str): Hypothesis text
        
    Returns:
        float: Word Error Rate
    """
    # Tokenize
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Calculate Levenshtein distance (simplified implementation)
    m, n = len(ref_words), len(hyp_words)
    
    # Special case: empty strings
    if m == 0:
        return 1.0 if n > 0 else 0.0
    
    # Initialize cost matrix
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill first row and column
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    
    # Fill cost matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]  # No operation needed
            else:
                # Take minimum of substitution, insertion, deletion
                d[i][j] = min(
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion  
                    d[i-1][j-1] + 1   # substitution
                )
    
    # WER is the edit distance divided by the number of words in reference
    return float(d[m][n]) / m if m > 0 else 0.0

def test_model(model_path, test_dataset_path, output_dir, max_samples=100):
    """
    Test a model on LibriSpeech test data using local transformers
    
    Args:
        model_path (str): Path to local model directory
        test_dataset_path (str): Path to LibriSpeech test-clean
        output_dir (str): Directory to save results
        max_samples (int): Maximum number of samples to test
        
    Returns:
        dict: Results of the inference
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
        # Load model and processor
    print(f"Loading model from {model_path}...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        # Fix generation config issues by resetting forced_decoder_ids
        model.generation_config.forced_decoder_ids = None
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Process LibriSpeech test data
    processor_lib = LibriSpeechProcessor(test_dataset_path)
    samples = processor_lib.prepare_dataset(max_samples=max_samples)
    
    if not samples:
        print(f"Error: No samples found in {test_dataset_path}")
        return None
    
    # Get model name for display
    model_name = os.path.basename(model_path)
    print(f"Testing model '{model_name}' on {len(samples)} samples from {test_dataset_path}")
    
    # Setup results tracking
    results = {
        "model": model_path,
        "model_name": model_name,
        "dataset": test_dataset_path,
        "samples": len(samples),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transcriptions": [],
        "metrics": {
            "wer_values": [],
            "avg_wer": 0.0,
            "processing_time": 0.0
        }
    }
    
    # Run inference on each sample
    processing_times = []
    for i, sample in enumerate(tqdm(samples, desc=f"Testing {model_name}")):
        audio_path = sample["audio_path"]
        reference = sample["text"]  # Use 'text' key instead of 'transcript'
        
        try:
            # Load and preprocess audio
            start_time = time.time()
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio with Whisper processor
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="en",
                    task="transcribe"
                )
                transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Calculate WER
            wer = calculate_wer(reference, transcript)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            transcript = ""
            wer = 1.0
            processing_time = 0.0
            processing_times.append(processing_time)
        
        # Store the results
        results["transcriptions"].append({
            "audio": os.path.basename(audio_path),
            "reference": reference,
            "transcript": transcript,
            "wer": wer,
            "processing_time": processing_time
        })
        
        results["metrics"]["wer_values"].append(wer)
        
        # Print progress every 10 samples or for the first one
        if i == 0 or (i+1) % 10 == 0:
            print(f"\nSample {i+1}:")
            print(f"Reference: {reference}")
            print(f"Transcript: {transcript}")
            print(f"WER: {wer:.4f}")
    
    # Calculate average metrics
    avg_wer = np.mean(results["metrics"]["wer_values"]) if results["metrics"]["wer_values"] else 1.0
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    results["metrics"]["avg_wer"] = float(avg_wer)
    results["metrics"]["processing_time"] = float(avg_processing_time)
    
    # Save results to file
    output_filename = f"inference_results_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults for {model_name}:")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Processing time: {avg_processing_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    
    return results
def compare_wer(base_model_path, finetuned_model_path, test_dataset_path, max_samples=20, output_dir="./output"):
    """
    Run a simple comparison of WER between base model and fine-tuned model
    
    Args:
        base_model_path (str): Path to base model directory
        finetuned_model_path (str): Path to fine-tuned model directory
        test_dataset_path (str): Path to test dataset
        max_samples (int): Maximum number of samples to test
        output_dir (str): Directory to save results
    """
    print(f"\nComparing models on {max_samples} samples from {test_dataset_path}")
    print(f"Base model: {base_model_path}")
    print(f"Fine-tuned model: {finetuned_model_path}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate model paths
    if not os.path.exists(base_model_path):
        print(f"Error: Base model path does not exist: {base_model_path}")
        return
    
    if not os.path.exists(finetuned_model_path):
        print(f"Error: Fine-tuned model path does not exist: {finetuned_model_path}")
        return
    
    # Run inference on base model
    print("\n== Testing Base Model ==")
    base_results = test_model(
        model_path=base_model_path,
        test_dataset_path=test_dataset_path,
        output_dir=output_dir,
        max_samples=max_samples
    )
    
    # Run inference on fine-tuned model
    print("\n== Testing Fine-tuned Model ==")
    finetuned_results = test_model(
        model_path=finetuned_model_path,
        test_dataset_path=test_dataset_path,
        output_dir=output_dir,
        max_samples=max_samples
    )
    
    if not base_results or not finetuned_results:
        print("Error: Could not get results from one or both models")
        return
    
    # Extract model names for display
    base_model_name = os.path.basename(base_model_path)
    finetuned_model_name = os.path.basename(finetuned_model_path)
    
    # Calculate average WER values
    base_wer = base_results["metrics"]["avg_wer"]
    finetuned_wer = finetuned_results["metrics"]["avg_wer"]
    
    # Print comparison results
    print("\n======= WER Comparison Results =======")
    print(f"Base Model ({base_model_name}): {base_wer:.4f}")
    print(f"Fine-tuned Model ({finetuned_model_name}): {finetuned_wer:.4f}")
    
    # Calculate improvement
    if base_wer > 0:
        improvement = ((base_wer - finetuned_wer) / base_wer) * 100
        print(f"Improvement: {improvement:.2f}%")
        if improvement > 0:
            print("✅ Fine-tuning improved the model!")
        else:
            print("⚠️  Fine-tuning did not improve the model")
    else:
        improvement = 0.0
        print("Cannot calculate improvement percentage (base WER is zero)")
    
    # Save comparison results
    comparison_results = {
        "base_model": {
            "name": base_model_name,
            "path": base_model_path,
            "wer": base_wer
        },
        "finetuned_model": {
            "name": finetuned_model_name,
            "path": finetuned_model_path,
            "wer": finetuned_wer
        },
        "improvement_percent": float(improvement),
        "test_dataset": test_dataset_path,
        "samples_tested": max_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    comparison_file = os.path.join(output_dir, "simple_wer_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create a simple bar chart visualization
    create_wer_comparison_chart(
        base_model_name, 
        finetuned_model_name, 
        base_wer, 
        finetuned_wer, 
        output_dir
    )
    
    print(f"\nComparison results saved to: {comparison_file}")
    print(f"Visualization saved to: {os.path.join(output_dir, 'wer_comparison.png')}")

def create_wer_comparison_chart(base_name, finetuned_name, base_wer, finetuned_wer, output_dir):
    """
    Create a simple bar chart showing WER comparison
    
    Args:
        base_name (str): Name of base model
        finetuned_name (str): Name of fine-tuned model
        base_wer (float): Base model WER
        finetuned_wer (float): Fine-tuned model WER
        output_dir (str): Output directory
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    models = [base_name, finetuned_name]
    wer_values = [base_wer, finetuned_wer]
    colors = ['lightblue', 'lightgreen']
    
    # Create bars
    bars = plt.bar(models, wer_values, color=colors)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Word Error Rate (WER)')
    plt.title('WER Comparison: Base vs Fine-tuned Model')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    # Add a horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Calculate improvement percentage
    if base_wer > 0:
        improvement = ((base_wer - finetuned_wer) / base_wer) * 100
        plt.figtext(0.5, 0.01, f"Improvement: {improvement:.2f}%", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wer_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Simple WER comparison between whisper-small and whisper-finetuned models")
    parser.add_argument('--base_model', type=str,
                      default="/Users/abhangsudhirpawar/Documents/Akai/whisper-small",
                      help='Path to base model directory (default: ./whisper-small)')
    parser.add_argument('--finetuned_model', type=str,
                      default="/Users/abhangsudhirpawar/Documents/Akai/whisper-finetuned",
                      help='Path to fine-tuned model directory (default: ./whisper-finetuned)')
    parser.add_argument('--test_dataset', type=str,
                      default="/Users/abhangsudhirpawar/Documents/Akai/LibriSpeech_test/test-clean",
                      help='Path to LibriSpeech test dataset')
    parser.add_argument('--max_samples', type=int, default=50,
                      help='Maximum number of samples to test (default: 50)')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate test dataset path
    if not os.path.exists(args.test_dataset):
        print(f"Error: Test dataset path does not exist: {args.test_dataset}")
        print("Please ensure LibriSpeech_test/test-clean is available")
        return
    
    # Run comparison
    compare_wer(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        test_dataset_path=args.test_dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
