"""
Run inference on audio files using the fine-tuned model.
"""
import os
import argparse
import json
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
from huggingface_client import InferenceModelClient
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
                d[i][j] = min(d[i-1][j-1] + 1,  # Substitution
                             d[i][j-1] + 1,      # Insertion
                             d[i-1][j] + 1)      # Deletion
    
    # Return WER
    return d[m][n] / m

def test_model(model_path, test_dataset_path, output_dir, max_samples=100):
    """
    Test a model on LibriSpeech test data
    
    Args:
        model_path (str): Path to local model or HF model name
        test_dataset_path (str): Path to LibriSpeech test-clean
        output_dir (str): Directory to save results
        max_samples (int): Maximum number of samples to test
        
    Returns:
        dict: Results of the inference
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model is local or HF
    is_local = os.path.exists(model_path) and model_path.endswith('.json')
    client = InferenceModelClient(model_path=model_path if is_local else None)
    
    # Process LibriSpeech test data
    processor = LibriSpeechProcessor(test_dataset_path)
    samples = processor.prepare_dataset(max_samples=max_samples)
    
    if not samples:
        print(f"Error: No samples found in {test_dataset_path}")
        return None
    
    # Get model name for display
    model_name = model_path.split('/')[-1].replace('.json', '') if is_local else model_path
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
        reference = sample["text"]
        
        # Transcribe the audio
        start_time = time.time()
        transcript = client.transcribe_audio(audio_path)
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Calculate WER if we have a transcript
        if transcript:
            wer = calculate_wer(reference, transcript)
        else:
            wer = 1.0  # If transcription failed, WER is 100%
            
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

def main():
    parser = argparse.ArgumentParser(description="Run inference using fine-tuned model")
    parser.add_argument('--model', required=True,
                     help='Path to model JSON or Hugging Face model name')
    parser.add_argument('--test_dataset', type=str,
                     default="/Users/abhangsudhirpawar/Documents/Akai/LibriSpeech_test/test-clean",
                     help='Path to LibriSpeech test dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                     help='Directory to save output files')
    parser.add_argument('--max_samples', type=int, default=20,
                     help='Maximum number of samples to test')
    
    args = parser.parse_args()
    
    # Test the model
    test_model(
        model_path=args.model,
        test_dataset_path=args.test_dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
