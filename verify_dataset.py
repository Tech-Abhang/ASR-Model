"""
Utility script to verify the LibriSpeech dataset and provide helpful guidance.
"""
import os
import argparse
from pathlib import Path

def verify_dataset(dataset_path):
    """
    Verify that the specified dataset path exists and has the expected structure.
    
    Args:
        dataset_path (str): Path to the LibriSpeech dataset
        
    Returns:
        bool: True if the dataset looks valid, False otherwise
    """
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    # Check if it's a directory
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        return False
    
    # Try to find speaker directories (numeric subdirectories)
    speaker_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()]
    
    if not speaker_dirs:
        print(f"Error: No valid speaker directories found in {dataset_path}")
        print("Speaker directories should be numeric (e.g., 103/, 1034/, etc.)")
        return False
    
    # Check for chapter directories within first speaker
    example_speaker_dir = os.path.join(dataset_path, speaker_dirs[0])
    chapter_dirs = [d for d in os.listdir(example_speaker_dir) 
                   if os.path.isdir(os.path.join(example_speaker_dir, d)) and d.isdigit()]
    
    if not chapter_dirs:
        print(f"Error: No valid chapter directories found in speaker directory {speaker_dirs[0]}")
        return False
    
    # Check for flac and txt files in chapter directory
    example_chapter_dir = os.path.join(example_speaker_dir, chapter_dirs[0])
    flac_files = [f for f in os.listdir(example_chapter_dir) if f.endswith('.flac')]
    txt_files = [f for f in os.listdir(example_chapter_dir) if f.endswith('.txt')]
    
    if not flac_files:
        print(f"Error: No .flac files found in chapter directory {chapter_dirs[0]}")
        return False
    
    if not txt_files:
        print(f"Error: No .txt files found in chapter directory {chapter_dirs[0]}")
        return False
    
    print(f"Dataset validation passed for {dataset_path}")
    print(f"Found {len(speaker_dirs)} speaker directories")
    print(f"Sample structure: {example_speaker_dir}/{chapter_dirs[0]}/")
    print(f"  - {len(flac_files)} .flac files")
    print(f"  - {len(txt_files)} .txt files")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify the LibriSpeech dataset structure")
    parser.add_argument('--dataset_path', type=str, 
                        default='/Users/abhangsudhirpawar/Documents/Akai/LibriSpeech/train-clean-100',
                        help='Path to the LibriSpeech train-clean-100 dataset')
    args = parser.parse_args()
    
    # Check for common dataset locations if the default path doesn't exist
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at the specified path: {dataset_path}")
        print("Checking common alternative locations...")
        
        alternatives = [
            os.path.join(str(Path.home()), "LibriSpeech/train-clean-100"),
            os.path.join(str(Path.home()), "Downloads/LibriSpeech/train-clean-100"),
            os.path.join(str(Path.home()), "Documents/LibriSpeech/train-clean-100"),
            "/data/LibriSpeech/train-clean-100",
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                print(f"Found dataset at: {alt_path}")
                dataset_path = alt_path
                break
    
    is_valid = verify_dataset(dataset_path)
    
    if is_valid:
        print("\nDataset verification successful!")
        print(f"You can use this path with the fine-tuning script:")
        print(f"python fine_tune.py --dataset_path \"{dataset_path}\" --dry_run")
    else:
        print("\nDataset verification failed!")
        print("Please make sure the LibriSpeech dataset is correctly downloaded and extracted.")
        print("The expected structure is:")
        print("train-clean-100/")
        print("  ├── <speaker_id>/")
        print("  │   ├── <chapter_id>/")
        print("  │   │   ├── <speaker_id>-<chapter_id>-<utterance_id>.flac")
        print("  │   │   └── <speaker_id>-<chapter_id>.txt")
        
        print("\nTo download the LibriSpeech train-clean-100 dataset:")
        print("curl -O https://www.openslr.org/resources/12/train-clean-100.tar.gz")
        print("tar -xzf train-clean-100.tar.gz")

if __name__ == "__main__":
    main()
