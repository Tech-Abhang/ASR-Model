#!/usr/bin/env python3
"""
Test script to validate the local fine-tuning setup.
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
        print("✓ Transformers imports successful")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch available (version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from dataset_processor import LibriSpeechProcessor
        print("✓ Dataset processor import successful")
    except ImportError as e:
        print(f"✗ Dataset processor import failed: {e}")
        return False
    
    try:
        from fine_tuning_client import FineTuningClient
        print("✓ Fine-tuning client import successful")
    except ImportError as e:
        print(f"✗ Fine-tuning client import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that the local model can be loaded."""
    print("\nTesting model loading...")
    
    model_path = "./whisper-small"
    if not os.path.exists(model_path):
        print(f"✗ Model directory not found: {model_path}")
        print("Please ensure you have downloaded the model using downloadModel.py")
        return False
    
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        print("✓ Model and processor loaded successfully")
        print(f"  Model parameters: {model.num_parameters():,}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_fine_tuning_client():
    """Test the fine-tuning client initialization."""
    print("\nTesting fine-tuning client...")
    
    try:
        from fine_tuning_client import FineTuningClient
        
        client = FineTuningClient(model_path="./whisper-small")
        print("✓ Fine-tuning client initialized successfully")
        
        # Test training arguments creation
        training_args = client.prepare_training_arguments(
            output_dir="./test-output",
            per_device_train_batch_size=1,
            num_train_epochs=1
        )
        print("✓ Training arguments created successfully")
        return True
    except Exception as e:
        print(f"✗ Fine-tuning client test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running local fine-tuning setup validation...\n")
    
    tests = [
        test_imports,
        test_model_loading,
        test_fine_tuning_client
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Your setup is ready for local fine-tuning.")
        print("\nNext steps:")
        print("1. Run: python fine_tune.py --dry_run (to test dataset processing)")
        print("2. Run: python fine_tune.py (to start actual fine-tuning)")
    else:
        print("✗ Some tests failed. Please resolve the issues above.")
        print("Make sure you have:")
        print("1. Installed requirements: pip install -r requirements.txt")
        print("2. Downloaded the model: python downloadModel.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
