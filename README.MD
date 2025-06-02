# Akai: Audio Processing with LibriSpeech

A tool for processing audio data from the LibriSpeech dataset using Hugging Face's speech recognition capabilities, with fine-tuning capabilities.

## Overview

This project provides utilities to work with the LibriSpeech dataset, perform speech recognition using Hugging Face's speech models, and fine-tune pre-trained speech recognition models through the Hugging Face API using the local LibriSpeech train-clean-100 dataset. It's optimized for Mac M3 chips without requiring CUDA.

## Files

- `complete_workflow.py`: Main script to run the entire fine-tuning and evaluation process
- `huggingface_client.py`: Client for interacting with Hugging Face's speech recognition models
- `dataset_processor.py`: Processes the LibriSpeech dataset for fine-tuning
- `fine_tuning_client.py`: Client for interacting with the Hugging Face API for fine-tuning
- `fine_tune.py`: Main script for fine-tuning
- `check_job_status.py`: Utility to check fine-tuning job status
- `verify_dataset.py`: Validates the LibriSpeech dataset structure 
- `simple_evaluate.py`: Compares model performance using WER metrics
- `transcribe.py`: Command-line tool to transcribe individual audio files
- `inference.py`: Script for batch inference on multiple audio files
- `visualize_dataset.py`: Creates visualizations of dataset statistics
- `create_base_model.py`: Creates base model configuration files
- `test.flac`: Sample audio file for testing

## Requirements

Dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Fine-tuning a Speech Recognition Model

This project includes functionality to fine-tune a pre-trained speech recognition model using your local LibriSpeech train-clean-100 dataset through the Hugging Face API. The implementation is optimized for Mac M3 with no CUDA requirements.

### Complete Workflow

To run the entire pipeline from fine-tuning to evaluation, use the complete workflow script:

```bash
python complete_workflow.py --dataset_path /path/to/LibriSpeech/train-clean-100 \
                           --test_dataset /path/to/LibriSpeech_test/test-clean \
                           --max_samples 100 \
                           --delete_existing
```

This script performs the following steps:
1. **Delete existing models** (optional)
2. **Fine-tune** a new model using the specified dataset
3. **Run evaluation** on the test dataset
4. **Compare models** to measure improvements
5. **Generate reports** with visualizations

### Individual Scripts

For more control over the process, you can also use the individual scripts:

#### Verifying Your Dataset

```bash
python verify_dataset.py --dataset_path /path/to/LibriSpeech/train-clean-100
```

#### Running a Dry Run

Process the dataset without sending it to the API:

```bash
python fine_tune.py --dry_run --max_samples 200
```

#### Visualizing Dataset Statistics

After a dry run, visualize the dataset statistics:

```bash
python visualize_dataset.py
```

#### Starting Fine-tuning

To fine-tune a model using the default parameters:

```bash
python fine_tune.py
```

You can customize the fine-tuning with various options:

```bash
python fine_tune.py \
  --dataset_path /path/to/LibriSpeech/train-clean-100 \
  --max_samples 200 \
  --model_name openai/whisper-small \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 5e-5
```

#### Checking Job Status

After starting a fine-tuning job, you can check its status:

```bash
python check_job_status.py --job_id <job_id> --model openai/whisper-small
```

To wait for the job to complete:

```bash
python check_job_status.py --job_id <job_id> --model openai/whisper-small --wait
```

## Simulating and Testing Your Model

If you don't have access to the Hugging Face API or want to test the workflow locally, you can use our simulation capabilities:

### Running Local Fine-tuning Simulation

```bash
python simulate_local.py --max_samples 200 --epochs 3
```

This creates a simulated fine-tuned model saved in `output/models/`.

### Testing Your Fine-tuned Model

Once you have a fine-tuned model (either real or simulated), you can test it on the LibriSpeech test-clean dataset:

```bash
python inference.py --model ./output/models/whisper-small-finetuned.json --test_dataset /path/to/LibriSpeech_test/test-clean --max_samples 20
```

For real Hugging Face models:

```bash
python inference.py --model openai/whisper-small --test_dataset /path/to/LibriSpeech_test/test-clean --max_samples 20
```

### Transcribing Single Audio Files

To transcribe a single audio file:

```bash
python huggingface_client.py --audio path/to/audio.flac --model ./output/models/whisper-small-finetuned.json --is_local
```

Or using a Hugging Face model:

```bash
python huggingface_client.py --audio path/to/audio.flac --model openai/whisper-small
```

## Dataset

The project uses the LibriSpeech dataset, which contains readings from audiobooks. The dataset is organized into folders by speaker IDs. The training data is from `train-clean-100` and the evaluation uses `test-clean`.


