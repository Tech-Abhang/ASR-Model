# Akai: Speech Recognition Fine-Tuning Platform

A comprehensive toolkit for processing audio data, fine-tuning speech recognition models, and evaluating their performance using the LibriSpeech dataset and Hugging Face's speech models.

## Project Overview

This project provides end-to-end utilities for:
1. Processing and validating LibriSpeech dataset
2. Fine-tuning Whisper speech recognition models 
3. Evaluating model performance using WER (Word Error Rate)
4. Visualizing results and generating reports

It's optimized for Mac M-series chips without requiring CUDA acceleration.

## Project Structure

### Core Components

```
├── dataset_processor.py      # LibriSpeech dataset processing utilities
├── fine_tune.py              # Main fine-tuning script
├── fine_tuning_client.py     # Hugging Face API client for fine-tuning
├── simple_evaluate.py        # Evaluation script comparing model performance
├── inference.py              # Batch inference for testing models


### Directories

```
├── LibriSpeech/              # Training dataset (train-clean-100)
├── LibriSpeech_test/         # Test dataset (test-clean)
├── output/                   # Output directory for results and reports
├── whisper-small/            # Base whisper-small model
├── whisper-finetuned/        # Fine-tuned model
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- pip/pip3
- Hugging Face account (for API access)
- LibriSpeech dataset

### 2. Installation

1. Clone the repository:

```bash
git clone https://your-repository-url/Akai.git
cd Akai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Hugging Face API token:

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

### 3. Prepare Dataset

1. Download LibriSpeech datasets:

```bash
# Download train-clean-100 subset
mkdir -p LibriSpeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C LibriSpeech

# Download test-clean subset
mkdir -p LibriSpeech_test
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz -C LibriSpeech_test
```

2. Verify the dataset structure:

```bash
python verify_dataset.py --dataset_path ./LibriSpeech/train-clean-100
python verify_dataset.py --dataset_path ./LibriSpeech_test/test-clean
```

### 4. Download Base Model

Download the pre-trained base model:

```bash
python downloadModel.py --model_name "openai/whisper-small" --output_dir "./whisper-small"
```

## Complete Workflow Guide

#### Verify Dataset Structure

```bash
python verify_dataset.py --dataset_path ./LibriSpeech/train-clean-100
```

#### Analyze Dataset Statistics

```bash
python visualize_dataset.py --dataset_path ./LibriSpeech/train-clean-100 --output_dir ./output
```

### Fine-Tuning

#### Dry Run (Test without API submission)

```bash
python fine_tune.py \
  --dry_run \
  --dataset_path ./LibriSpeech/train-clean-100 \
  --max_samples 200 \
  --output_dir ./output
```

#### Start Fine-Tuning Job

```bash
python fine_tune.py \
  --dataset_path ./LibriSpeech/train-clean-100 \
  --max_samples 500 \
  --model_path ./whisper-small \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --output_dir ./whisper-finetuned
```
or

(shown better results)
```bash
python fine_tune.py \
  --output_dir ./whisper-finetuned \
  --max_samples 200 \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --min_duration 1.0 \
  --max_duration 15.0
```

```bash
cp ./whisper-small/preprocessor_config.json ./whisper-finetuned-new/
```

### Model Evaluation

#### Compare Base Model and Fine-Tuned Model

```bash
python simple_evaluate.py \
  --base_model ./whisper-small \
  --finetuned_model ./whisper-finetuned \
  --test_dataset ./LibriSpeech_test/test-clean \
  --max_samples 50 \
  --output_dir ./output
```

### Inference and Transcription

#### Batch Inference

```bash
python inference.py \
  --model ./whisper-finetuned \
  --test_dataset ./LibriSpeech_test/test-clean \
  --max_samples 20 \
  --output_dir ./output
```

#### Single File Transcription


## Advanced Usage

### Understanding Word Error Rate (WER) Calculation

The project uses the Levenshtein distance algorithm to calculate WER between reference and hypothesis transcriptions:

1. Calculate Edit Distance: Count minimum insertions, deletions, and substitutions to transform hypothesis to reference
2. Normalize: Divide edit distance by the number of words in the reference

WER = (Insertions + Deletions + Substitutions) / Number of words in reference

Lower WER indicates better model performance.

### Code Architecture

#### Key Classes and Functions

1. **LibriSpeechProcessor**: Processes and loads LibriSpeech dataset
   - Handles audio files, transcriptions, and dataset preparation
   - Creates sample structures with audio path, text, and duration

2. **InferenceModelClient**: Client for model inference
   - Abstracts away model loading and inference details
   - Supports both local models and Hugging Face hosted models

3. **calculate_wer**: Implements Word Error Rate calculation
   - Uses dynamic programming for Levenshtein distance
   - Provides normalized error rates for model comparison

#### Data Flow

1. Dataset Processing:
   ```
   LibriSpeechProcessor → prepare_dataset() → samples
   ```

2. Model Inference:
   ```
   audio_file → InferenceModelClient → transcribe_audio() → transcript
   ```

3. Evaluation:
   ```
   reference + transcript → calculate_wer() → WER score
   ```

### Customizing the Fine-Tuning Process

For advanced settings and fine-tuning parameters:

```bash
python fine_tune.py \
  --dataset_path ./LibriSpeech/train-clean-100 \
  --model_name openai/whisper-small \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --max_samples 1000 \
  --output_dir ./output \
  --push_to_hub \
  --hub_model_id your-username/whisper-small-finetuned
```

## Troubleshooting

### Common Issues

1. **Missing Dataset Files**:
   - Verify dataset structure with `verify_dataset.py`
   - Ensure correct paths are provided to scripts

2. **API Authentication Errors**:
   - Check if HUGGINGFACE_TOKEN environment variable is set
   - Verify token has write permissions

3. **Memory Issues**:
   - Reduce batch size or max_samples
   - Use model with fewer parameters (e.g., whisper-tiny)

## License

This project uses the LibriSpeech dataset, which is licensed under CC BY 4.0.

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- Hugging Face for model hosting and fine-tuning API
- LibriSpeech dataset creators and contributors
