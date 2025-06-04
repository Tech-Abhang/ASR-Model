# Akai: Speech Recognition Fine-Tuning Platform

A comprehensive toolkit for processing audio data, fine-tuning speech recognition models, and evaluating their performance using the LibriSpeech dataset and Hugging Face's speech models.

## Project Overview

This project provides end-to-end utilities for:
1. Processing and validating LibriSpeech dataset
2. Fine-tuning Whisper speech recognition models 
3. Evaluating model performance using WER (Word Error Rate)
4. Visualizing results and generating reports

It's optimized for Mac M-series chips without requiring CUDA acceleration.

## Dataset Information

### LibriSpeech Dataset Overview

[LibriSpeech](http://www.openslr.org/12/) is a large corpus of read English speech designed for speech recognition research. It's derived from audiobooks in the LibriVox project and contains approximately 1000 hours of speech data.

### Dataset Subsets Used

This project uses two specific subsets of LibriSpeech:

#### Training Data: `train-clean-100`
- **Size**: ~100 hours of clean speech
- **Speakers**: 251 speakers
- **Audio Files**: ~28,539 utterances
- **File Format**: 16kHz FLAC audio files
- **Quality**: High-quality recordings with minimal background noise
- **Purpose**: Used for fine-tuning the Whisper model

#### Test Data: `test-clean`
- **Size**: ~5.4 hours of clean speech
- **Speakers**: 40 speakers
- **Audio Files**: ~2,620 utterances
- **File Format**: 16kHz FLAC audio files
- **Quality**: High-quality recordings for evaluation
- **Purpose**: Used for model evaluation and WER calculation

### Dataset Structure

```
LibriSpeech/train-clean-100/
├── [speaker_id]/
│   ├── [chapter_id]/
│   │   ├── [speaker_id]-[chapter_id]-[utterance_id].flac
│   │   ├── [speaker_id]-[chapter_id]-[utterance_id].txt
│   │   └── ...
│   └── ...
└── ...

LibriSpeech_test/test-clean/
├── [speaker_id]/
│   ├── [chapter_id]/
│   │   ├── [speaker_id]-[chapter_id]-[utterance_id].flac
│   │   ├── [speaker_id]-[chapter_id]-[utterance_id].txt
│   │   └── ...
│   └── ...
└── ...
```

## Dataset Statistics

These statistics are visualized in `output/dataset_stats.png` after running dataset analysis scripts.

---

## Project Structure

### Core Components

```
├── dataset_processor.py      # LibriSpeech dataset processing utilities
├── fine_tune.py              # Main fine-tuning script
├── fine_tuning_client.py     # Hugging Face API client for fine-tuning
├── simple_evaluate.py        # Evaluation script comparing model performance
├── inference.py              # Batch inference for testing models
```

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

#### Download LibriSpeech Datasets

1. **Training Dataset (train-clean-100)**:
   ```bash
   # Create directory and download (~6.3 GB)
   mkdir -p LibriSpeech
   wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
   tar -xzf train-clean-100.tar.gz -C LibriSpeech
   ```

2. **Test Dataset (test-clean)**:
   ```bash
   # Create directory and download (~346 MB)
   mkdir -p LibriSpeech_test
   wget https://www.openslr.org/resources/12/test-clean.tar.gz
   tar -xzf test-clean.tar.gz -C LibriSpeech_test
   ```

#### Verify Dataset Structure

After downloading, verify the datasets are correctly structured:

```bash
# Verify training dataset
python verify_dataset.py --dataset_path ./LibriSpeech/train-clean-100

# Verify test dataset
python verify_dataset.py --dataset_path ./LibriSpeech_test/test-clean
```

**Expected Output:**
```
Dataset verification completed successfully!
✓ Found [X] speakers
✓ Found [X] audio files  
✓ Found [X] transcript files
✓ All audio files have corresponding transcripts
✓ Sample rate consistency verified (16000 Hz)
```

**4. Download Base Model**

Download the pre-trained base model:

```bash
python downloadModel.py --model_name "openai/whisper-small" --output_dir "./whisper-small"
```

## Complete Workflow Guide

#### Verify Dataset Structure

```bash
python verify_dataset.py --dataset_path ./LibriSpeech/train-clean-100
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
cp ./whisper-small/preprocessor_config.json ./whisper-finetuned/
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

## Running the API Server

### Start the Transcription Server

The project includes a FastAPI server for real-time audio transcription via HTTP API.

1. **Start the server:**
   ```bash
   python app.py
   ```
   
   The server will start on `http://localhost:8000` and load your fine-tuned model.

2. **Test with curl command:**
   ```bash
   curl -X POST "http://localhost:8000/transcribe" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_audio_file.wav"
   ```

3. **Use the test script:**
   ```bash
   ./test_api.sh test1.wav
   ```

### API Endpoints

- **POST `/transcribe`**: Upload an audio file (.wav format) and get transcription
  - **Input**: Audio file via multipart/form-data
  - **Output**: JSON response with transcription text
  - **Example response**: `{"transcription":"YOUR TRANSCRIBED TEXT HERE"}`

### Server Configuration

The server automatically:
- Loads your fine-tuned model from `./whisper-finetuned`
- Falls back to base Whisper model if fine-tuned model fails
- Handles temporary file management
- Provides error handling and logging

### Testing Audio Files

You can test the API with any `.wav` audio file:
```bash
# Using curl directly
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test1.wav"

# Using the provided test script
./test_api.sh test1.wav
./test_api.sh your_custom_audio.wav
```

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

## Output Files and Results

The `./output/` directory contains evaluation results and visualizations generated by the various scripts:

### Generated Files

| File | Description | Generated By |
|------|-------------|--------------|
| `latest_base_model_results.json` | Detailed evaluation results for base Whisper model | `inference.py` |
| `latest_finetuned_model_results.json` | Detailed evaluation results for fine-tuned model | `inference.py` |
| `simple_wer_comparison.json` | WER comparison between base and fine-tuned models | `simple_evaluate.py` |
| `wer_comparison.png` | Visual chart comparing WER performance | `simple_evaluate.py` |
| `dataset_stats.png` | Dataset statistics and duration distribution | Dataset analysis scripts |
| `dataset_manifest.json` | Sample dataset manifest with audio paths and transcripts | Dataset processing |

### Understanding Results

**Word Error Rate (WER)**: Lower values indicate better performance
- **0.0 - 0.1**: Excellent (0-10% error rate)
- **0.1 - 0.2**: Good (10-20% error rate)  
- **0.2 - 0.3**: Fair (20-30% error rate)
- **0.3+**: Poor (30%+ error rate)

**Sample Result Structure** (`simple_wer_comparison.json`):
```json
{
  "base_model": {
    "name": "whisper-small",
    "wer": 0.222
  },
  "finetuned_model": {
    "name": "whisper-finetuned", 
    "wer": 0.168
  },
  "improvement_percent": 24.69
}
```

## Troubleshooting

### Common Issues

1. **Dataset Issues**:
   - **Missing Dataset Files**: Verify dataset structure with `python verify_dataset.py --dataset_path ./LibriSpeech/train-clean-100`
   - **Incomplete Download**: Re-download datasets if verification fails
   - **Wrong Directory Structure**: Ensure datasets are extracted to correct paths:
     - Training: `./LibriSpeech/train-clean-100/`
     - Test: `./LibriSpeech_test/test-clean/`
   - **Audio Format Issues**: LibriSpeech uses FLAC format; ensure your system supports FLAC decoding
   - **Disk Space**: Training dataset requires ~6.3 GB, test dataset ~346 MB

2. **Dataset Processing Issues**:
   - **Slow Processing**: Large dataset processing can take time; use `--max_samples` to limit for testing
   - **Memory Issues During Processing**: Reduce batch size or process in smaller chunks
   - **Audio Loading Errors**: Install required audio libraries: `pip install librosa soundfile`

3. **API Authentication Errors**:
   - Check if HUGGINGFACE_TOKEN environment variable is set
   - Verify token has write permissions

4. **Memory Issues**:
   - Reduce batch size or max_samples
   - Use model with fewer parameters (e.g., whisper-tiny)

5. **Performance Issues**:
   - **Slow Training**: Consider using smaller dataset subsets for initial testing
   - **High WER**: Ensure dataset quality and sufficient training samples
   - **Model Loading Issues**: Verify model files are not corrupted and paths are correct

## License

This project uses the LibriSpeech dataset, which is licensed under CC BY 4.0.

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- Hugging Face for model hosting and fine-tuning API
- LibriSpeech dataset creators and contributors
