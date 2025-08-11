"""
want to push lateest changes
Dataset processor for LibriSpeech dataset.
Prepares the dataset for fine-tuning by loading local files.
"""
import os
import json
import librosa
from tqdm import tqdm

class LibriSpeechProcessor:
    def __init__(self, dataset_path):
        """
        Initialize the LibriSpeech dataset processor.
        
        Args:
            dataset_path (str): Path to the LibriSpeech dataset directory
        """
        self.dataset_path = dataset_path
        self.samples = []
    
    def process_speaker_directory(self, speaker_dir):
        """
        Process all chapter directories within a speaker directory.
        
        Args:
            speaker_dir (str): Path to the speaker directory
        """
        try:
            for chapter_name in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_name)
                if os.path.isdir(chapter_dir):
                    self.process_chapter_directory(chapter_dir)
        except FileNotFoundError:
            print(f"Warning: Speaker directory not found: {speaker_dir}")
        except PermissionError:
            print(f"Warning: Permission denied when accessing: {speaker_dir}")
        except Exception as e:
            print(f"Error processing speaker directory {speaker_dir}: {str(e)}")
    
    def process_chapter_directory(self, chapter_dir):
        """
        Process all audio files within a chapter directory.
        
        Args:
            chapter_dir (str): Path to the chapter directory
        """
        try:
            # LibriSpeech format:
            # - Each .flac file has its own transcription line in the .txt file
            # - Filename format: speaker-chapter-utterance.flac (e.g., 1088-134315-0000.flac)
            # - Transcript format: utterance_id transcript_text
            
            # Get all flac files
            flac_files = [f for f in os.listdir(chapter_dir) if f.endswith('.flac')]
            
            if not flac_files:
                return
                
            # Get all txt files (usually just one per chapter)
            txt_files = [f for f in os.listdir(chapter_dir) if f.endswith('.txt')]
            
            if not txt_files:
                return
                
            # Parse transcript file to map utterance IDs to transcripts
            transcripts = {}
            for txt_file in txt_files:
                txt_path = os.path.join(chapter_dir, txt_file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                utterance_id = parts[0]
                                text = parts[1]
                                transcripts[utterance_id] = text
                except Exception as e:
                    print(f"Warning: Could not read transcript file {txt_path}: {str(e)}")
            
            # Process each audio file
            for flac_file in flac_files:
                try:
                    # Extract utterance ID from filename (e.g., 1088-134315-0000.flac -> 1088-134315-0000)
                    utterance_id = flac_file.rsplit('.', 1)[0]
                    audio_path = os.path.join(chapter_dir, flac_file)
                    
                    # Check if we have a transcript for this utterance
                    if utterance_id in transcripts:
                        transcript = transcripts[utterance_id]
                        
                        # Load audio and get duration
                        try:
                            # Load audio with librosa (resamples to 16kHz by default)
                            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
                            duration = len(audio_array) / sampling_rate
                            
                            self.samples.append({
                                'audio': {
                                    'array': audio_array.tolist(),  # Convert numpy array to list
                                    'sampling_rate': sampling_rate
                                },
                                'text': transcript,
                                'duration': duration,
                                'audio_path': audio_path  # Keep path for reference
                            })
                        except Exception as e:
                            print(f"Warning: Could not load audio for {audio_path}: {str(e)}")
                except Exception as e:
                    print(f"Warning: Could not process audio file {flac_file}: {str(e)}")
        except FileNotFoundError:
            print(f"Warning: Chapter directory not found: {chapter_dir}")
        except PermissionError:
            print(f"Warning: Permission denied when accessing: {chapter_dir}")
        except Exception as e:
            print(f"Error processing chapter directory {chapter_dir}: {str(e)}")
    
    def prepare_dataset(self, max_samples=None, min_duration=1.0, max_duration=20.0):
        """
        Prepare the dataset by processing all speaker directories.
        
        Args:
            max_samples (int): Maximum number of samples to include
            min_duration (float): Minimum duration of audio files to include (in seconds)
            max_duration (float): Maximum duration of audio files to include (in seconds)
            
        Returns:
            list: List of processed samples
        """
        print(f"Processing LibriSpeech dataset from {self.dataset_path}...")
        
        # Get all speaker directories
        speaker_dirs = [d for d in os.listdir(self.dataset_path) 
                       if os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()]
        
        print(f"Found {len(speaker_dirs)} speaker directories to process")
        
        for idx, speaker_name in enumerate(tqdm(speaker_dirs, desc="Processing speakers")):
            speaker_dir = os.path.join(self.dataset_path, speaker_name)
            self.process_speaker_directory(speaker_dir)
                
            # Check if we've reached the maximum number of samples
            if max_samples and len(self.samples) >= max_samples:
                print(f"Reached maximum number of samples ({max_samples}). Stopping dataset processing.")
                break
        
        # Filter samples by duration
        filtered_samples = []
        for sample in self.samples:
            if min_duration <= sample['duration'] <= max_duration:
                filtered_samples.append(sample)
        
        print(f"Processed {len(self.samples)} samples total. "
              f"{len(filtered_samples)} samples after duration filtering.")
        
        # Apply max_samples limit if specified
        if max_samples and len(filtered_samples) > max_samples:
            filtered_samples = filtered_samples[:max_samples]
            print(f"Limited to {max_samples} samples.")
        
        self.samples = filtered_samples
        return self.samples
    
    def export_manifest(self, output_path, include_audio_arrays=False):
        """
        Export the dataset as a manifest file.
        
        Args:
            output_path (str): Path to save the manifest file
            include_audio_arrays (bool): Whether to include audio arrays in the manifest
                                       (can make files very large)
        """
        export_samples = self.samples.copy()
        
        # Optionally remove audio arrays to reduce file size
        if not include_audio_arrays:
            for sample in export_samples:
                if 'audio' in sample:
                    # Keep structure but remove large array data
                    sample['audio'] = {
                        'sampling_rate': sample['audio']['sampling_rate'],
                        'array_length': len(sample['audio']['array'])
                    }
        
        with open(output_path, 'w') as f:
            json.dump(export_samples, f, indent=2)
        print(f"Exported {len(export_samples)} samples to {output_path}")
        return output_path
    
    def preprocess_for_training(self, processor):
        """
        Preprocess samples for training with WhisperProcessor.
        
        Args:
            processor: WhisperProcessor instance
            
        Returns:
            list: Preprocessed samples ready for training
        """
        preprocessed_samples = []
        
        print(f"Preprocessing {len(self.samples)} samples for training...")
        
        for sample in tqdm(self.samples, desc="Preprocessing audio"):
            try:
                audio = sample["audio"]
                # Process audio to get input features
                input_features = processor(
                    audio["array"], 
                    sampling_rate=audio["sampling_rate"], 
                    return_tensors="pt"
                ).input_features[0]
                
                # Tokenize text to get labels
                labels = processor.tokenizer(sample["text"]).input_ids
                
                preprocessed_sample = {
                    "input_features": input_features,
                    "labels": labels,
                    "text": sample["text"],  # Keep original text for reference
                    "duration": sample["duration"]
                }
                
                preprocessed_samples.append(preprocessed_sample)
                
            except Exception as e:
                print(f"Warning: Could not preprocess sample: {str(e)}")
                continue
        
        print(f"Successfully preprocessed {len(preprocessed_samples)} samples")
        return preprocessed_samples
